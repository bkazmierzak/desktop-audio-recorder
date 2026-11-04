"""Undo/redo: the ``Command`` interface and a memory-bounded ``EditHistory`` (no Qt).

A command turns one samples array into the next (``apply``) and back (``revert``). It keeps only
what it needs to undo itself (a deleted segment, the original samples of a gained region, …),
never a copy of the whole clip unless the edit replaces the whole clip. Commands may modify the
array they are given in place (length-preserving edits do) or return a new one (edits that change
the length do); either way the caller must use the returned array from then on.

Bit-exact round trips: ``revert`` restores the exact samples ``apply`` saw, and a redo calls
``apply`` again on those same samples, so it produces the same result again.

Memory: ``EditHistory`` keeps the total ``nbytes`` of its commands under ``max_bytes`` by dropping
the oldest undo entries. The newest entry is always kept, even when it alone is over the limit,
so the last edit can always be undone. Room is made *before* a new command stores its undo data
(``check()`` first, so an edit that would change nothing drops nothing), which keeps the peak
lower: a whole-clip gain on an hour-long clip copies 1.3 GB, and the previous whole-clip copy is
gone by then instead of being dropped afterwards.

Disk: with a ``SpillStore`` (``spill=``), going over ``max_bytes`` first moves the oldest large
undo buffers (``Payload``, see ``spill.py``) to temporary files, and drops entries only when the
store can't take more (its disk budget, or low free space; dropping the oldest entry frees its
file, so the files work as a queue). The newest entry stays in memory unless a new one is about
to replace it as the newest. Writing happens on the store's thread; an edit that is about to
allocate a large array waits only while more than one write is pending (it runs off the GUI
thread anyway).
Undoing a spilled entry reads its file, which ``work_frames`` counts, so the GUI runs such an
undo in the background.
"""

from __future__ import annotations

import itertools
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

import numpy as np

from desktop_audio_recorder.audio import buffers
from desktop_audio_recorder.editing.spill import Payload, SpillStore

log = logging.getLogger(__name__)

DEFAULT_MAX_BYTES = 500 * 1024 * 1024
DISK_READ_COST = 4  # reading a frame from an undo file costs about as much as 4 in memory


class NoChange(Exception):  # noqa: N818 (not an error: the edit would do nothing)
    """Raised by ``Command.apply`` (before touching anything) when the edit would change nothing.

    The message says why, for the status bar.
    """


@dataclass(frozen=True, slots=True)
class ViewHint:
    """Where the selection and cursor should be after an edit (frames in the new clip).

    ``cursor`` ``None`` keeps the current cursor (clamped to the new length). ``shift`` moves the
    visible range, in frames, so the audio that stays doesn't jump on screen (e.g. after a crop).
    """

    selection: tuple[int, int] | None = None
    cursor: int | None = None
    shift: int = 0


class Command(ABC):
    """One undoable edit."""

    label: str = "Edit"
    #: Changes the clip length (needs a new array and a full waveform rebuild).
    structural: bool = False

    def check(self, samples: np.ndarray) -> None:
        """Validate against ``samples`` before anything is stored; raise ``NoChange`` (or
        ``ValueError``) if the edit can't or needn't happen. ``apply`` must do the same checks.
        The default accepts everything."""
        return None

    def undo_bytes(self, samples: np.ndarray) -> int:
        """What ``nbytes`` will be after applying to ``samples`` (for making room first)."""
        return 0

    @abstractmethod
    def apply(self, samples: np.ndarray) -> np.ndarray:
        """Do (or redo) the edit; return the new samples. May raise ``NoChange``."""

    @abstractmethod
    def revert(self, samples: np.ndarray) -> np.ndarray:
        """Undo the edit; return the previous samples (bit-exact)."""

    def payloads(self) -> tuple[Payload, ...]:
        """The undo buffers this command holds (they may be moved to disk)."""
        return ()

    @property
    def nbytes(self) -> int:
        """Memory held for undo/redo."""
        return sum(p.nbytes for p in self.payloads())

    @property
    def disk_bytes(self) -> int:
        """Undo data held in temporary files."""
        return sum(p.disk_bytes for p in self.payloads())

    @property
    def disk_frames(self) -> int:
        return sum(p.frames for p in self.payloads() if p.on_disk)

    def discard(self) -> None:
        """The command is leaving the history: drop its undo data (memory and files)."""
        for p in self.payloads():
            p.discard()

    @property
    def region(self) -> tuple[int, int]:
        """Frames changed by a length-preserving edit (the same before and after)."""
        return (0, 0)

    def work_frames(self, total_frames: int) -> int:
        """Rough cost of applying or reverting, in frames touched (incl. the waveform rebuild
        and reading undo data back from disk)."""
        if self.structural:
            work = total_frames
        else:
            a, b = self.region
            work = b - a
        return work + DISK_READ_COST * self.disk_frames

    def hint(self, applied: bool) -> ViewHint:
        """Selection/cursor after ``apply`` (``applied``) or after ``revert``."""
        return ViewHint()


def _settle_before(
    command: Command, samples: np.ndarray, reserve: int = 0, spill: SpillStore | None = None
) -> None:
    """Before a command allocates a large array (a clip-sized result, or a big undo copy), wait
    for large buffers already dropped (an evicted entry, the clip before the last edit) to be
    freed by the background thread, and while more than one undo buffer is on its way to disk,
    so the old and new memory aren't committed at once. Free when nothing is pending; only large
    edits get here, and those run off the GUI thread."""
    allocates = max(reserve, samples.nbytes if command.structural else 0)
    if allocates >= buffers.LARGE_BYTES:
        if spill is not None:
            spill.wait_pending(1)
        buffers.wait_released()


class EditHistory:
    """Linear undo/redo stacks with a memory cap (see module docstring)."""

    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES, spill: SpillStore | None = None) -> None:
        self.max_bytes = max_bytes
        self.spill = spill
        self.spilled = 0  # payloads moved to disk
        self._undo: deque[Command] = deque()
        self._redo: list[Command] = []
        self.evicted = 0  # undo entries dropped to stay under the cap
        # ``position``: each command gets a serial number when it is first recorded; the base
        # (the content with an empty undo stack) is the serial of the newest evicted entry.
        self._serials = itertools.count(1)
        self._serial: dict[int, int] = {}  # id(command) -> serial, for recorded commands
        self._base = 0

    # ------------------------------------------------------------------ queries

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def next_undo(self) -> Command | None:
        return self._undo[-1] if self._undo else None

    @property
    def next_redo(self) -> Command | None:
        return self._redo[-1] if self._redo else None

    @property
    def undo_label(self) -> str | None:
        return self._undo[-1].label if self._undo else None

    @property
    def redo_label(self) -> str | None:
        return self._redo[-1].label if self._redo else None

    @property
    def undo_count(self) -> int:
        return len(self._undo)

    @property
    def redo_count(self) -> int:
        return len(self._redo)

    @property
    def nbytes(self) -> int:
        """Undo data held in memory."""
        return sum(c.nbytes for c in self._undo) + sum(c.nbytes for c in self._redo)

    @property
    def disk_bytes(self) -> int:
        """Undo data held in temporary files."""
        return sum(c.disk_bytes for c in self._undo) + sum(c.disk_bytes for c in self._redo)

    @property
    def position(self) -> int:
        """Identifies the current content of the clip: equal positions mean equal samples.

        It is the serial number of the newest applied command (0 for the original clip, or the
        newest evicted command's when the undo stack is empty). Undoing back to a position
        returns it; a new edit after an undo gets a new one. Used to tell whether the clip has
        changed since it was saved.
        """
        return self._serial[id(self._undo[-1])] if self._undo else self._base

    # ------------------------------------------------------------------ editing

    def execute(self, command: Command, samples: np.ndarray) -> np.ndarray:
        """Apply ``command`` and record it. ``NoChange`` (or any error) records nothing.

        After ``command.check()`` passes, the redo stack is dropped (the new edit replaces it)
        and old entries are evicted to fit the command's undo data, before it is applied.
        """
        command.check(samples)
        reserve = command.undo_bytes(samples)
        self._drop_redo()
        self._trim(reserve=reserve)
        _settle_before(command, samples, reserve, self.spill)
        result = command.apply(samples)
        self.push(command)
        return result

    def push(self, command: Command) -> None:
        """Record an already applied command (clears the redo stack)."""
        self._drop_redo()
        self._serial[id(command)] = next(self._serials)
        self._undo.append(command)
        self._trim()

    def undo(self, samples: np.ndarray) -> tuple[np.ndarray, Command]:
        """Revert the newest command. Raises ``NoChange`` when there is nothing to undo.

        If ``revert`` raises, the history is unchanged.
        """
        if not self._undo:
            raise NoChange("Nothing to undo")
        command = self._undo[-1]
        _settle_before(command, samples, spill=self.spill)
        result = command.revert(samples)
        self._undo.pop()
        self._redo.append(command)
        return result, command

    def redo(self, samples: np.ndarray) -> tuple[np.ndarray, Command]:
        """Re-apply the newest undone command. Raises ``NoChange`` when there is none."""
        if not self._redo:
            raise NoChange("Nothing to redo")
        command = self._redo[-1]
        _settle_before(command, samples, spill=self.spill)
        result = command.apply(samples)
        self._redo.pop()
        self._undo.append(command)
        return result, command

    def clear(self) -> None:
        """Forget every entry (the clip keeps its content, and ``position``)."""
        self._base = self.position
        for command in (*self._undo, *self._redo):
            command.discard()
        self._undo.clear()
        self._redo.clear()
        self._serial.clear()

    def _drop_redo(self) -> None:
        for command in self._redo:
            self._serial.pop(id(command), None)
            command.discard()
        self._redo.clear()

    def _trim(self, reserve: int = 0) -> int:
        """Get the undo data in memory under the cap (with ``reserve`` bytes to spare): move the
        oldest large buffers to disk if there is a store, and drop the oldest entries when that
        isn't enough or possible. With ``reserve`` (a command about to be pushed) every entry
        may go; without, the newest stays (in memory, too). Returns the bytes dropped."""
        keep = 0 if reserve else 1
        dropped_bytes = 0
        total = self.nbytes + reserve
        while total > self.max_bytes:
            payload = self._spill_candidate(keep)
            if payload is not None and payload.spill(self.spill):
                total -= payload.size_bytes
                self.spilled += 1
                continue
            if len(self._undo) <= keep:
                break
            if payload is None and not any(
                c.nbytes for c in list(self._undo)[: len(self._undo) - keep]
            ):
                break  # what is left in memory is in the newest entry (or the reserve)
            # A spill was refused (disk budget, low space) or the memory is in buffers too
            # small to spill: drop the oldest entry. If it was on disk, that makes room there
            # for the next candidate.
            dropped = self._undo.popleft()
            self._base = self._serial.pop(id(dropped))
            total -= dropped.nbytes
            self.evicted += 1
            dropped_bytes += dropped.nbytes + dropped.disk_bytes
            log.info(
                "Undo history over %.0f MB: dropped the oldest edit (%s, %.1f MB in memory,"
                " %.1f MB on disk)",
                self.max_bytes / 2**20,
                dropped.label,
                dropped.nbytes / 2**20,
                dropped.disk_bytes / 2**20,
            )
            dropped.discard()
        return dropped_bytes

    def _spill_candidate(self, keep: int) -> Payload | None:
        """The oldest undo buffer that may move to disk (not in the newest ``keep`` entries)."""
        if self.spill is None:
            return None
        entries = list(self._undo)
        for command in entries[: len(entries) - keep]:
            for payload in command.payloads():
                if payload.spillable(self.spill.min_spill_bytes):
                    return payload
        return None
