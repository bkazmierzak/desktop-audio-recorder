"""``EditSession``: the clip being edited, its waveform pyramid and its undo history (no Qt).

Every ``apply``/``undo``/``redo`` returns an ``EditResult`` with the new clip and an up-to-date
``EnvelopePyramid``: refreshed in place over the changed region for length-preserving edits
(O(region)), rebuilt for edits that change the length (O(clip), about 0.15 s per hour).

The GUI runs small edits directly and big ones (``work_frames`` above a threshold) on a
background thread; a session must only be used by one thread at a time.

Length-preserving edits modify the samples array in place, so an earlier ``AudioClip`` object
that shares the array sees the change too; always use the clip from the latest result.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from desktop_audio_recorder.audio.clip import AudioClip
from desktop_audio_recorder.audio.envelope import EnvelopePyramid
from desktop_audio_recorder.editing.history import (
    DEFAULT_MAX_BYTES,
    Command,
    EditHistory,
    NoChange,
    ViewHint,
)
from desktop_audio_recorder.editing.spill import SpillStore

log = logging.getLogger(__name__)

APPLY, UNDO, REDO = "apply", "undo", "redo"


@dataclass(frozen=True, slots=True)
class EditResult:
    action: str  # APPLY, UNDO or REDO
    command: Command
    clip: AudioClip
    pyramid: EnvelopePyramid
    hint: ViewHint
    seconds: float  # time the edit took

    @property
    def message(self) -> str:
        verb = {APPLY: "", UNDO: "Undid: ", REDO: "Redid: "}[self.action]
        return f"{verb}{self.command.label}"


class EditSession:
    def __init__(
        self,
        clip: AudioClip,
        pyramid: EnvelopePyramid | None = None,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        spill: SpillStore | None = None,
    ) -> None:
        self.clip = clip
        self.pyramid = pyramid if pyramid is not None else EnvelopePyramid(clip.samples)
        self.history = EditHistory(max_bytes, spill=spill)

    def close(self) -> None:
        """Drop the undo history (and its files); the session isn't used after this."""
        self.history.clear()

    # ------------------------------------------------------------------ cost

    def work_frames(self, action: str, command: Command | None = None) -> int:
        """Frames an action touches (0 if there's nothing to undo/redo)."""
        if action == UNDO:
            command = self.history.next_undo
        elif action == REDO:
            command = self.history.next_redo
        if command is None:
            return 0
        return command.work_frames(self.clip.frames)

    # ------------------------------------------------------------------ actions

    def apply(self, command: Command) -> EditResult:
        """Apply and record ``command``. Raises ``NoChange`` (nothing recorded) if it's a no-op."""
        started = time.perf_counter()
        samples = self.history.execute(command, self.clip.samples)
        return self._commit(APPLY, command, samples, True, started)

    def undo(self) -> EditResult:
        started = time.perf_counter()
        samples, command = self.history.undo(self.clip.samples)
        return self._commit(UNDO, command, samples, False, started)

    def redo(self) -> EditResult:
        started = time.perf_counter()
        samples, command = self.history.redo(self.clip.samples)
        return self._commit(REDO, command, samples, True, started)

    def run(self, action: str, command: Command | None = None) -> EditResult:
        if action == APPLY:
            if command is None:
                raise ValueError("apply needs a command")
            return self.apply(command)
        if action == UNDO:
            return self.undo()
        if action == REDO:
            return self.redo()
        raise ValueError(f"unknown action {action!r}")

    # ------------------------------------------------------------------ internals

    def _commit(
        self, action: str, command: Command, samples, applied: bool, started: float
    ) -> EditResult:
        old = self.clip
        if samples is old.samples:
            self.pyramid.refresh(*command.region)
            clip = AudioClip(samples, old.sample_rate, old.source)
        else:
            clip = AudioClip(samples, old.sample_rate, old.source)
            self.pyramid = EnvelopePyramid(clip.samples)
        self.clip = clip
        seconds = time.perf_counter() - started
        log.info(
            "%s %s: %d -> %d frames in %.0f ms (undo memory %.1f MB, disk %.1f MB,"
            " %d undo / %d redo)",
            action,
            command.label,
            old.frames,
            clip.frames,
            seconds * 1000,
            self.history.nbytes / 2**20,
            self.history.disk_bytes / 2**20,
            self.history.undo_count,
            self.history.redo_count,
        )
        return EditResult(action, command, clip, self.pyramid, command.hint(applied), seconds)


__all__ = ["APPLY", "REDO", "UNDO", "EditResult", "EditSession", "NoChange"]
