"""The undoable edits behind the edit toolbar (no Qt). See ``history.py`` for the contract.

Length-preserving edits (silence, gain, fades, normalize) change the live array in place and keep
a copy of the original region for undo. Edits that change the length (delete, crop, insert
silence) return a new array and keep only what the reverse needs:

=================  ===================================  =====================================
Command            Keeps for undo                       Apply / revert
=================  ===================================  =====================================
``Delete``         the removed segment                  concatenate / re-insert it
``Crop``           the previous array (whole clip)      copy of the selection / that array
``InsertSilence``  nothing                              new array with zeros / delete them
``Silence`` etc.   the original region                  transform in place / copy it back
``ReplaceRegion``  one region-sized buffer              swap it with the region (both ways)
=================  ===================================  =====================================

``ReplaceRegion`` carries samples computed beforehand (noise reduction, which is slow and
cancellable, so it never runs inside a command).

Every kept buffer is a ``Payload`` (``spill.py``), so the history can move it to disk; reverting
reads it back (a region straight into the clip, without a temporary).

Frame ranges are ``[start, stop)`` in the clip as it is when the command applies.
"""

from __future__ import annotations

import numpy as np

from desktop_audio_recorder.audio import buffers
from desktop_audio_recorder.editing import operations as ops
from desktop_audio_recorder.editing.history import Command, NoChange, ViewHint
from desktop_audio_recorder.editing.spill import Payload

ALREADY_NORMALIZED_DB = 0.001  # Normalize is a no-op when the peak is this close to the target


def _check_range(samples: np.ndarray, start: int, stop: int) -> None:
    ops.check_range(samples.shape[0], start, stop)


# --------------------------------------------------------------------------- length-preserving


class RegionCommand(Command):
    """Transforms ``[start, stop)`` in place. ``selected``: the range was the user's selection
    (it stays selected); otherwise the edit applied to the whole clip with nothing selected."""

    def __init__(self, start: int, stop: int, *, selected: bool = True) -> None:
        if not 0 <= start < stop:
            raise ValueError(f"empty or invalid range [{start}, {stop})")
        self.start, self.stop = int(start), int(stop)
        self.selected = selected
        self._original: Payload | None = None
        self._checked = False

    @property
    def region(self) -> tuple[int, int]:
        return self.start, self.stop

    def payloads(self) -> tuple[Payload, ...]:
        return () if self._original is None else (self._original,)

    def prepare(self, region: np.ndarray) -> None:
        """Check the edit would change something (raise ``NoChange``) before anything is saved."""

    def transform(self, region: np.ndarray) -> None:
        raise NotImplementedError

    def check(self, samples: np.ndarray) -> None:
        if self._original is None and not self._checked:
            _check_range(samples, self.start, self.stop)
            self.prepare(samples[self.start : self.stop])
            self._checked = True

    def undo_bytes(self, samples: np.ndarray) -> int:
        return (self.stop - self.start) * samples[:1].nbytes

    def apply(self, samples: np.ndarray) -> np.ndarray:
        _check_range(samples, self.start, self.stop)
        region = samples[self.start : self.stop]
        if self._original is None:
            self.check(samples)
            self._original = Payload(buffers.copy(region))
        self.transform(region)
        return samples

    def revert(self, samples: np.ndarray) -> np.ndarray:
        if self._original is None:
            raise RuntimeError(f"{self.label}: revert before apply")
        _check_range(samples, self.start, self.stop)
        self._original.read_into(samples[self.start : self.stop])
        return samples

    def hint(self, applied: bool) -> ViewHint:
        return ViewHint(selection=(self.start, self.stop) if self.selected else None)


class Silence(RegionCommand):
    label = "Silence"

    def transform(self, region: np.ndarray) -> None:
        ops.silence_region(region)


class FadeIn(RegionCommand):
    label = "Fade in"

    def transform(self, region: np.ndarray) -> None:
        ops.fade_region(region, fade_in=True)


class FadeOut(RegionCommand):
    label = "Fade out"

    def transform(self, region: np.ndarray) -> None:
        ops.fade_region(region, fade_in=False)


class Gain(RegionCommand):
    def __init__(self, start: int, stop: int, db: float, *, selected: bool = True) -> None:
        super().__init__(start, stop, selected=selected)
        self.db = float(db)
        self.label = f"Gain {self.db:+.1f} dB"

    def prepare(self, region: np.ndarray) -> None:
        if self.db == 0.0:
            raise NoChange("A gain of 0 dB changes nothing")

    def transform(self, region: np.ndarray) -> None:
        ops.gain_region(region, ops.db_to_gain(self.db))


class Normalize(RegionCommand):
    """Scales the region so its true sample peak (overs included) is ``target_db`` dBFS. The
    factor is measured on the first apply and reused by redo."""

    def __init__(
        self,
        start: int,
        stop: int,
        target_db: float = ops.NORMALIZE_TARGET_DB,
        *,
        selected: bool = True,
    ) -> None:
        super().__init__(start, stop, selected=selected)
        self.target_db = float(target_db)
        self.factor: float | None = None
        self.label = f"Normalize to {self.target_db:g} dBFS"

    def prepare(self, region: np.ndarray) -> None:
        factor = ops.normalize_factor(ops.peak(region), self.target_db)
        if factor is None:
            raise NoChange("Nothing to normalize: the audio is silent")
        if abs(ops.gain_to_db(factor)) < ALREADY_NORMALIZED_DB:
            raise NoChange(f"Already normalized to {self.target_db:g} dBFS")
        self.factor = factor

    def transform(self, region: np.ndarray) -> None:
        assert self.factor is not None
        ops.gain_region(region, self.factor)


SWAP_CHUNK_FRAMES = 1 << 20


class ReplaceRegion(RegionCommand):
    """Replaces ``[start, stop)`` with ``replacement``, samples computed beforehand (noise
    reduction runs for a long time, cancellably, without touching the clip; this command then
    only copies). It keeps a single region-sized buffer and swaps it with the clip: before
    ``apply`` it holds the new samples, afterwards the originals, and ``revert`` swaps back, so
    undo and redo are bit-exact and cost one swap each. The buffer becomes the command's (don't
    use ``replacement`` afterwards)."""

    def __init__(
        self,
        start: int,
        stop: int,
        replacement: np.ndarray,
        label: str = "Replace",
        *,
        selected: bool = True,
    ) -> None:
        super().__init__(start, stop, selected=selected)
        if replacement.ndim != 2 or replacement.shape[0] != stop - start:
            raise ValueError(
                f"replacement of shape {replacement.shape} doesn't fit [{start}, {stop})"
            )
        self._buffer = Payload(replacement)
        self._applied = False
        self.label = label

    def payloads(self) -> tuple[Payload, ...]:
        return (self._buffer,)

    def prepare(self, region: np.ndarray) -> None:
        if region.shape != self._buffer.shape or region.dtype != self._buffer.dtype:
            raise ValueError(
                f"replacement {self._buffer.shape} {self._buffer.dtype} doesn't match the clip"
                f" region {region.shape} {region.dtype}"
            )

    def apply(self, samples: np.ndarray) -> np.ndarray:
        if self._applied:
            raise RuntimeError(f"{self.label}: applied twice")
        _check_range(samples, self.start, self.stop)
        self.check(samples)
        self.prepare(samples[self.start : self.stop])
        _swap(samples[self.start : self.stop], self._buffer.take())
        self._applied = True
        return samples

    def revert(self, samples: np.ndarray) -> np.ndarray:
        if not self._applied:
            raise RuntimeError(f"{self.label}: revert before apply")
        _swap(samples[self.start : self.stop], self._buffer.take())
        self._applied = False
        return samples


def _swap(a: np.ndarray, b: np.ndarray) -> None:
    """Exchange the contents of two equally shaped arrays, chunk by chunk (small temporary)."""
    for i in range(0, a.shape[0], SWAP_CHUNK_FRAMES):
        j = min(a.shape[0], i + SWAP_CHUNK_FRAMES)
        tmp = a[i:j].copy()
        a[i:j] = b[i:j]
        b[i:j] = tmp


# --------------------------------------------------------------------------- length-changing


class Delete(Command):
    label = "Delete"
    structural = True

    def __init__(self, start: int, stop: int) -> None:
        if not 0 <= start < stop:
            raise ValueError(f"empty or invalid range [{start}, {stop})")
        self.start, self.stop = int(start), int(stop)
        self._removed: Payload | None = None

    def payloads(self) -> tuple[Payload, ...]:
        return () if self._removed is None else (self._removed,)

    def check(self, samples: np.ndarray) -> None:
        _check_range(samples, self.start, self.stop)

    def undo_bytes(self, samples: np.ndarray) -> int:
        return (self.stop - self.start) * samples[:1].nbytes

    def apply(self, samples: np.ndarray) -> np.ndarray:
        _check_range(samples, self.start, self.stop)
        result = ops.delete(samples, self.start, self.stop)
        if self._removed is None:
            self._removed = Payload(buffers.copy(samples[self.start : self.stop]))
        return result

    def revert(self, samples: np.ndarray) -> np.ndarray:
        if self._removed is None:
            raise RuntimeError("Delete: revert before apply")
        removed = self._removed
        ops.check_range(samples.shape[0], self.start, self.start)
        n = removed.frames
        out = buffers.empty((samples.shape[0] + n,) + samples.shape[1:], samples.dtype)
        out[: self.start] = samples[: self.start]
        removed.read_into(out[self.start : self.start + n])  # from memory or straight from disk
        out[self.start + n :] = samples[self.start :]
        return out

    def hint(self, applied: bool) -> ViewHint:
        if applied:
            return ViewHint(selection=None, cursor=self.start)
        return ViewHint(selection=(self.start, self.stop), cursor=self.start)


class Crop(Command):
    """Keeps only ``[start, stop)``. Undo needs everything else, so it keeps the previous array
    itself (no copy; it isn't modified while this command is in the history, because every
    later edit is undone before this one is). Undo hands that array back as the live clip
    (``Payload.take``: read back first if it was moved to disk)."""

    label = "Crop"
    structural = True

    def __init__(self, start: int, stop: int) -> None:
        if not 0 <= start < stop:
            raise ValueError(f"empty or invalid range [{start}, {stop})")
        self.start, self.stop = int(start), int(stop)
        self._previous: Payload | None = None

    def payloads(self) -> tuple[Payload, ...]:
        return () if self._previous is None else (self._previous,)

    def check(self, samples: np.ndarray) -> None:
        _check_range(samples, self.start, self.stop)
        if self.start == 0 and self.stop == samples.shape[0]:
            raise NoChange("The selection already covers the whole clip")

    def undo_bytes(self, samples: np.ndarray) -> int:
        return samples.nbytes

    def apply(self, samples: np.ndarray) -> np.ndarray:
        self.check(samples)
        result = ops.crop(samples, self.start, self.stop)
        self._previous = Payload(samples)
        return result

    def revert(self, samples: np.ndarray) -> np.ndarray:
        if self._previous is None:
            raise RuntimeError("Crop: revert before apply")
        return self._previous.take()

    def hint(self, applied: bool) -> ViewHint:
        if applied:
            return ViewHint(selection=None, cursor=0, shift=-self.start)
        return ViewHint(selection=(self.start, self.stop), cursor=self.start, shift=self.start)


class InsertSilence(Command):
    structural = True

    def __init__(self, at: int, frames: int, sample_rate: int | None = None) -> None:
        if at < 0 or frames <= 0:
            raise ValueError(f"invalid insert of {frames} frames at {at}")
        self.at, self.frames = int(at), int(frames)
        self.label = (
            f"Insert {frames / sample_rate:g} s of silence" if sample_rate else "Insert silence"
        )

    def check(self, samples: np.ndarray) -> None:
        _check_range(samples, self.at, self.at)

    def apply(self, samples: np.ndarray) -> np.ndarray:
        self.check(samples)
        return ops.insert_silence(samples, self.at, self.frames)

    def revert(self, samples: np.ndarray) -> np.ndarray:
        return ops.delete(samples, self.at, self.at + self.frames)

    def hint(self, applied: bool) -> ViewHint:
        if applied:
            return ViewHint(selection=(self.at, self.at + self.frames), cursor=self.at)
        return ViewHint(selection=None, cursor=self.at)
