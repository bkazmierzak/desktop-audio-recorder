"""Editing operations on ``float32`` ``[frames, channels]`` sample arrays (pure numpy, no Qt).

Two layers:

* **Clip functions** (``delete``, ``crop``, ``insert_silence``, ``silence``, ``gain``,
  ``fade_in``, ``fade_out``, ``normalize``) take the whole array and a frame range
  ``[start, stop)`` and return a *new* array; the input is never modified.
* **Region transforms** (``silence_region``, ``gain_region``, ``fade_region``) modify the array
  they are given *in place*. The undo commands (``editing/commands.py``) run them on a slice of
  the live clip, so an edit costs O(selection) instead of copying the whole clip.

Values above 1.0 (overs from the Windows float mix) are kept: nothing here clips. Gain can push
samples over 1.0 and ``normalize`` uses the real (unclipped) sample peak, so it also brings overs
back under full scale.

Large regions are processed in chunks so no temporary is larger than ``CHUNK_FRAMES`` frames
(a fade over an hour of audio would otherwise allocate a 0.7 GB ramp). New arrays come from
``audio.buffers`` so that dropping them later doesn't stall the GUI.
"""

from __future__ import annotations

import math

import numpy as np

from desktop_audio_recorder.audio import buffers

CHUNK_FRAMES = 1 << 20
NORMALIZE_TARGET_DB = -1.0


# --------------------------------------------------------------------------- helpers


def check_range(frames: int, start: int, stop: int) -> tuple[int, int]:
    """Validate ``0 <= start <= stop <= frames``; return the range as ints."""
    start, stop = int(start), int(stop)
    if not 0 <= start <= stop <= frames:
        raise ValueError(f"range [{start}, {stop}) is outside [0, {frames}]")
    return start, stop


def db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


def gain_to_db(gain: float) -> float:
    return 20.0 * math.log10(gain) if gain > 0 else -math.inf


def peak(samples: np.ndarray, start: int = 0, stop: int | None = None) -> float:
    """Largest absolute value in ``samples[start:stop]`` (overs included), without a temporary
    array (``np.abs`` would copy the whole range)."""
    stop = samples.shape[0] if stop is None else stop
    start, stop = check_range(samples.shape[0], start, stop)
    if stop == start:
        return 0.0
    region = samples[start:stop]
    return float(max(region.max(), -region.min()))


def normalize_factor(current_peak: float, target_db: float = NORMALIZE_TARGET_DB) -> float | None:
    """Linear gain that brings ``current_peak`` to ``target_db`` dBFS, or ``None`` for silence."""
    if not math.isfinite(current_peak) or current_peak <= 0.0:
        return None
    return db_to_gain(target_db) / current_peak


# --------------------------------------------------------------------------- region transforms


def silence_region(region: np.ndarray) -> np.ndarray:
    """Zero ``region`` in place."""
    region[...] = 0.0
    return region


def gain_region(region: np.ndarray, factor: float) -> np.ndarray:
    """Multiply ``region`` by a linear ``factor`` in place (float32 arithmetic)."""
    np.multiply(region, np.float32(factor), out=region)
    return region


def fade_ramp(start: int, stop: int, length: int, fade_in: bool) -> np.ndarray:
    """Gains for frames ``[start, stop)`` of a linear fade ``length`` frames long.

    A fade in goes ``0, 1/n, …, (n-1)/n`` (it ends one step short of unity, where the audio after
    it continues); a fade out is its mirror image, ``(n-1)/n, …, 1/n, 0``.
    """
    idx = np.arange(start, stop, dtype=np.float64)
    if not fade_in:
        idx = (length - 1) - idx
    return (idx / length).astype(np.float32)


def fade_region(region: np.ndarray, fade_in: bool) -> np.ndarray:
    """Apply a linear fade in (or out) across ``region``, in place."""
    n = region.shape[0]
    for i in range(0, n, CHUNK_FRAMES):
        j = min(n, i + CHUNK_FRAMES)
        ramp = fade_ramp(i, j, n, fade_in)
        if region.ndim == 2:
            ramp = ramp[:, np.newaxis]
        np.multiply(region[i:j], ramp, out=region[i:j])
    return region


# --------------------------------------------------------------------------- clip functions


def delete(samples: np.ndarray, start: int, stop: int) -> np.ndarray:
    """The clip without frames ``[start, stop)``."""
    start, stop = check_range(samples.shape[0], start, stop)
    out = buffers.empty((samples.shape[0] - (stop - start),) + samples.shape[1:], samples.dtype)
    out[:start] = samples[:start]
    out[start:] = samples[stop:]
    return out


def crop(samples: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Only frames ``[start, stop)`` (a copy)."""
    start, stop = check_range(samples.shape[0], start, stop)
    return buffers.copy(samples[start:stop])


def insert(samples: np.ndarray, at: int, segment: np.ndarray) -> np.ndarray:
    """The clip with ``segment`` inserted before frame ``at``."""
    at, _ = check_range(samples.shape[0], at, at)
    n = segment.shape[0]
    out = buffers.empty((samples.shape[0] + n,) + samples.shape[1:], samples.dtype)
    out[:at] = samples[:at]
    out[at : at + n] = segment
    out[at + n :] = samples[at:]
    return out


def insert_silence(samples: np.ndarray, at: int, frames: int) -> np.ndarray:
    """The clip with ``frames`` frames of silence inserted before frame ``at``."""
    if frames < 0:
        raise ValueError(f"cannot insert {frames} frames")
    at, _ = check_range(samples.shape[0], at, at)
    out = buffers.empty((samples.shape[0] + frames,) + samples.shape[1:], samples.dtype)
    out[:at] = samples[:at]
    out[at : at + frames] = 0.0
    out[at + frames :] = samples[at:]
    return out


def _transformed(samples: np.ndarray, start: int, stop: int, transform) -> np.ndarray:
    start, stop = check_range(samples.shape[0], start, stop)
    out = buffers.copy(samples)
    if stop > start:
        transform(out[start:stop])
    return out


def silence(samples: np.ndarray, start: int, stop: int) -> np.ndarray:
    """A copy with frames ``[start, stop)`` set to zero."""
    return _transformed(samples, start, stop, silence_region)


def gain(samples: np.ndarray, start: int, stop: int, db: float) -> np.ndarray:
    """A copy with ``db`` decibels of gain on ``[start, stop)`` (no clipping)."""
    factor = db_to_gain(db)
    return _transformed(samples, start, stop, lambda r: gain_region(r, factor))


def fade_in(samples: np.ndarray, start: int, stop: int) -> np.ndarray:
    """A copy with a linear fade in across ``[start, stop)``."""
    return _transformed(samples, start, stop, lambda r: fade_region(r, fade_in=True))


def fade_out(samples: np.ndarray, start: int, stop: int) -> np.ndarray:
    """A copy with a linear fade out across ``[start, stop)``."""
    return _transformed(samples, start, stop, lambda r: fade_region(r, fade_in=False))


def normalize(
    samples: np.ndarray, start: int, stop: int, target_db: float = NORMALIZE_TARGET_DB
) -> np.ndarray:
    """A copy with ``[start, stop)`` scaled so its sample peak is ``target_db`` dBFS.

    Uses the true (unclipped) peak, so a range with overs is turned down. Silence is unchanged.
    """
    start, stop = check_range(samples.shape[0], start, stop)
    factor = normalize_factor(peak(samples, start, stop), target_db)
    if factor is None:
        return buffers.copy(samples)
    return _transformed(samples, start, stop, lambda r: gain_region(r, factor))
