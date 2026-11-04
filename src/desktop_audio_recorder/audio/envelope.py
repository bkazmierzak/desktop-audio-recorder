"""Min/max waveform envelopes for drawing (pure numpy, no Qt).

Drawing an hour of 48 kHz audio means summarising ~173 M frames into ~2000 on-screen bins.
Scanning the raw samples on every zoom or scroll would take hundreds of milliseconds, so
``EnvelopePyramid`` precomputes min/max per block of ``base_block`` frames (level 0), then halves
the resolution level by level. A query picks the coarsest level that still has at least
``bins`` blocks in the visible range and reduces only that slice. Ranges short enough
(fewer than ``bins * base_block`` frames) are reduced straight from the samples, so the
deepest zoom is exact.

The envelope combines both channels (min over L/R, max over L/R): one lane, and an edit applies
to both channels anyway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_BASE_BLOCK = 256
_BUILD_CHUNK_BLOCKS = 8192  # level-0 blocks reduced per step (keeps the working set in cache)


@dataclass(frozen=True, slots=True)
class Envelope:
    """Per-bin minimum and maximum. ``x`` is each bin's first frame (float64)."""

    x: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    frames_per_bin: int

    def __len__(self) -> int:
        return len(self.x)

    def zigzag(self, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
        """Points ``(t0, lo0), (t0, hi0), (t1, lo1), …`` for a single connected polyline.

        At one bin per pixel column the zigzag fills the area between min and max.
        """
        n = len(self.x)
        t = np.repeat(self.x / sample_rate, 2)
        y = np.empty(2 * n, dtype=np.float32)
        y[0::2] = self.lo
        y[1::2] = self.hi
        return t, y


def _empty() -> Envelope:
    return Envelope(np.array([], dtype=np.float64), _f32(), _f32(), 1)


def _f32() -> np.ndarray:
    return np.array([], dtype=np.float32)


def reduce_bins(lo: np.ndarray, hi: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Group consecutive values into at most ``bins`` bins of equal size (last may be short).

    Returns ``(lo, hi, group)`` where ``group`` is the number of inputs per bin.
    """
    n = len(lo)
    if n == 0:
        return _f32(), _f32(), 1
    group = max(1, math.ceil(n / max(1, bins)))
    if group == 1:
        return lo, hi, 1
    starts = np.arange(0, n, group)
    return np.minimum.reduceat(lo, starts), np.maximum.reduceat(hi, starts), group


def minmax_envelope(samples: np.ndarray, start: int, stop: int, bins: int) -> Envelope:
    """Brute-force envelope of ``samples[start:stop]`` (``[frames]`` or ``[frames, ch]``)."""
    frames = samples.shape[0]
    start, stop = max(0, start), min(frames, stop)
    if stop <= start or bins <= 0:
        return _empty()
    seg = samples[start:stop]
    if seg.ndim == 2:
        # Element-wise over the channel columns: much faster than min(axis=1) on a short axis.
        lo_src, hi_src = seg[:, 0].copy(), seg[:, 0].copy()
        for ch in range(1, seg.shape[1]):
            np.minimum(lo_src, seg[:, ch], out=lo_src)
            np.maximum(hi_src, seg[:, ch], out=hi_src)
    else:
        lo_src = hi_src = seg
    lo, hi, group = reduce_bins(lo_src, hi_src, bins)
    x = start + np.arange(len(lo), dtype=np.float64) * group
    return Envelope(x, lo.astype(np.float32, copy=False), hi.astype(np.float32, copy=False), group)


class EnvelopePyramid:
    """Precomputed min/max levels over a ``[frames, channels]`` float array."""

    def __init__(self, samples: np.ndarray, base_block: int = DEFAULT_BASE_BLOCK) -> None:
        if base_block < 1:
            raise ValueError("base_block must be >= 1")
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]
        self._samples = samples
        self.base_block = base_block
        self.frames = int(samples.shape[0])
        self.levels: list[tuple[np.ndarray, np.ndarray]] = []
        if self.frames:
            self._build()

    @property
    def peak(self) -> float:
        """Largest absolute sample value (from the top level, no scan)."""
        if not self.levels:
            return 0.0
        lo, hi = self.levels[-1]
        return float(max(abs(lo[0]), abs(hi[0])))

    @property
    def nbytes(self) -> int:
        return sum(lo.nbytes + hi.nbytes for lo, hi in self.levels)

    def block_size(self, level: int) -> int:
        return self.base_block << level

    def _build(self) -> None:
        full = self.frames // self.base_block
        count = full + (1 if self.frames > full * self.base_block else 0)
        lo = np.empty(count, dtype=np.float32)
        hi = np.empty(count, dtype=np.float32)
        self.levels.append((lo, hi))
        self._reduce_base(0, count)
        while len(lo) > 1:
            size = (len(lo) + 1) // 2
            lo, hi = np.empty(size, dtype=np.float32), np.empty(size, dtype=np.float32)
            self.levels.append((lo, hi))
            self._reduce_level(len(self.levels) - 1, 0, len(lo))

    def _reduce_base(self, i0: int, i1: int) -> None:
        """Recompute level-0 blocks ``[i0, i1)`` from the samples."""
        samples, base = self._samples, self.base_block
        channels = samples.shape[1]
        lo, hi = self.levels[0]
        full = self.frames // base
        for i in range(i0, min(i1, full), _BUILD_CHUNK_BLOCKS):
            j = min(i1, full, i + _BUILD_CHUNK_BLOCKS)
            # A view for C-contiguous input (the normal case); a chunk-sized copy otherwise.
            chunk = np.ascontiguousarray(samples[i * base : j * base])
            blocks = chunk.reshape(j - i, base * channels)
            np.min(blocks, axis=1, out=lo[i:j])
            np.max(blocks, axis=1, out=hi[i:j])
        if i1 > full:  # the short last block
            rest = samples[full * base :]
            lo[full] = rest.min()
            hi[full] = rest.max()

    def _reduce_level(self, level: int, i0: int, i1: int) -> None:
        """Recompute entries ``[i0, i1)`` of ``level`` from pairs of the level below (an odd last
        entry is copied)."""
        plo, phi = self.levels[level - 1]
        lo, hi = self.levels[level]
        n = len(plo)
        pairs_end = min(i1, n // 2)
        if pairs_end > i0:
            a, b = 2 * i0, 2 * pairs_end
            np.minimum(plo[a:b:2], plo[a + 1 : b : 2], out=lo[i0:pairs_end])
            np.maximum(phi[a:b:2], phi[a + 1 : b : 2], out=hi[i0:pairs_end])
        if n & 1 and i1 > n // 2:
            lo[n // 2], hi[n // 2] = plo[-1], phi[-1]

    def refresh(self, start: int, stop: int) -> None:
        """Update the levels after frames ``[start, stop)`` of the samples changed in place.

        Costs O(stop - start) plus a few entries per level, so a small edit on an hour-long clip
        doesn't need a full rebuild. (An edit that changes the length needs a new pyramid.)
        """
        start, stop = max(0, int(start)), min(self.frames, int(stop))
        if stop <= start or not self.levels:
            return
        i0, i1 = start // self.base_block, -(-stop // self.base_block)
        self._reduce_base(i0, i1)
        for level in range(1, len(self.levels)):
            i0, i1 = i0 // 2, -(-i1 // 2)
            self._reduce_level(level, i0, min(i1, len(self.levels[level][0])))

    def envelope(self, start: int, stop: int, bins: int) -> Envelope:
        """Envelope of frames ``[start, stop)`` with at most ``bins`` bins.

        Bins from a pyramid level are aligned to that level's blocks, so the first and last bin
        may reach up to one block outside the range.
        """
        start, stop = max(0, int(start)), min(self.frames, int(stop))
        if stop <= start or bins <= 0:
            return _empty()
        span = stop - start
        if span < bins * self.base_block or not self.levels:
            return minmax_envelope(self._samples, start, stop, bins)
        level = int(math.floor(math.log2(span / (bins * self.base_block))))
        level = min(max(level, 0), len(self.levels) - 1)
        block = self.block_size(level)
        lo_src, hi_src = self.levels[level]
        i0 = start // block
        i1 = min(len(lo_src), -(-stop // block))
        lo, hi, group = reduce_bins(lo_src[i0:i1], hi_src[i0:i1], bins)
        x = (i0 + np.arange(len(lo), dtype=np.float64) * group) * block
        return Envelope(x, lo, hi, group * block)
