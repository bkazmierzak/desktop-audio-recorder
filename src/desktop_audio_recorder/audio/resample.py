"""Streaming sample-rate conversion (no Qt): ``StreamResampler``.

The recorder uses it when the recording moves to a device that runs at another rate than the
file (the file keeps the rate of the device the recording started on). Blocks of any size go
in, converted blocks come out, with no clicks at block edges.

The filter is the one ``scipy.signal.resample_poly`` designs by default (``firwin`` with
``10 * max(up, down)`` taps per side, Kaiser window, beta 5), applied as a polyphase sum, and
the output is aligned the same way: feeding a stream in pieces and calling ``flush()`` gives
the same samples as ``resample_poly`` on the whole stream (within float32 rounding), and
``ceil(n * up / down)`` of them.

Output ``k`` is ``sum_i x[i] * h[k*down + half - i*up]``: it needs input up to
``(k*down + half) // up``, about ``half / up`` input frames (10 at most) of look-ahead, so a
block's last few output frames come with the next block.
"""

from __future__ import annotations

import math

import numpy as np


class StreamResampler:
    """Convert a stream of ``[frames, channels]`` float32 blocks from ``rate_in`` to
    ``rate_out`` (see the module docstring)."""

    def __init__(self, rate_in: int, rate_out: int, channels: int = 2) -> None:
        from scipy.signal import firwin

        if rate_in <= 0 or rate_out <= 0:
            raise ValueError(f"rates must be positive, got {rate_in} -> {rate_out}")
        g = math.gcd(int(rate_in), int(rate_out))
        self.rate_in, self.rate_out = int(rate_in), int(rate_out)
        self.up, self.down = self.rate_out // g, self.rate_in // g
        self.channels = channels
        max_rate = max(self.up, self.down)
        self.half = 10 * max_rate
        if self.up == self.down:  # same rate: blocks pass through
            h = np.ones(1)
        else:
            h = firwin(2 * self.half + 1, 1.0 / max_rate, window=("kaiser", 5.0)) * self.up
        self.taps = -(-len(h) // self.up) + 1  # input frames per output frame (upper bound)
        padded = np.zeros(self.up * self.taps)
        padded[: len(h)] = h
        # phases[p, t] = h[p + up * t]: the taps for an output whose phase is p.
        self._phases = padded.reshape(self.taps, self.up).T.astype(np.float32).copy()
        # Input history: frames from absolute index ``_base`` on (zeros before the start).
        self._buf = np.zeros((self.taps, channels), np.float32)
        self._base = -self.taps
        self._next = 0  # next output frame
        self.frames_in = 0
        self.frames_out = 0
        self._flushed = False

    @property
    def identity(self) -> bool:
        return self.up == self.down

    def _ready(self, available: int) -> int:
        """Outputs below this index can be computed from ``available`` input frames."""
        # k is ready when (k*down + half) // up <= available - 1, i.e. k*down + half < avail*up.
        return max(0, -(-(available * self.up - self.half) // self.down))

    def _compute(self, stop: int) -> np.ndarray:
        if stop <= self._next:
            return np.zeros((0, self.channels), np.float32)
        ks = np.arange(self._next, stop, dtype=np.int64)
        num = ks * self.down + self.half
        newest = num // self.up  # the newest input frame each output uses
        phase = num - newest * self.up
        idx = newest[:, None] - np.arange(self.taps)[None, :] - self._base
        frames = self._buf[idx]  # [k, taps, channels]
        out = np.einsum("kt,ktc->kc", self._phases[phase], frames, dtype=np.float32)
        self._next = stop
        # Drop history no later output needs.
        oldest_needed = (stop * self.down + self.half) // self.up - (self.taps - 1)
        drop = oldest_needed - self._base
        if drop > 0:
            self._buf = self._buf[drop:]
            self._base += drop
        return out.astype(np.float32, copy=False)

    def process(self, block: np.ndarray) -> np.ndarray:
        """Feed ``block`` (``[frames, channels]``); return the output frames now complete."""
        if self._flushed:
            raise RuntimeError("process() after flush()")
        block = np.asarray(block, dtype=np.float32)
        if block.ndim != 2 or block.shape[1] != self.channels:
            raise ValueError(f"expected [frames, {self.channels}], got {block.shape}")
        if self.identity:
            self.frames_in += len(block)
            self.frames_out += len(block)
            return block
        self._buf = np.concatenate([self._buf, block], axis=0)
        self.frames_in += len(block)
        out = self._compute(self._ready(self.frames_in))
        self.frames_out += len(out)
        return out

    def flush(self) -> np.ndarray:
        """End of the stream: the remaining output, up to ``ceil(frames_in * up / down)``."""
        if self._flushed:
            return np.zeros((0, self.channels), np.float32)
        self._flushed = True
        if self.identity:
            return np.zeros((0, self.channels), np.float32)
        total = -(-self.frames_in * self.up // self.down)
        # Zeros after the end (as resample_poly pads) until every remaining output is ready.
        needed = ((total - 1) * self.down + self.half) // self.up + 1 if total else 0
        pad = max(0, needed - self.frames_in)
        if pad:
            self._buf = np.concatenate([self._buf, np.zeros((pad, self.channels), np.float32)])
        out = self._compute(max(self._next, total))
        self.frames_out += len(out)
        return out


__all__ = ["StreamResampler"]
