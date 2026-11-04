"""``AudioClip``: the in-memory audio the app plays back and edits.

Samples are ``float32`` shaped ``[frames, 2]`` (stereo, C-contiguous) at the recording's native
sample rate. Values may exceed ±1.0 (the Windows float mix produces overs); they are kept here
and only clipped at the playback output and at integer export.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from desktop_audio_recorder.audio import buffers
from desktop_audio_recorder.audio.downmix import downmix_to_stereo

CHANNELS = 2


@dataclass(frozen=True, eq=False)
class AudioClip:
    samples: np.ndarray
    sample_rate: int
    source: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample rate must be positive, got {self.sample_rate}")
        samples = np.ascontiguousarray(self.samples, dtype=np.float32)
        if samples.ndim != 2 or samples.shape[1] != CHANNELS:
            raise ValueError(f"samples must be shaped [frames, 2], got {samples.shape}")
        object.__setattr__(self, "samples", samples)

    # ------------------------------------------------------------------ construction

    @classmethod
    def load(cls, path: str | Path) -> AudioClip:
        """Read a WAV (any channel count) into a float32 stereo clip.

        Stereo files are read straight into a ``buffers.empty`` array (so dropping the clip
        later doesn't stall the GUI); other channel counts are downmixed, then copied into one.
        """
        with sf.SoundFile(str(path)) as f:
            rate = f.samplerate
            if f.channels == CHANNELS:
                out = buffers.empty((max(0, f.frames), CHANNELS), np.float32)
                data = f.read(dtype="float32", always_2d=True, out=out)
            else:
                data = buffers.copy(downmix_to_stereo(f.read(dtype="float32", always_2d=True)))
        return cls(data, int(rate), Path(path))

    @classmethod
    def silent(cls, seconds: float, sample_rate: int = 48000) -> AudioClip:
        frames = max(0, round(seconds * sample_rate))
        return cls(np.zeros((frames, CHANNELS), dtype=np.float32), sample_rate)

    # ------------------------------------------------------------------ properties

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])

    @property
    def channels(self) -> int:
        return CHANNELS

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return self.frames / self.sample_rate

    @property
    def peak(self) -> float:
        """Largest absolute sample value (can be above 1.0)."""
        if self.frames == 0:
            return 0.0
        return float(max(np.abs(self.samples.max()), np.abs(self.samples.min())))

    # ------------------------------------------------------------------ conversions

    def seconds_to_frame(self, seconds: float) -> int:
        """Nearest frame index for a time, clamped to ``[0, frames]``."""
        if not math.isfinite(seconds):
            return self.frames if seconds > 0 else 0
        return min(max(round(seconds * self.sample_rate), 0), self.frames)

    def frame_to_seconds(self, frame: int) -> float:
        return frame / self.sample_rate

    def clamp_seconds(self, seconds: float) -> float:
        if not math.isfinite(seconds):
            return self.duration if seconds > 0 else 0.0
        return min(max(seconds, 0.0), self.duration)

    def frame_range(self, start: float, end: float) -> tuple[int, int]:
        """``(start_frame, end_frame)`` for a time range, ordered and clamped."""
        a, b = self.seconds_to_frame(start), self.seconds_to_frame(end)
        return (a, b) if a <= b else (b, a)
