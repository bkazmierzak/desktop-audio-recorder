"""Noise reduction for recorded voice: a zero-phase high-pass, spectral gating, normalize (no Qt).

``denoise(samples, rate, settings, noise_profile=None, progress=None, cancelled=None)`` returns a
new ``float32`` array with the same shape (length and channels) as ``samples``; the sample rate
is unchanged and the input is never modified. Steps, in order:

1. **High-pass** (optional, default 80 Hz): a 2nd-order Butterworth run forwards and backwards
   (``sosfiltfilt``), so it is zero-phase with a 24 dB/octave slope (-6 dB at the cutoff). It
   removes rumble before the gate measures the noise.
2. **Spectral gating** with ``noisereduce`` (strength = ``prop_decrease``; 0 skips this step):

   * *Stationary* (steady hiss, hum, fan): the noise is described by the mean and spread of a
     noise profile's spectrum, and every time-frequency bin below that threshold is turned
     down. The profile is the user's selection of noise, or (``noise_profile=None``) the
     quietest quarter-seconds of the audio itself (``estimate_noise_profile``): speech has
     pauses, and those are noise only. noisereduce's own default (the whole signal as the
     profile) counts the speech as noise too and removes most of it.
   * *Non-stationary* (noise that changes): each bin is compared with its own average over the
     last ~2 s. The threshold is 1 instead of noisereduce's default of 2 (see
     ``NONSTATIONARY_THRESHOLD``).
3. **Normalize** (optional): scale the result so its sample peak is at -1 dBFS.

Chunks: the audio is processed ``CHUNK_SECONDS`` at a time, each with ``context`` frames of the
neighbouring audio on both sides (so the filters and the gate see what's around the chunk), and
neighbouring chunks are cross-faded over ``CROSSFADE_SECONDS``, so chunk boundaries can't click.
That bounds the temporaries (noisereduce works in float64 and keeps a few spectrogram-sized
arrays: ~150 MB for a 20 s chunk) and gives regular points to report ``progress(fraction)`` and
to stop when ``cancelled()`` returns true (``Cancelled`` is raised; nothing has been changed).

Identical channels (a mono source recorded as stereo) are processed once.

The output comes from ``buffers.empty``, so dropping it later doesn't stall the GUI.
"""

from __future__ import annotations

import enum
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from desktop_audio_recorder.audio import buffers
from desktop_audio_recorder.editing import operations as ops

log = logging.getLogger(__name__)

CHUNK_SECONDS = 20.0
CROSSFADE_SECONDS = 0.05
STATIONARY_CONTEXT_SECONDS = 0.5  # covers the STFT window, the 50 ms mask smoothing, the HPF
NONSTATIONARY_TIME_CONSTANT_S = 2.0  # noisereduce's default
NONSTATIONARY_CONTEXT_SECONDS = 1.5 * NONSTATIONARY_TIME_CONSTANT_S
# noisereduce's default (2) needs a bin to be 3x its running average to pass; on speech-like test
# signals that removes most of the voice (it lifts the running average). 1 (2x) kept about 1.5x
# more of it with a better SNR, at both 9 dB and -1 dB input SNR.
NONSTATIONARY_THRESHOLD = 1.0
STATIONARY_N_STD = 1.5  # noisereduce's default
N_FFT = 1024  # 21 ms at 48 kHz (noisereduce's default)
HOP = N_FFT // 4  # noisereduce's default; chunks start on this grid so their STFT frames line up
HIGHPASS_ORDER = 2  # doubled by the forward-backward pass
HIGHPASS_RANGE_HZ = (20.0, 500.0)

PROFILE_BLOCK_SECONDS = 0.25
PROFILE_QUIET_FRACTION = 0.1  # the quietest 10 % of the blocks...
PROFILE_MIN_BLOCKS = 2
PROFILE_MAX_SECONDS = 10.0  # ...up to 10 s; longer profiles only cost time
SILENT_BLOCK_ENERGY = 1e-12  # mean square; digital silence (zero-filled gaps) isn't noise

MIN_FRAMES = 2 * N_FFT  # shorter audio can't be analysed (43 ms at 48 kHz)

ProgressCallback = Callable[[float], None]
CancelledCallback = Callable[[], bool]


class Cancelled(Exception):  # noqa: N818 (not an error: the user stopped it)
    """``cancelled()`` returned true; nothing was changed."""


class NoiseMode(enum.Enum):
    STATIONARY = "stationary"
    NON_STATIONARY = "non-stationary"


@dataclass(frozen=True, slots=True)
class DenoiseSettings:
    mode: NoiseMode = NoiseMode.STATIONARY
    strength: float = 0.9  # noisereduce's prop_decrease: 1.0 removes the noise completely
    highpass_hz: float | None = 80.0  # None: no high-pass
    normalize: bool = False
    normalize_db: float = ops.NORMALIZE_TARGET_DB

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be within [0, 1], got {self.strength}")
        if self.highpass_hz is not None and not (
            HIGHPASS_RANGE_HZ[0] <= self.highpass_hz <= HIGHPASS_RANGE_HZ[1]
        ):
            raise ValueError(f"high-pass cutoff must be within {HIGHPASS_RANGE_HZ} Hz")

    @property
    def stationary(self) -> bool:
        return self.mode is NoiseMode.STATIONARY

    @property
    def gates(self) -> bool:
        """Whether the spectral gate runs (strength above 0)."""
        return self.strength > 0.0

    @property
    def is_noop(self) -> bool:
        return not self.gates and self.highpass_hz is None and not self.normalize

    @property
    def uses_profile(self) -> bool:
        return self.gates and self.stationary

    def context_frames(self, rate: int) -> int:
        seconds = (
            STATIONARY_CONTEXT_SECONDS
            if self.stationary or not self.gates
            else NONSTATIONARY_CONTEXT_SECONDS
        )
        return round(seconds * rate)

    def describe(self) -> str:
        parts = []
        if self.gates:
            parts.append(f"{self.mode.value}, {self.strength * 100:.0f} %")
        if self.highpass_hz is not None:
            parts.append(f"high-pass {self.highpass_hz:g} Hz")
        if self.normalize:
            parts.append(f"normalized to {self.normalize_db:g} dBFS")
        return ", ".join(parts) or "no change"


# --------------------------------------------------------------------------- building blocks


def highpass_sos(rate: int, cutoff_hz: float) -> np.ndarray:
    return butter(HIGHPASS_ORDER, cutoff_hz, btype="highpass", fs=rate, output="sos")


def highpass(samples: np.ndarray, rate: int, cutoff_hz: float) -> np.ndarray:
    """Zero-phase high-pass of a short array along axis 0 (a float64 copy)."""
    return _filtfilt(highpass_sos(rate, cutoff_hz), samples)


def _filtfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    padlen = min(3 * (2 * len(sos) + 1), x.shape[0] - 1)
    return sosfiltfilt(sos, x, axis=0, padlen=padlen)


def _hops(frames: int) -> int:
    """``frames`` rounded up to a whole number of STFT hops."""
    return -(-frames // HOP) * HOP


def _check_samples(samples: np.ndarray, what: str = "audio") -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[1] < 1:
        raise ValueError(f"{what} must be shaped [frames, channels], got {samples.shape}")
    return samples


def estimate_noise_profile(
    samples: np.ndarray,
    rate: int,
    *,
    block_seconds: float = PROFILE_BLOCK_SECONDS,
    max_seconds: float = PROFILE_MAX_SECONDS,
) -> np.ndarray:
    """The quietest blocks of ``samples`` (by mean square), in order, joined into one array:
    the noise between words. Digitally silent blocks are skipped. Returns a small copy."""
    samples = _check_samples(samples)
    block = max(1, round(block_seconds * rate))
    n_blocks = samples.shape[0] // block
    if n_blocks <= PROFILE_MIN_BLOCKS:
        return np.array(samples[: round(max_seconds * rate)], dtype=np.float32)
    energy = np.empty(n_blocks, dtype=np.float64)
    step = max(1, (1 << 20) // block)  # blocks per pass: no temporary over ~1 M frames
    for i in range(0, n_blocks, step):
        j = min(n_blocks, i + step)
        flat = samples[i * block : j * block].reshape(j - i, -1)
        energy[i:j] = np.einsum("ij,ij->i", flat, flat, dtype=np.float64) / flat.shape[1]
    candidates = np.flatnonzero(energy > SILENT_BLOCK_ENERGY)
    if candidates.size == 0:
        return np.array(samples[: round(max_seconds * rate)], dtype=np.float32)
    count = round(n_blocks * PROFILE_QUIET_FRACTION)
    count = min(candidates.size, max(PROFILE_MIN_BLOCKS, count), round(max_seconds / block_seconds))
    quiet = np.sort(candidates[np.argsort(energy[candidates], kind="stable")[:count]])
    return np.concatenate([samples[i * block : (i + 1) * block] for i in quiet]).astype(np.float32)


def _prepare_profile(
    profile: np.ndarray, rate: int, sos: np.ndarray | None, channels: int
) -> np.ndarray:
    """Mono (noisereduce averages the profile's channels anyway), high-passed like the audio,
    at most ``PROFILE_MAX_SECONDS``."""
    profile = _check_samples(profile, "noise profile")
    profile = profile[: round(PROFILE_MAX_SECONDS * rate)]
    if profile.shape[0] < MIN_FRAMES:
        raise ValueError(
            f"The noise profile is too short (at least {MIN_FRAMES / rate * 1000:.0f} ms)"
        )
    mono = profile.astype(np.float64).mean(axis=1)
    if sos is not None:
        mono = _filtfilt(sos, mono)
    return mono


def _gate(block: np.ndarray, rate: int, settings: DenoiseSettings, profile) -> np.ndarray:
    """noisereduce on one block ``[frames, channels]``; returns ``[frames, channels]``."""
    import noisereduce as nr  # slow import (joblib, tqdm); only when needed

    kwargs = dict(
        sr=rate,
        stationary=settings.stationary,
        y_noise=profile,
        prop_decrease=settings.strength,
        time_constant_s=NONSTATIONARY_TIME_CONSTANT_S,
        thresh_n_mult_nonstationary=NONSTATIONARY_THRESHOLD,
        n_std_thresh_stationary=STATIONARY_N_STD,
        n_fft=N_FFT,
        chunk_size=None,  # chunked here (with progress, cancel and cross-fades)
        padding=0,
        n_jobs=1,
        use_tqdm=False,
    )
    frames, channels = block.shape
    # noisereduce's STFT drops a last partial hop (padded=False), which leaves the last
    # ``frames % HOP`` samples at zero: pad to a whole number of hops and trim afterwards.
    pad = -frames % HOP
    if pad:
        block = np.concatenate([block, np.zeros((pad, channels), block.dtype)])
    if channels > 1 and all(np.array_equal(block[:, 0], block[:, c]) for c in range(1, channels)):
        mono = nr.reduce_noise(y=block[:, 0], **kwargs)
        return np.repeat(mono[:frames, np.newaxis], channels, axis=1)
    return nr.reduce_noise(y=block.T, **kwargs).T[:frames]


# --------------------------------------------------------------------------- denoise


def denoise(
    samples: np.ndarray,
    rate: int,
    settings: DenoiseSettings,
    noise_profile: np.ndarray | None = None,
    progress: ProgressCallback | None = None,
    cancelled: CancelledCallback | None = None,
    *,
    chunk_seconds: float = CHUNK_SECONDS,
) -> np.ndarray:
    """Process ``samples`` (``[frames, channels]``) as described in the module docstring.

    ``noise_profile`` (``[frames, channels]``, at least ``MIN_FRAMES``) is used by the stationary
    mode; without one it is estimated from ``samples``. Raises ``Cancelled`` when
    ``cancelled()`` turns true, and ``ValueError`` for audio shorter than ``MIN_FRAMES``.
    """
    samples = _check_samples(samples)
    frames, channels = samples.shape
    if frames < MIN_FRAMES:
        raise ValueError(f"Too short to process (at least {MIN_FRAMES / rate * 1000:.0f} ms)")
    started = time.perf_counter()

    def check_cancel() -> None:
        if cancelled is not None and cancelled():
            raise Cancelled("Noise reduction cancelled")

    def report(fraction: float) -> None:
        if progress is not None:
            progress(min(1.0, max(0.0, fraction)))

    report(0.0)
    sos = highpass_sos(rate, settings.highpass_hz) if settings.highpass_hz is not None else None
    profile = None
    if settings.uses_profile:
        if noise_profile is None:
            noise_profile = estimate_noise_profile(samples, rate)
            check_cancel()
        profile = _prepare_profile(noise_profile, rate, sos, channels)

    out = buffers.empty((frames, channels), np.float32)
    chunk = _hops(max(MIN_FRAMES, round(chunk_seconds * rate)))
    context = _hops(max(MIN_FRAMES, settings.context_frames(rate)))
    fade = min(round(CROSSFADE_SECONDS * rate), context)
    for start in range(0, frames, chunk):
        check_cancel()
        stop = min(frames, start + chunk)
        lead = fade if start > 0 else 0  # cross-fade with the previous chunk's tail
        a, b = max(0, start - context), min(frames, stop + context)
        block = samples[a:b]
        if sos is not None:
            block = _filtfilt(sos, block)
        if settings.gates:
            block = _gate(np.asarray(block, dtype=np.float64), rate, settings, profile)
        result = block[start - lead - a : stop - a]
        if lead:
            ramp = ((np.arange(lead) + 0.5) / lead)[:, np.newaxis]
            previous = out[start - lead : start]
            previous[...] = previous + ramp * (result[:lead] - previous)
        out[start:stop] = result[lead:]
        report(stop / frames)

    if settings.normalize:
        factor = ops.normalize_factor(ops.peak(out), settings.normalize_db)
        if factor is not None:
            ops.gain_region(out, factor)
    seconds = time.perf_counter() - started
    log.info(
        "Denoised %.1f s of %d-channel audio (%s) in %.2f s (%.2f s per minute)",
        frames / rate,
        channels,
        settings.describe(),
        seconds,
        seconds / (frames / rate) * 60,
    )
    report(1.0)
    return out


def snr_db(reference: np.ndarray, signal: np.ndarray) -> float:
    """Signal-to-noise ratio of ``signal`` against a clean ``reference``, in dB."""
    reference = np.asarray(reference, dtype=np.float64)
    error = np.asarray(signal, dtype=np.float64) - reference
    noise = float(np.sum(error * error))
    power = float(np.sum(reference * reference))
    if noise == 0.0:
        return math.inf
    return 10.0 * math.log10(power / noise) if power > 0 else -math.inf


__all__ = [
    "Cancelled",
    "DenoiseSettings",
    "NoiseMode",
    "denoise",
    "estimate_noise_profile",
    "highpass",
    "snr_db",
]
