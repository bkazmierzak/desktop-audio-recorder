"""Noise reduction (processing/denoise.py) on synthetic voice-like signals with white noise."""

import math

import numpy as np
import pytest
from conftest import speech_like

from desktop_audio_recorder.processing import denoise as dn
from desktop_audio_recorder.processing.denoise import (
    Cancelled,
    DenoiseSettings,
    NoiseMode,
    denoise,
    estimate_noise_profile,
    snr_db,
)

RATE = 48000
NO_HPF = DenoiseSettings(highpass_hz=None)  # the gate alone (the voice has no rumble)
NON_STATIONARY = DenoiseSettings(mode=NoiseMode.NON_STATIONARY, highpass_hz=None)


def stereo(mono):
    return np.repeat(np.asarray(mono, dtype=np.float32)[:, None], 2, axis=1)


def noisy_voice(seconds=8.0, level=0.1, seed=1, pauses=True):
    """``(clean stereo, noisy stereo, syllable mask)``; independent noise on each channel."""
    clean, active = speech_like(seconds, RATE, pauses=pauses)
    clean = stereo(clean)
    noise = level * np.random.default_rng(seed).standard_normal(clean.shape)
    return clean, (clean + noise).astype(np.float32), active


def power_db(x):
    x = np.asarray(x, dtype=np.float64)
    return 10 * math.log10(np.mean(x * x))


def tone(x, hz, rate=RATE):
    """``(amplitude, phase)`` of the ``hz`` sine in ``x`` (least squares fit of sin and cos)."""
    t = np.arange(len(x)) / rate
    basis = np.stack([np.sin(2 * np.pi * hz * t), np.cos(2 * np.pi * hz * t)], axis=1)
    (s, c), *_ = np.linalg.lstsq(basis, np.asarray(x, dtype=np.float64), rcond=None)
    return float(np.hypot(s, c)), math.atan2(c, s)


def tone_amplitude(x, hz, rate=RATE):
    return tone(x, hz, rate)[0]


def away_from(active, seconds):
    """Frames at least ``seconds`` from any true frame of ``active``."""
    n = round(seconds * RATE)
    near = np.convolve(active.astype(np.int32), np.ones(2 * n + 1, np.int32), mode="same") > 0
    return ~near


# --------------------------------------------------------------------------- settings


def test_settings_defaults_and_validation():
    s = DenoiseSettings()
    assert s.mode is NoiseMode.STATIONARY and s.stationary
    assert (s.strength, s.highpass_hz, s.normalize, s.normalize_db) == (0.9, 80.0, False, -1.0)
    assert s.gates and s.uses_profile and not s.is_noop
    assert s.describe() == "stationary, 90 %, high-pass 80 Hz"
    for bad in ({"strength": -0.1}, {"strength": 1.01}, {"highpass_hz": 5}, {"highpass_hz": 900}):
        with pytest.raises(ValueError):
            DenoiseSettings(**bad)


def test_settings_flags():
    off = DenoiseSettings(strength=0.0, highpass_hz=None)
    assert off.is_noop and not off.gates and not off.uses_profile
    assert off.describe() == "no change"
    assert not DenoiseSettings(strength=0.0, normalize=True, highpass_hz=None).is_noop
    assert not NON_STATIONARY.uses_profile
    assert NON_STATIONARY.context_frames(RATE) > NO_HPF.context_frames(RATE)
    assert (
        DenoiseSettings(mode=NoiseMode.NON_STATIONARY, strength=0.5, normalize=True).describe()
        == "non-stationary, 50 %, high-pass 80 Hz, normalized to -1 dBFS"
    )


# --------------------------------------------------------------------------- shape and format


@pytest.mark.parametrize("settings", [DenoiseSettings(), NON_STATIONARY], ids=["stat", "nonstat"])
@pytest.mark.parametrize("frames", [dn.MIN_FRAMES, 48000 * 3 + 123, 48000 * 25 + 7])
def test_keeps_length_channels_and_input(settings, frames):
    rng = np.random.default_rng(0)
    x = (0.1 * rng.standard_normal((frames, 2))).astype(np.float32)
    before = x.copy()
    y = denoise(x, RATE, settings)
    assert y.shape == x.shape and y.dtype == np.float32 and y.flags.c_contiguous
    assert np.isfinite(y).all()
    assert np.array_equal(x, before)  # the input is never modified
    # The last samples are processed too (noisereduce alone leaves the last partial hop at 0).
    assert np.count_nonzero(y[-300:] == 0) == 0


def test_mono_and_multichannel_shapes():
    rng = np.random.default_rng(0)
    for channels in (1, 3):
        x = (0.1 * rng.standard_normal((RATE, channels))).astype(np.float32)
        assert denoise(x, RATE, DenoiseSettings()).shape == x.shape


def test_rejects_bad_input():
    with pytest.raises(ValueError, match="(?i)too short"):
        denoise(np.zeros((dn.MIN_FRAMES - 1, 2), np.float32), RATE, DenoiseSettings())
    with pytest.raises(ValueError, match="frames, channels"):
        denoise(np.zeros(RATE, np.float32), RATE, DenoiseSettings())
    with pytest.raises(ValueError, match="noise profile is too short"):
        denoise(np.zeros((RATE, 2), np.float32), RATE, NO_HPF, noise_profile=np.zeros((100, 2)))


def test_silence_stays_silent():
    y = denoise(np.zeros((RATE, 2), np.float32), RATE, DenoiseSettings(normalize=True))
    assert np.array_equal(y, np.zeros((RATE, 2), np.float32))


# --------------------------------------------------------------------------- cleaner audio


@pytest.mark.parametrize(
    ("level", "min_gain_db"),
    [(0.1, 6.0), (0.03, 2.0)],  # input SNR about -1 dB and +9 dB
)
def test_stationary_improves_snr_without_a_profile(level, min_gain_db):
    clean, x, _ = noisy_voice(level=level)
    before = snr_db(clean, x)
    after = snr_db(clean, denoise(x, RATE, NO_HPF))
    assert after > before + min_gain_db, (before, after)


def test_default_settings_improve_snr():
    clean, x, _ = noisy_voice(level=0.1)
    assert snr_db(clean, denoise(x, RATE, DenoiseSettings())) > snr_db(clean, x) + 5


def test_nonstationary_improves_snr_on_a_noisy_clip():
    clean, x, _ = noisy_voice(level=0.1)
    before = snr_db(clean, x)
    after = snr_db(clean, denoise(x, RATE, NON_STATIONARY))
    assert after > before + 3, (before, after)


def test_noise_between_words_is_reduced_by_the_strength():
    clean, x, active = noisy_voice(level=0.05)
    gaps = away_from(active, 0.06)  # the gate's mask is smoothed over 50 ms
    gaps[: RATE // 10] = gaps[-RATE // 10 :] = False  # away from the clip edges
    assert gaps.sum() > RATE  # over a second of noise between words
    reductions = {}
    for strength in (0.5, 0.9, 1.0):
        y = denoise(x, RATE, DenoiseSettings(strength=strength, highpass_hz=None))
        reductions[strength] = power_db(x[gaps]) - power_db(y[gaps])
    # prop_decrease s turns gated bins down by 1 - s (-6 dB at 0.5, -20 dB at 0.9); noise peaks
    # above the threshold pass, so the measured reduction is less (4.6 / 11.6 / 14.6 dB).
    assert 3 < reductions[0.5] < 6, reductions
    assert 9 < reductions[0.9] < 16, reductions
    assert reductions[1.0] > reductions[0.9] + 2, reductions


def test_profile_from_a_noise_selection_helps_the_stationary_mode():
    """Continuous voice with no pauses: the quietest parts are still voice, so the automatic
    profile is poor; one second of noise before the voice (what the user would select) as the
    profile makes a large difference."""
    rng = np.random.default_rng(5)
    voice, _ = speech_like(4.0, RATE, pauses=False)
    clean = stereo(np.concatenate([np.zeros(RATE, np.float32), voice]))
    x = (clean + 0.05 * rng.standard_normal(clean.shape)).astype(np.float32)
    target, reference = x[RATE:], clean[RATE:]
    before = snr_db(reference, target)
    auto = snr_db(reference, denoise(target, RATE, NO_HPF))
    profiled = snr_db(reference, denoise(target, RATE, NO_HPF, noise_profile=x[:RATE]))
    assert profiled > before + 3, (before, profiled)
    assert profiled > auto + 5, (auto, profiled)


def test_profile_is_ignored_by_the_nonstationary_mode():
    _, x, _ = noisy_voice(seconds=3.0)
    a = denoise(x, RATE, NON_STATIONARY)
    b = denoise(x, RATE, NON_STATIONARY, noise_profile=x[: RATE // 2])
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- high-pass


def test_highpass_removes_rumble_and_keeps_the_voice_band_in_phase():
    t = np.arange(4 * RATE) / RATE
    voice = 0.2 * np.sin(2 * np.pi * 1000 * t)
    rumble = 0.3 * np.sin(2 * np.pi * 30 * t)
    x = stereo(voice + rumble)
    y = denoise(x, RATE, DenoiseSettings(strength=0.0))  # high-pass only
    mid = slice(RATE // 2, -RATE // 2)
    left = y[mid, 0]
    assert 20 * math.log10(tone_amplitude(left, 30) / 0.3) < -30  # rumble gone
    amplitude, phase = tone(left, 1000)
    original_phase = tone(voice[mid], 1000)[1]
    assert abs(20 * math.log10(amplitude / 0.2)) < 0.05  # voice kept
    assert abs(math.degrees(phase - original_phase)) < 0.1  # zero phase: not delayed


def test_highpass_is_6db_down_at_the_cutoff():
    t = np.arange(4 * RATE) / RATE
    y = dn.highpass(np.sin(2 * np.pi * 80 * t)[:, None], RATE, 80.0)
    gain_db = 20 * math.log10(tone_amplitude(y[RATE:-RATE, 0], 80))
    assert -6.5 < gain_db < -5.5


def test_highpass_removes_rumble_in_the_full_chain():
    clean, x, _ = noisy_voice(level=0.03)
    t = np.arange(len(x)) / RATE
    rumbly = (x + 0.2 * np.sin(2 * np.pi * 25 * t)[:, None]).astype(np.float32)
    without = denoise(rumbly, RATE, NO_HPF)
    with_hpf = denoise(rumbly, RATE, DenoiseSettings())
    assert tone_amplitude(with_hpf[:, 0], 25) < 0.2 / 100
    assert snr_db(clean, with_hpf) > snr_db(clean, without) + 3


# --------------------------------------------------------------------------- chunks


def test_chunked_stationary_equals_a_single_pass():
    _, x, _ = noisy_voice(seconds=9.0)
    whole = denoise(x, RATE, DenoiseSettings(), chunk_seconds=100)
    chunked = denoise(x, RATE, DenoiseSettings(), chunk_seconds=2)
    assert np.max(np.abs(whole - chunked)) < 1e-6


def test_chunked_nonstationary_is_close_and_has_no_steps():
    _, x, _ = noisy_voice(seconds=9.0)
    whole = denoise(x, RATE, NON_STATIONARY, chunk_seconds=100)
    chunked = denoise(x, RATE, NON_STATIONARY, chunk_seconds=2)
    error = chunked - whole
    assert power_db(error) - power_db(whole) < -20
    # The running noise estimate differs a little near chunk boundaries; the cross-fade keeps
    # the error smooth there: no step in it comes near the audio's own sample-to-sample steps.
    steps = np.max(np.abs(np.diff(error, axis=0)))
    assert steps < 0.25 * np.max(np.abs(np.diff(whole, axis=0)))


def test_identical_channels_are_processed_once(monkeypatch):
    import noisereduce

    calls = []
    real = noisereduce.reduce_noise

    def spy(y, **kwargs):
        calls.append(np.shape(y))
        return real(y=y, **kwargs)

    monkeypatch.setattr(noisereduce, "reduce_noise", spy)
    _, x, _ = noisy_voice(seconds=2.0)
    mono = np.repeat(x[:, :1], 2, axis=1)
    y = denoise(mono, RATE, DenoiseSettings())
    assert np.array_equal(y[:, 0], y[:, 1])
    assert all(len(shape) == 1 for shape in calls)  # one channel per call
    calls.clear()
    denoise(x, RATE, DenoiseSettings())
    assert calls and all(shape[0] == 2 for shape in calls)


# --------------------------------------------------------------------------- progress / cancel


def test_progress_is_monotonic_and_ends_at_one():
    _, x, _ = noisy_voice(seconds=5.0)
    seen = []
    denoise(x, RATE, DenoiseSettings(), progress=seen.append, chunk_seconds=1)
    assert seen[0] == 0.0 and seen[-1] == 1.0
    assert seen == sorted(seen) and len(seen) >= 6


def test_cancel_stops_with_cancelled_and_leaves_the_input_alone():
    _, x, _ = noisy_voice(seconds=5.0)
    before = x.copy()
    seen = []
    with pytest.raises(Cancelled):
        denoise(
            x,
            RATE,
            DenoiseSettings(),
            progress=seen.append,
            cancelled=lambda: len(seen) >= 3,
            chunk_seconds=1,
        )
    assert seen[-1] < 1.0
    assert np.array_equal(x, before)


def test_cancel_before_starting():
    _, x, _ = noisy_voice(seconds=2.0)
    with pytest.raises(Cancelled):
        denoise(x, RATE, DenoiseSettings(), cancelled=lambda: True)


# --------------------------------------------------------------------------- normalize


def test_normalize_afterwards():
    _, x, _ = noisy_voice(seconds=3.0, level=0.03)
    y = denoise(x, RATE, DenoiseSettings(normalize=True))
    assert 20 * math.log10(np.max(np.abs(y))) == pytest.approx(-1.0, abs=1e-4)
    z = denoise(x, RATE, DenoiseSettings(normalize=True, normalize_db=-3.0))
    assert 20 * math.log10(np.max(np.abs(z))) == pytest.approx(-3.0, abs=1e-4)


def test_overs_are_kept():
    x = np.full((RATE, 2), 0.0, np.float32)
    x[RATE // 2 :: 100] = 1.5  # clicks above full scale
    y = denoise(x, RATE, DenoiseSettings(strength=0.0, highpass_hz=None, normalize=False))
    assert np.array_equal(x, y)  # nothing to do but copy: overs aren't clipped


# --------------------------------------------------------------------------- noise profile


def test_estimate_noise_profile_takes_the_quiet_parts_and_skips_silence():
    rng = np.random.default_rng(0)
    silence = np.zeros((RATE, 2), np.float32)
    hiss = (0.01 * rng.standard_normal((2 * RATE, 2))).astype(np.float32)
    loud = (0.3 * rng.standard_normal((7 * RATE, 2))).astype(np.float32)
    x = np.concatenate([silence, loud[: 3 * RATE], hiss, loud[3 * RATE :]])
    profile = estimate_noise_profile(x, RATE)
    assert profile.dtype == np.float32 and profile.shape[1] == 2
    # 10 % of 40 blocks = 4 quarter-seconds, all hiss (not the digital silence).
    assert profile.shape[0] == 4 * RATE // 4
    assert np.sqrt(np.mean(profile**2)) == pytest.approx(0.01, rel=0.1)


def test_estimate_noise_profile_limits_and_fallbacks():
    rng = np.random.default_rng(0)
    long = (0.1 * rng.standard_normal((400 * RATE // 4, 2))).astype(np.float32)  # 100 s
    assert estimate_noise_profile(long, RATE).shape[0] == 40 * RATE // 4  # 10 % of 100 s
    huge = np.zeros((1000 * RATE // 4 * 2, 1), np.float32)
    huge[:] = 0.01
    assert estimate_noise_profile(huge, RATE).shape[0] == dn.PROFILE_MAX_SECONDS * RATE
    short = (0.1 * rng.standard_normal((RATE // 3, 2))).astype(np.float32)
    assert np.array_equal(estimate_noise_profile(short, RATE), short)
    silent = np.zeros((4 * RATE, 2), np.float32)
    assert estimate_noise_profile(silent, RATE).shape[0] == 4 * RATE


def test_snr_db():
    a = np.ones(100)
    assert snr_db(a, a) == math.inf
    assert snr_db(a, 2 * a) == pytest.approx(0.0)
    assert snr_db(np.zeros(10), np.ones(10)) == -math.inf
