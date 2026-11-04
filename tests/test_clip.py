"""AudioClip: construction, loading and time/frame helpers."""

import math

import numpy as np
import pytest
import soundfile as sf
from conftest import write_test_wav

from desktop_audio_recorder.audio.clip import AudioClip


def test_load_stereo_float_wav(tmp_path):
    path = write_test_wav(tmp_path / "a.wav", 1.5, rate=48000)
    clip = AudioClip.load(path)
    assert clip.samples.dtype == np.float32
    assert clip.samples.shape == (72000, 2)
    assert clip.samples.flags.c_contiguous
    assert clip.sample_rate == 48000
    assert clip.frames == 72000
    assert clip.channels == 2
    assert clip.duration == pytest.approx(1.5)
    assert clip.source == path
    assert clip.peak == pytest.approx(0.5, abs=1e-3)


def test_load_keeps_overs_above_full_scale(tmp_path):
    data = np.zeros((100, 2), dtype=np.float32)
    data[10] = (1.2, -1.17)
    sf.write(tmp_path / "over.wav", data, 48000, subtype="FLOAT")
    clip = AudioClip.load(tmp_path / "over.wav")
    assert clip.samples[10, 0] == pytest.approx(1.2)
    assert clip.samples[10, 1] == pytest.approx(-1.17)
    assert clip.peak == pytest.approx(1.2)


def test_load_mono_is_duplicated(tmp_path):
    path = write_test_wav(tmp_path / "m.wav", 0.1, rate=44100, channels=1)
    clip = AudioClip.load(path)
    assert clip.samples.shape == (4410, 2)
    assert clip.sample_rate == 44100
    np.testing.assert_array_equal(clip.samples[:, 0], clip.samples[:, 1])


def test_load_multichannel_is_downmixed(tmp_path):
    data = np.zeros((10, 8), dtype=np.float32)
    data[:, 0] = 0.25  # FL
    data[:, 1] = -0.25  # FR
    sf.write(tmp_path / "s.wav", data, 48000, subtype="FLOAT")
    clip = AudioClip.load(tmp_path / "s.wav")
    assert clip.samples.shape == (10, 2)
    np.testing.assert_allclose(clip.samples[0], [0.25, -0.25])


def test_load_pcm16(tmp_path):
    data = np.full((480, 2), 0.5, dtype=np.float32)
    sf.write(tmp_path / "p.wav", data, 48000, subtype="PCM_16")
    clip = AudioClip.load(tmp_path / "p.wav")
    assert clip.samples.dtype == np.float32
    assert clip.samples[0, 0] == pytest.approx(0.5, abs=1e-4)


def test_constructor_validates_shape_and_rate():
    with pytest.raises(ValueError):
        AudioClip(np.zeros(10, dtype=np.float32), 48000)
    with pytest.raises(ValueError):
        AudioClip(np.zeros((10, 3), dtype=np.float32), 48000)
    with pytest.raises(ValueError):
        AudioClip(np.zeros((10, 2), dtype=np.float32), 0)


def test_constructor_converts_to_contiguous_float32():
    data = np.zeros((2, 10), dtype=np.float64).T  # Fortran-ordered float64
    clip = AudioClip(data, 1000)
    assert clip.samples.dtype == np.float32
    assert clip.samples.flags.c_contiguous


def test_silent_and_empty():
    clip = AudioClip.silent(0.5, 1000)
    assert clip.frames == 500
    assert clip.peak == 0.0
    empty = AudioClip.silent(0, 1000)
    assert empty.frames == 0
    assert empty.duration == 0.0
    assert empty.peak == 0.0


@pytest.mark.parametrize(
    ("seconds", "frame"),
    [
        (0.0, 0),
        (0.0004, 0),  # rounds to the nearest frame
        (0.0006, 1),
        (0.25, 250),
        (1.0, 1000),
        (1.5, 1000),  # clamped to the end
        (-0.1, 0),
        (math.inf, 1000),
        (-math.inf, 0),
        (math.nan, 0),
    ],
)
def test_seconds_to_frame(seconds, frame):
    clip = AudioClip.silent(1.0, 1000)
    assert clip.seconds_to_frame(seconds) == frame


def test_frame_to_seconds_and_clamp():
    clip = AudioClip.silent(2.0, 48000)
    assert clip.frame_to_seconds(24000) == pytest.approx(0.5)
    assert clip.clamp_seconds(-1) == 0.0
    assert clip.clamp_seconds(3.0) == pytest.approx(2.0)
    assert clip.clamp_seconds(1.25) == pytest.approx(1.25)
    assert clip.clamp_seconds(math.nan) == 0.0


def test_frame_range_is_ordered_and_clamped():
    clip = AudioClip.silent(1.0, 1000)
    assert clip.frame_range(0.2, 0.4) == (200, 400)
    assert clip.frame_range(0.4, 0.2) == (200, 400)
    assert clip.frame_range(-1.0, 5.0) == (0, 1000)
