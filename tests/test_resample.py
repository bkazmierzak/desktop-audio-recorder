"""audio/resample.py: StreamResampler equals resample_poly over the whole stream."""

import math

import numpy as np
import pytest
from scipy.signal import resample_poly

from desktop_audio_recorder.audio.resample import StreamResampler


@pytest.mark.parametrize(
    ("rate_in", "rate_out"),
    [
        (44100, 48000),
        (48000, 44100),
        (96000, 48000),
        (22050, 48000),
        (48000, 96000),
        (32000, 48000),
    ],
)
def test_any_block_sizes_equal_one_pass(rate_in, rate_out):
    rng = np.random.default_rng(rate_in + rate_out)
    x = rng.uniform(-1, 1, (rate_in // 2 + 777, 2)).astype(np.float32)
    g = math.gcd(rate_in, rate_out)
    expected = resample_poly(x, rate_out // g, rate_in // g, axis=0)
    r = StreamResampler(rate_in, rate_out)
    out, pos = [], 0
    while pos < len(x):
        n = int(rng.integers(1, 3000))
        out.append(r.process(x[pos : pos + n]))
        pos += n
    out.append(r.flush())
    y = np.concatenate(out)
    assert y.dtype == np.float32
    assert y.shape == expected.shape == (math.ceil(len(x) * rate_out / rate_in), 2)
    np.testing.assert_allclose(y, expected, atol=2e-6)
    assert (r.frames_in, r.frames_out) == (len(x), len(y))


def test_a_tone_keeps_its_frequency_and_level():
    rate_in, rate_out = 44100, 48000
    t = np.arange(rate_in) / rate_in
    tone = 0.5 * np.sin(2 * np.pi * 1000 * t)
    r = StreamResampler(rate_in, rate_out)
    blocks = [r.process(np.stack([tone, tone], 1)[a : a + 441]) for a in range(0, rate_in, 441)]
    y = np.concatenate(blocks + [r.flush()])[1000:-1000, 0]
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    assert np.argmax(spectrum) * rate_out / len(y) == pytest.approx(1000, abs=2)
    assert np.max(np.abs(y)) == pytest.approx(0.5, abs=0.005)


def test_same_rate_passes_blocks_through():
    r = StreamResampler(48000, 48000)
    block = np.ones((10, 2), np.float32)
    assert r.process(block) is block
    assert len(r.flush()) == 0


def test_errors():
    with pytest.raises(ValueError):
        StreamResampler(0, 48000)
    r = StreamResampler(44100, 48000)
    with pytest.raises(ValueError):
        r.process(np.zeros((10, 3), np.float32))
    r.flush()
    with pytest.raises(RuntimeError):
        r.process(np.zeros((10, 2), np.float32))
    assert len(r.flush()) == 0  # flushing twice gives nothing more
