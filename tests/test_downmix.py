"""N-channel → stereo downmix math."""

import numpy as np
import pytest

from desktop_audio_recorder.audio.downmix import (
    MINUS_3DB,
    downmix_matrix,
    downmix_to_stereo,
    interleaved_to_stereo,
)


def impulse(channels, channel, value=0.5, frames=4):
    block = np.zeros((frames, channels), dtype=np.float32)
    block[:, channel] = value
    return block


def test_stereo_passthrough_is_a_copy():
    block = np.random.default_rng(0).uniform(-0.9, 0.9, (100, 2)).astype(np.float32)
    out = downmix_to_stereo(block)
    np.testing.assert_array_equal(out, block)
    assert out is not block
    assert not np.shares_memory(out, block)


def test_mono_duplicates_to_both_sides():
    block = np.linspace(-0.5, 0.5, 10, dtype=np.float32)[:, np.newaxis]
    out = downmix_to_stereo(block)
    assert out.shape == (10, 2)
    np.testing.assert_array_equal(out[:, 0], block[:, 0])
    np.testing.assert_array_equal(out[:, 1], block[:, 0])


def test_1d_input_treated_as_mono():
    out = downmix_to_stereo(np.array([0.1, -0.2], dtype=np.float32))
    np.testing.assert_allclose(out, [[0.1, 0.1], [-0.2, -0.2]])


@pytest.mark.parametrize(
    ("channel", "expected"),
    [
        (0, (0.5, 0.0)),  # FL
        (1, (0.0, 0.5)),  # FR
        (2, (0.5 * MINUS_3DB, 0.5 * MINUS_3DB)),  # FC
        (3, (0.0, 0.0)),  # LFE dropped
        (4, (0.5 * MINUS_3DB, 0.0)),  # BL
        (5, (0.0, 0.5 * MINUS_3DB)),  # BR
        (6, (0.5 * MINUS_3DB, 0.0)),  # SL
        (7, (0.0, 0.5 * MINUS_3DB)),  # SR
    ],
)
def test_7_1_downmix(channel, expected):
    out = downmix_to_stereo(impulse(8, channel))
    np.testing.assert_allclose(out[0], expected, rtol=1e-6)


def test_5_1_drops_lfe_and_folds_surrounds():
    m = downmix_matrix(6)
    np.testing.assert_allclose(m[3], [0, 0])
    np.testing.assert_allclose(m[4], [MINUS_3DB, 0], rtol=1e-6)
    np.testing.assert_allclose(m[5], [0, MINUS_3DB], rtol=1e-6)


def test_stereo_content_in_8ch_container_keeps_unity_gain():
    # What Windows delivers for a stereo source on the INZONE 7.1 endpoint: FL/FR only.
    block = np.zeros((48, 8), dtype=np.float32)
    block[:, 0] = 0.3
    block[:, 1] = -0.4
    out = downmix_to_stereo(block)
    np.testing.assert_allclose(out, np.tile([0.3, -0.4], (48, 1)), rtol=1e-6)


@pytest.mark.parametrize("channels", [1, 2, 3, 4, 5, 6, 7, 8, 10, 12])
def test_every_channel_count_produces_stereo(channels):
    m = downmix_matrix(channels)
    assert m.shape == (channels, 2)
    out = downmix_to_stereo(np.full((16, channels), 0.1, dtype=np.float32))
    assert out.shape == (16, 2)
    assert out.dtype == np.float32
    # Both sides get signal.
    assert np.all(out > 0)


def test_extra_channels_alternate_sides():
    m = downmix_matrix(10)
    np.testing.assert_allclose(m[8], [MINUS_3DB, 0], rtol=1e-6)
    np.testing.assert_allclose(m[9], [0, MINUS_3DB], rtol=1e-6)


def test_overs_are_kept_by_default():
    # Windows' float mix can exceed full scale; the float32 pipeline keeps it.
    out = downmix_to_stereo(np.full((4, 2), 1.2, dtype=np.float32))
    np.testing.assert_allclose(out, 1.2, rtol=1e-6)
    dense = downmix_to_stereo(np.full((4, 8), 0.9, dtype=np.float32))
    assert dense.max() > 1.0


def test_optional_clipping():
    block = np.full((4, 8), 0.9, dtype=np.float32)
    assert downmix_to_stereo(block, clip=True).max() == 1.0
    assert downmix_to_stereo(np.full((4, 2), -3.0, dtype=np.float32), clip=True).min() == -1.0


def test_matrix_is_read_only():
    with pytest.raises(ValueError):
        downmix_matrix(8)[0, 0] = 2.0


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_channel_count(bad):
    with pytest.raises(ValueError):
        downmix_matrix(bad)


def test_interleaved_bytes():
    frames = np.arange(24, dtype=np.float32).reshape(3, 8) / 100
    out = interleaved_to_stereo(frames.tobytes(), 8)
    np.testing.assert_allclose(out, downmix_to_stereo(frames), rtol=1e-6)
