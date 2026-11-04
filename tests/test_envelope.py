"""Min/max envelope: pyramid queries must match a brute-force scan bin for bin."""

import numpy as np
import pytest

from desktop_audio_recorder.audio.envelope import (
    EnvelopePyramid,
    minmax_envelope,
    reduce_bins,
)


@pytest.fixture(scope="module")
def samples():
    rng = np.random.default_rng(1234)
    data = rng.standard_normal((300_017, 2)).astype(np.float32)  # odd length: partial blocks
    data[123_456, 1] = 7.5  # a lone spike on one channel must survive every level
    data[250_000, 0] = -9.0
    return data


def brute_force_bins(samples, env):
    """Recompute each bin's min/max from the frames it claims to cover."""
    frames = samples.shape[0]
    for x, lo, hi in zip(env.x, env.lo, env.hi, strict=True):
        a = int(x)
        b = min(frames, a + env.frames_per_bin)
        seg = samples[a:b]
        assert lo == seg.min(), (a, b)
        assert hi == seg.max(), (a, b)


def test_reduce_bins_groups_evenly():
    lo = np.arange(10, dtype=np.float32)
    hi = lo + 100
    rlo, rhi, group = reduce_bins(lo, hi, 4)
    assert group == 3
    np.testing.assert_array_equal(rlo, [0, 3, 6, 9])
    np.testing.assert_array_equal(rhi, [102, 105, 108, 109])
    same_lo, _, one = reduce_bins(lo, hi, 50)
    assert one == 1
    assert same_lo is lo


def test_minmax_envelope_raw():
    data = np.array([[0, 1], [2, -3], [4, 5], [-6, 7], [8, 9]], dtype=np.float32)
    env = minmax_envelope(data, 0, 5, 2)
    assert env.frames_per_bin == 3
    np.testing.assert_array_equal(env.x, [0, 3])
    np.testing.assert_array_equal(env.lo, [-3, -6])
    np.testing.assert_array_equal(env.hi, [5, 9])


def test_pyramid_levels(samples):
    pyramid = EnvelopePyramid(samples, base_block=256)
    assert len(pyramid.levels[0][0]) == -(-samples.shape[0] // 256)
    assert len(pyramid.levels[-1][0]) == 1
    lo, hi = pyramid.levels[-1]
    assert lo[0] == samples.min()
    assert hi[0] == samples.max() == 7.5
    for level in range(1, len(pyramid.levels)):
        assert len(pyramid.levels[level][0]) == -(-len(pyramid.levels[level - 1][0]) // 2)
    assert pyramid.nbytes < samples.nbytes / 50


@pytest.mark.parametrize(
    ("start", "stop", "bins"),
    [
        (0, 300_017, 2000),  # whole clip: pyramid level
        (0, 300_017, 100),  # coarser level
        (1_000, 299_000, 500),
        (123_000, 124_000, 2000),  # short range: raw samples
        (100_000, 100_050, 2000),  # deeper zoom than bins: one frame per bin
        (299_990, 300_017, 10),  # tail
        (0, 300_017, 1),
    ],
)
def test_pyramid_matches_brute_force(samples, start, stop, bins):
    pyramid = EnvelopePyramid(samples, base_block=256)
    env = pyramid.envelope(start, stop, bins)
    assert 0 < len(env) <= bins
    brute_force_bins(samples, env)
    # The bins cover the requested range (possibly reaching one block beyond each edge).
    assert env.x[0] <= start
    assert env.x[-1] + env.frames_per_bin >= stop
    # Resolution: with enough frames the result isn't much coarser than asked for.
    if stop - start >= 2 * bins:
        assert len(env) > bins / 4


def test_spike_visible_at_every_zoom(samples):
    pyramid = EnvelopePyramid(samples, base_block=64)
    for bins in (1, 10, 100, 1000, 5000):
        env = pyramid.envelope(0, samples.shape[0], bins)
        assert env.hi.max() == 7.5
        assert env.lo.min() == -9.0


def test_out_of_range_and_empty_queries(samples):
    pyramid = EnvelopePyramid(samples)
    assert len(pyramid.envelope(10, 10, 100)) == 0
    assert len(pyramid.envelope(500_000, 600_000, 100)) == 0
    assert len(pyramid.envelope(0, 100, 0)) == 0
    env = pyramid.envelope(-50, 100, 1000)  # clamped to [0, 100)
    assert env.x[0] == 0
    assert len(env) == 100


def test_empty_samples():
    pyramid = EnvelopePyramid(np.zeros((0, 2), dtype=np.float32))
    assert pyramid.levels == []
    assert len(pyramid.envelope(0, 100, 10)) == 0


def test_small_clip_shorter_than_one_block():
    data = np.array([[0.1, -0.2], [0.3, 0.0]], dtype=np.float32)
    pyramid = EnvelopePyramid(data, base_block=256)
    assert len(pyramid.levels) == 1
    env = pyramid.envelope(0, 2, 2000)
    np.testing.assert_allclose(env.lo, [-0.2, 0.0])
    np.testing.assert_allclose(env.hi, [0.1, 0.3])


def test_mono_input():
    data = np.linspace(-1, 1, 10_000, dtype=np.float32)
    env = EnvelopePyramid(data, base_block=16).envelope(0, 10_000, 50)
    assert env.lo[0] == pytest.approx(-1.0)
    assert env.hi[-1] == pytest.approx(1.0)


def test_zigzag_interleaves_min_and_max():
    data = np.array([[0, 1], [2, -3], [4, 5], [-6, 7]], dtype=np.float32)
    env = minmax_envelope(data, 0, 4, 2)
    t, y = env.zigzag(sample_rate=2.0)
    np.testing.assert_array_equal(t, [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(y, [-3, 2, -6, 7])


def test_pyramid_peak(samples):
    assert EnvelopePyramid(samples).peak == 9.0
    assert EnvelopePyramid(np.zeros((0, 2), dtype=np.float32)).peak == 0.0
    assert EnvelopePyramid(np.full((10, 2), -0.5, dtype=np.float32)).peak == 0.5


# ------------------------------------------------------------------------- partial refresh


def assert_levels_equal(a, b):
    assert len(a.levels) == len(b.levels)
    for (lo, hi), (flo, fhi) in zip(a.levels, b.levels, strict=True):
        np.testing.assert_array_equal(lo, flo)
        np.testing.assert_array_equal(hi, fhi)


@pytest.mark.parametrize("frames", [256 * 37, 256 * 37 + 100, 1000, 5])
@pytest.mark.parametrize(
    ("start", "stop"),
    [(0, 1), (0, 10**9), (300, 1200), (-5, 3), (900, 950)],
)
def test_refresh_matches_a_fresh_build(frames, start, stop):
    rng = np.random.default_rng(frames)
    samples = rng.uniform(-1, 1, (frames, 2)).astype(np.float32)
    pyramid = EnvelopePyramid(samples)
    a, b = max(0, start), min(frames, stop)
    samples[a:b] = rng.uniform(-3, 3, (max(0, b - a), 2))  # bigger values: the peak changes
    pyramid.refresh(start, stop)
    assert_levels_equal(pyramid, EnvelopePyramid(samples))
    assert pyramid.peak == pytest.approx(float(np.abs(samples).max()))


def test_refresh_after_silencing_lowers_the_peak():
    samples = np.full((256 * 64, 2), 0.25, np.float32)
    samples[5000] = 2.0
    pyramid = EnvelopePyramid(samples)
    assert pyramid.peak == 2.0
    samples[4096:8192] = 0.0
    pyramid.refresh(4096, 8192)
    assert pyramid.peak == 0.25
    assert_levels_equal(pyramid, EnvelopePyramid(samples))


def test_refresh_empty_range_or_pyramid_is_a_no_op():
    empty = EnvelopePyramid(np.zeros((0, 2), np.float32))
    empty.refresh(0, 10)
    assert empty.levels == []
    samples = np.ones((1000, 2), np.float32)
    pyramid = EnvelopePyramid(samples)
    samples[:] = 5
    pyramid.refresh(500, 500)
    assert pyramid.peak == 1.0
