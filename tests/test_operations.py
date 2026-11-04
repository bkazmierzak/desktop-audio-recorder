"""editing/operations.py: pure ndarray edits."""

import math

import numpy as np
import pytest

from desktop_audio_recorder.editing import operations as ops


def ramp_clip(frames=1000):
    """Distinct values per frame and channel, some overs (> 1.0) included."""
    left = np.linspace(-1.5, 1.5, frames, dtype=np.float32)
    return np.stack([left, -0.5 * left], axis=1)


@pytest.fixture
def x():
    return ramp_clip()


# ------------------------------------------------------------------------- helpers


@pytest.mark.parametrize(("start", "stop"), [(-1, 5), (5, 4), (0, 1001), (1001, 1001)])
def test_check_range_rejects_bad_ranges(start, stop):
    with pytest.raises(ValueError):
        ops.check_range(1000, start, stop)


def test_db_conversions():
    assert ops.db_to_gain(0) == 1.0
    assert ops.db_to_gain(20) == pytest.approx(10.0)
    assert ops.db_to_gain(-6) == pytest.approx(0.501187, rel=1e-5)
    assert ops.gain_to_db(ops.db_to_gain(-3.5)) == pytest.approx(-3.5)
    assert ops.gain_to_db(0) == -math.inf


def test_peak_uses_true_value_including_overs(x):
    assert ops.peak(x) == pytest.approx(1.5)
    assert ops.peak(x, 400, 600) == pytest.approx(np.abs(x[400:600]).max())
    assert ops.peak(x, 10, 10) == 0.0
    neg = np.zeros((10, 2), np.float32)
    neg[3, 1] = -2.0
    assert ops.peak(neg) == 2.0


def test_normalize_factor():
    assert ops.normalize_factor(0.0) is None
    assert ops.normalize_factor(math.nan) is None
    assert ops.normalize_factor(1.0) == pytest.approx(ops.db_to_gain(-1.0))
    assert ops.normalize_factor(0.25, -6.0) == pytest.approx(ops.db_to_gain(-6.0) / 0.25)


# ------------------------------------------------------------------------- length-changing


@pytest.mark.parametrize(("a", "b"), [(100, 200), (0, 300), (700, 1000), (0, 1000), (500, 500)])
def test_delete(x, a, b):
    out = ops.delete(x, a, b)
    np.testing.assert_array_equal(out, np.concatenate([x[:a], x[b:]]))
    assert out.shape == (1000 - (b - a), 2)
    assert out.dtype == np.float32
    assert out.flags.c_contiguous


@pytest.mark.parametrize(("a", "b"), [(100, 200), (0, 300), (700, 1000), (0, 1000), (5, 5)])
def test_crop(x, a, b):
    out = ops.crop(x, a, b)
    np.testing.assert_array_equal(out, x[a:b])
    assert not np.shares_memory(out, x)


@pytest.mark.parametrize("at", [0, 1, 500, 1000])
def test_insert_silence(x, at):
    out = ops.insert_silence(x, at, 48)
    assert out.shape == (1048, 2)
    np.testing.assert_array_equal(out[:at], x[:at])
    assert not out[at : at + 48].any()
    np.testing.assert_array_equal(out[at + 48 :], x[at:])
    np.testing.assert_array_equal(ops.insert_silence(x, at, 0), x)


def test_insert_silence_rejects_bad_input(x):
    with pytest.raises(ValueError):
        ops.insert_silence(x, 1001, 10)
    with pytest.raises(ValueError):
        ops.insert_silence(x, 10, -1)


def test_insert_segment_restores_delete(x):
    removed = x[300:420].copy()
    np.testing.assert_array_equal(ops.insert(ops.delete(x, 300, 420), 300, removed), x)


def test_operations_on_an_empty_clip():
    empty = np.zeros((0, 2), np.float32)
    assert ops.delete(empty, 0, 0).shape == (0, 2)
    assert ops.crop(empty, 0, 0).shape == (0, 2)
    assert ops.insert_silence(empty, 0, 3).shape == (3, 2)
    assert ops.silence(empty, 0, 0).shape == (0, 2)
    assert ops.normalize(empty, 0, 0).shape == (0, 2)
    assert ops.peak(empty) == 0.0


# ------------------------------------------------------------------------- length-preserving


def test_silence(x):
    out = ops.silence(x, 100, 250)
    assert not out[100:250].any()
    np.testing.assert_array_equal(out[:100], x[:100])
    np.testing.assert_array_equal(out[250:], x[250:])
    np.testing.assert_array_equal(x, ramp_clip())  # input untouched


@pytest.mark.parametrize("db", [6.0, -12.0, 0.0])
def test_gain_on_region(x, db):
    out = ops.gain(x, 200, 400, db)
    np.testing.assert_allclose(out[200:400], x[200:400] * ops.db_to_gain(db), rtol=1e-6)
    np.testing.assert_array_equal(out[:200], x[:200])
    np.testing.assert_array_equal(out[400:], x[400:])


def test_gain_does_not_clip(x):
    out = ops.gain(x, 0, 1000, 6.0)
    assert out.max() > 2.9  # 1.5 * 2: overs are kept for later normalize/export


def test_fade_in_shape():
    ones = np.ones((10, 2), np.float32)
    out = ops.fade_in(ones, 0, 10)
    np.testing.assert_allclose(out[:, 0], np.arange(10) / 10)
    np.testing.assert_array_equal(out[:, 0], out[:, 1])


def test_fade_out_mirrors_fade_in():
    ones = np.ones((10, 2), np.float32)
    np.testing.assert_array_equal(ops.fade_out(ones, 0, 10), ops.fade_in(ones, 0, 10)[::-1])
    assert ops.fade_out(ones, 0, 10)[-1, 0] == 0.0
    assert ops.fade_out(ones, 0, 10)[0, 0] == pytest.approx(0.9)


def test_fades_only_touch_the_region(x):
    for fn in (ops.fade_in, ops.fade_out):
        out = fn(x, 300, 500)
        np.testing.assert_array_equal(out[:300], x[:300])
        np.testing.assert_array_equal(out[500:], x[500:])
        assert np.all(np.abs(out[300:500]) <= np.abs(x[300:500]))


def test_one_frame_fades():
    one = np.ones((1, 2), np.float32)
    assert ops.fade_in(one, 0, 1)[0, 0] == 0.0
    assert ops.fade_out(one, 0, 1)[0, 0] == 0.0


def test_fade_across_chunks_matches_single_ramp(monkeypatch):
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, (10_000, 2)).astype(np.float32)
    whole_in = ops.fade_in(x, 0, 10_000)
    whole_out = ops.fade_out(x, 0, 10_000)
    monkeypatch.setattr(ops, "CHUNK_FRAMES", 777)
    np.testing.assert_array_equal(ops.fade_in(x, 0, 10_000), whole_in)
    np.testing.assert_array_equal(ops.fade_out(x, 0, 10_000), whole_out)


def test_fade_ramp_values():
    np.testing.assert_allclose(ops.fade_ramp(0, 4, 4, True), [0, 0.25, 0.5, 0.75])
    np.testing.assert_allclose(ops.fade_ramp(0, 4, 4, False), [0.75, 0.5, 0.25, 0])
    np.testing.assert_allclose(ops.fade_ramp(2, 4, 4, False), [0.25, 0])


def test_normalize_brings_overs_to_minus_one_dbfs(x):
    out = ops.normalize(x, 0, 1000)
    assert ops.peak(out) == pytest.approx(ops.db_to_gain(-1.0), rel=1e-6)
    assert ops.peak(out) < 1.0  # the overs are turned down


def test_normalize_quiet_selection_only(x):
    quiet = ops.gain(x, 0, 1000, -30.0)
    out = ops.normalize(quiet, 100, 200, target_db=-3.0)
    assert ops.peak(out, 100, 200) == pytest.approx(ops.db_to_gain(-3.0), rel=1e-6)
    np.testing.assert_array_equal(out[:100], quiet[:100])
    np.testing.assert_array_equal(out[200:], quiet[200:])


def test_normalize_silence_is_unchanged():
    zeros = np.zeros((100, 2), np.float32)
    np.testing.assert_array_equal(ops.normalize(zeros, 0, 100), zeros)


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: ops.silence(x, 0, 1000),
        lambda x: ops.gain(x, 0, 1000, 3),
        lambda x: ops.fade_in(x, 0, 1000),
        lambda x: ops.normalize(x, 0, 1000),
        lambda x: ops.silence(x, 999, 1000),  # last frame
        lambda x: ops.gain(x, 0, 1, 3),  # first frame
    ],
)
def test_whole_and_edge_regions_keep_length_and_input(x, fn):
    before = x.copy()
    out = fn(x)
    assert out.shape == x.shape
    assert out.dtype == np.float32
    np.testing.assert_array_equal(x, before)


def test_empty_region_is_a_copy(x):
    for out in (ops.silence(x, 5, 5), ops.gain(x, 5, 5, 6), ops.fade_in(x, 5, 5)):
        np.testing.assert_array_equal(out, x)
        assert not np.shares_memory(out, x)


def test_region_transforms_work_in_place(x):
    view = x[10:20]
    ops.silence_region(view)
    assert not x[10:20].any()
    ops.gain_region(x[30:40], 2.0)
    np.testing.assert_allclose(x[30:40], ramp_clip()[30:40] * 2)
    ops.fade_region(x[50:60], fade_in=True)
    assert x[50, 0] == 0.0
