"""audio/buffers.py: large arrays from VirtualAlloc, released off the GIL."""

import contextlib
import gc
import random
import sys

import numpy as np
import pytest
from conftest import write_test_wav

from desktop_audio_recorder.audio import buffers
from desktop_audio_recorder.audio.clip import AudioClip
from desktop_audio_recorder.editing import commands as cmd
from desktop_audio_recorder.editing.history import NoChange
from desktop_audio_recorder.editing.session import EditSession

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="VirtualAlloc is Windows-only")


@pytest.fixture
def virtual(monkeypatch):
    """Every buffers.empty() goes through VirtualAlloc."""
    monkeypatch.setattr(buffers, "LARGE_BYTES", 1)
    gc.collect()
    assert buffers.wait_released()
    return buffers.stats()


def released_since(before):
    gc.collect()
    assert buffers.wait_released()
    now = buffers.stats()
    return now["released"] - before["released"], now["live_bytes"] - before["live_bytes"]


def test_small_arrays_are_plain_numpy():
    before = buffers.stats()
    a = buffers.empty((1000, 2))
    assert a.dtype == np.float32 and a.shape == (1000, 2)
    assert a.flags.owndata
    assert buffers.stats()["allocated"] == before["allocated"]


@windows_only
def test_large_array_is_usable(virtual):
    a = buffers.empty((100_000, 2))
    assert buffers.stats()["allocated"] == virtual["allocated"] + 1
    assert buffers.stats()["live_bytes"] == virtual["live_bytes"] + 800_000
    assert a.shape == (100_000, 2) and a.dtype == np.float32
    assert a.flags.c_contiguous and a.flags.writeable
    a[:] = 0.5
    assert float(a.sum()) == 100_000.0
    b = buffers.empty(10, dtype=np.float64)
    assert b.shape == (10,) and b.dtype == np.float64


@windows_only
def test_memory_lives_while_any_view_does(virtual):
    a = buffers.empty((50_000, 2))
    a[:] = 1.0
    view = a[100:200, 1]
    del a
    assert released_since(virtual) == (0, 400_000)
    assert float(view.sum()) == 100.0  # still valid memory
    del view
    assert released_since(virtual) == (1, 0)


@windows_only
def test_copy(virtual):
    src = np.random.default_rng(0).standard_normal((1234, 2)).astype(np.float32)
    out = buffers.copy(src[::-1])
    np.testing.assert_array_equal(out, src[::-1])
    assert out.flags.c_contiguous
    assert not np.shares_memory(out, src)


def test_zero_size():
    assert buffers.empty((0, 2)).shape == (0, 2)


@windows_only
def test_clip_load_reads_into_a_buffer(virtual, tmp_path):
    stereo = AudioClip.load(write_test_wav(tmp_path / "s.wav", 0.5))
    assert buffers.stats()["allocated"] == virtual["allocated"] + 1
    assert stereo.frames == 24000 and stereo.samples.flags.c_contiguous
    mono = AudioClip.load(write_test_wav(tmp_path / "m.wav", 0.5, channels=1))
    np.testing.assert_array_equal(mono.samples, stereo.samples)
    eight = AudioClip.load(write_test_wav(tmp_path / "8.wav", 0.25, channels=8))
    assert eight.samples.shape == (12000, 2)
    del stereo, mono, eight
    released, live = released_since(virtual)
    assert released >= 3 and live == 0


@windows_only
@pytest.mark.parametrize("seed", range(3))
def test_edit_sessions_on_virtual_buffers_round_trip_and_free_everything(virtual, seed):
    rng = random.Random(seed)
    data = np.random.default_rng(seed).uniform(-1.2, 1.2, (5000, 2)).astype(np.float32)
    session = EditSession(AudioClip(buffers.copy(data), 48000))
    for _ in range(40):
        frames = session.clip.frames
        a = rng.randrange(0, frames)
        b = rng.randrange(a + 1, frames + 1)
        kind = rng.choice(["delete", "crop", "insert", "gain", "fade"])
        if kind == "delete" and (a, b) != (0, frames):
            command = cmd.Delete(a, b)
        elif kind == "crop":
            command = cmd.Crop(a, b)
        elif kind == "insert":
            command = cmd.InsertSilence(a, 100)
        elif kind == "gain":
            command = cmd.Gain(a, b, 3.0)
        else:
            command = cmd.FadeOut(a, b)
        with contextlib.suppress(NoChange):
            session.apply(command)
        if rng.random() < 0.3 and session.history.can_undo:
            session.undo()
    while session.history.can_undo:
        session.undo()
    np.testing.assert_array_equal(session.clip.samples, data)
    del session, command
    released, live = released_since(virtual)
    assert released > 40
    assert live == 0  # nothing leaked


def test_history_waits_for_pending_releases_before_large_allocations(monkeypatch):
    from desktop_audio_recorder.editing.history import EditHistory

    waits = []
    monkeypatch.setattr(buffers, "wait_released", lambda timeout=5.0: waits.append(1) or True)
    monkeypatch.setattr(buffers, "LARGE_BYTES", 30_000)
    h = EditHistory(max_bytes=50_000)
    x = np.zeros((4800, 2), np.float32)  # 38 400 bytes: "large" here
    x = h.execute(cmd.Silence(0, 100), x)  # small copy, not structural
    assert waits == []
    x = h.execute(cmd.Gain(0, 4800, 3, selected=False), x)  # large undo copy
    assert waits == [1]
    x = h.execute(cmd.Delete(0, 10), x)  # small copy, but a clip-sized result
    assert len(waits) == 2
    x, _ = h.undo(x)  # re-inserting allocates a clip-sized array too
    x, _ = h.redo(x)
    assert len(waits) == 4
    x, _ = h.undo(x)
    x, _ = h.undo(x)  # Gain: reverts in place
    assert len(waits) == 5
