"""editing/spill.py: undo buffers moved to disk, and EditHistory/EditSession using them."""

import gc
import random

import numpy as np
import pytest

from desktop_audio_recorder.audio.clip import AudioClip
from desktop_audio_recorder.editing import commands as cmd
from desktop_audio_recorder.editing import spill as spill_module
from desktop_audio_recorder.editing.history import DISK_READ_COST, EditHistory, NoChange
from desktop_audio_recorder.editing.session import EditSession
from desktop_audio_recorder.editing.spill import Payload, SpillStore

RATE = 48000


def noise(frames=4800, seed=0, scale=1.2):
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, (frames, 2)).astype(np.float32)


def same(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


@pytest.fixture
def store(tmp_path):
    s = SpillStore(tmp_path / "temp", min_spill_bytes=1, min_free_bytes=0)
    yield s
    s.close()


def files(store):
    return sorted(p.name for p in store.folder.iterdir() if p.suffix == ".f32")


# ------------------------------------------------------------------------- Payload


def test_payload_in_memory():
    x = noise(100)
    p = Payload(x)
    assert (p.nbytes, p.disk_bytes, p.frames, p.state) == (x.nbytes, 0, 100, "memory")
    out = np.empty_like(x)
    p.read_into(out)
    same(out, x)
    assert p.array() is x and p.take() is x
    with pytest.raises(ValueError):
        p.read_into(np.empty((99, 2), np.float32))


def test_spill_read_take_and_release(store):
    x = noise(1000)
    expected = x.copy()
    p = Payload(x)
    del x
    assert p.spill(store)
    assert p.nbytes == 0 and p.disk_bytes == expected.nbytes  # counted as gone at once
    assert p.wait(5) and p.state == "disk"
    assert files(store) == ["000001.f32"] and store.used_bytes == expected.nbytes
    out = np.empty_like(expected)
    p.read_into(out)
    same(out, expected)
    same(p.array(), expected)
    assert p.state == "disk"  # reading doesn't bring it back
    taken = p.take()
    same(taken, expected)
    assert p.state == "memory" and p.nbytes == expected.nbytes and p.disk_bytes == 0
    assert files(store) == [] and store.used_bytes == 0
    assert p.take() is taken


def test_read_into_a_slice_of_the_clip(store):
    clip = noise(5000)
    region = clip[1000:3000].copy()
    p = Payload(region.copy())
    p.spill(store)
    p.wait(5)
    clip[1000:3000] = 0
    p.read_into(clip[1000:3000])
    same(clip[1000:3000], region)


def test_discard_deletes_the_file(store):
    p = Payload(noise(100))
    p.spill(store)
    p.wait(5)
    p.discard()
    assert files(store) == [] and store.used_bytes == 0
    with pytest.raises(RuntimeError):
        p.read_into(np.empty((100, 2), np.float32))
    with pytest.raises(RuntimeError):
        p.take()


def test_garbage_collection_deletes_the_file(store):
    p = Payload(noise(100))
    p.spill(store)
    p.wait(5)
    assert files(store)
    del p
    gc.collect()
    assert files(store) == [] and store.used_bytes == 0


def test_take_or_discard_while_writing(store, monkeypatch):
    import threading

    gate = threading.Event()
    real = spill_module._write_array

    def slow(path, array):
        gate.wait(5)
        real(path, array)

    monkeypatch.setattr(spill_module, "_write_array", slow)
    x = noise(200)
    taken_from = Payload(x)
    discarded = Payload(noise(200, seed=1))
    assert taken_from.spill(store) and discarded.spill(store)
    out = np.empty_like(x)
    taken_from.read_into(out)  # still readable from memory while it is being written
    same(out, x)
    assert taken_from.take() is x  # back in memory, the pending file is dropped
    discarded.discard()
    gate.set()
    assert store.wait_idle(5)
    assert taken_from.state == "memory" and discarded.state == "gone"
    assert files(store) == [] and store.used_bytes == 0


def test_write_failure_keeps_it_in_memory(store, monkeypatch):
    def boom(path, array):
        raise OSError("disk full")

    monkeypatch.setattr(spill_module, "_write_array", boom)
    x = noise(100)
    p = Payload(x)
    assert p.spill(store)
    p.wait(5)
    assert p.state == "memory" and p.unspillable and p.nbytes == x.nbytes
    assert p.array() is x and store.used_bytes == 0
    assert not p.spill(store) and not p.spillable(1)


def test_budget_and_free_space(tmp_path):
    s = SpillStore(tmp_path, max_bytes=2000, min_free_bytes=0, min_spill_bytes=1)
    try:
        a, b = Payload(noise(200)), Payload(noise(100))  # 1600 and 800 bytes
        assert a.spill(s)
        assert not b.spill(s)  # over the budget
        assert b.state == "memory"
    finally:
        s.close()
    low = SpillStore(tmp_path, min_free_bytes=1 << 62, min_spill_bytes=1)  # "the disk is full"
    try:
        assert not Payload(noise(10)).spill(low)
    finally:
        low.close()


def test_small_payloads_stay_in_memory(tmp_path):
    p = Payload(noise(10))
    assert not p.spillable()  # the default minimum is tens of MB
    assert p.spillable(1)


def test_close_removes_the_folder_and_stale_folders_go(tmp_path):
    live = SpillStore(tmp_path, min_spill_bytes=1, min_free_bytes=0)
    p = Payload(noise(100))
    p.spill(live)
    p.wait(5)
    dead = tmp_path / "undo-99999-deadbeef"
    dead.mkdir()
    (dead / "000001.f32").write_bytes(b"x" * 10)
    (dead / ".lock").write_bytes(b"")
    (tmp_path / "recording-1.wav").write_bytes(b"keep")
    # The live store's lock file is open, so only the dead folder goes.
    assert SpillStore.remove_stale(tmp_path) == 1
    assert not dead.exists() and live.folder.exists()
    assert (tmp_path / "recording-1.wav").exists()
    folder = live.folder
    live.close()
    assert not folder.exists()
    live.close()  # idempotent
    assert not Payload(noise(100)).spill(live)  # closed


def test_no_folder_until_needed(tmp_path):
    s = SpillStore(tmp_path / "t", min_spill_bytes=1)
    assert s.folder is None
    s.close()
    assert not (tmp_path / "t").exists()


# ------------------------------------------------------------------------- history


def gain_steps(history, x, n):
    for i in range(n):
        x = history.execute(cmd.Gain(0, len(x), 1.0 + i, selected=False), x)
    return x


def test_history_spills_instead_of_evicting(store):
    x = noise(4800)  # 38.4 kB per whole-clip undo copy
    original = x.copy()
    h = EditHistory(max_bytes=50_000, spill=store)
    x = gain_steps(h, x, 6)
    store.wait_idle()
    assert h.undo_count == 6 and h.evicted == 0
    assert h.spilled == 5
    assert h.nbytes == 38_400  # only the newest entry in memory
    assert h.disk_bytes == 5 * 38_400 and len(files(store)) == 5
    for _ in range(6):
        x, _ = h.undo(x)
    same(x, original)
    assert not h.can_undo and h.redo_count == 6
    for _ in range(6):
        x, _ = h.redo(x)
    for _ in range(6):
        x, _ = h.undo(x)
    same(x, original)


def test_disk_budget_evicts_the_oldest(tmp_path):
    s = SpillStore(tmp_path, max_bytes=2 * 38_400, min_spill_bytes=1, min_free_bytes=0)
    try:
        h = EditHistory(max_bytes=50_000, spill=s)
        gain_steps(h, noise(4800), 6)
        s.wait_idle()
        # Two on disk, the newest in memory; the oldest three were dropped to make room.
        assert h.undo_count == 3 and h.evicted == 3
        assert h.disk_bytes == 2 * 38_400 == s.used_bytes
        assert len(files(s)) == 2
    finally:
        s.close()


def test_without_a_store_it_evicts_as_before():
    h = EditHistory(max_bytes=50_000)
    gain_steps(h, noise(4800), 4)
    assert h.undo_count == 1 and h.evicted == 3 and h.spilled == 0


def test_dropped_entries_delete_their_files(store):
    h = EditHistory(max_bytes=50_000, spill=store)
    x = gain_steps(h, noise(4800), 4)
    store.wait_idle()
    for _ in range(3):
        x, _ = h.undo(x)
    h.execute(cmd.Silence(0, 10), x)  # drops the redo stack (two spilled entries)
    store.wait_idle()
    assert h.redo_count == 0
    assert len(files(store)) == 1
    h.clear()
    assert files(store) == [] and store.used_bytes == 0


def test_work_frames_counts_reading_from_disk(store):
    h = EditHistory(max_bytes=50_000, spill=store)
    x = gain_steps(h, noise(4800), 2)
    store.wait_idle()
    newest, oldest = h.next_undo, list(h._undo)[0]
    assert newest.work_frames(4800) == 4800
    assert oldest.work_frames(4800) == 4800 + DISK_READ_COST * 4800
    session = EditSession(AudioClip(x, RATE), spill=store)
    assert session.work_frames("undo") == 0


@pytest.mark.parametrize(
    "make",
    [
        lambda n: cmd.Delete(100, n - 100),
        lambda n: cmd.Crop(100, 200),
        lambda n: cmd.Silence(0, n, selected=False),
        lambda n: cmd.ReplaceRegion(0, n, noise(n, seed=9), "Reduce noise", selected=False),
    ],
    ids=["delete", "crop", "silence", "replace"],
)
def test_each_kind_of_payload_survives_the_disk(store, make):
    x = noise(4800)
    original = x.copy()
    h = EditHistory(max_bytes=1, spill=store)
    x = h.execute(make(4800), x)
    after = x.copy()
    x = h.execute(cmd.Gain(0, 10, 3.0), x)  # pushes the first entry out to disk
    store.wait_idle()
    first = list(h._undo)[0]
    assert first.disk_bytes > 0 and first.nbytes == 0
    x, _ = h.undo(x)
    same(x, after)
    x, _ = h.undo(x)
    same(x, original)
    x, _ = h.redo(x)
    same(x, after)
    x, _ = h.undo(x)
    same(x, original)


@pytest.mark.parametrize("seed", range(6))
def test_random_sequences_with_spilling_are_bit_exact(store, seed):
    """The same model check as test_history's, with almost every undo buffer on disk (or on its
    way there) and the writes racing the edits."""
    rng = random.Random(seed)
    from test_history import random_command

    session = EditSession(AudioClip(noise(3000, seed=seed), RATE), max_bytes=1, spill=store)
    snapshots = [session.clip.samples.copy()]
    position = 0
    for _ in range(50):
        roll = rng.random()
        if rng.random() < 0.3:
            store.wait_idle()
        if roll < 0.5 or position == 0 and position == len(snapshots) - 1:
            try:
                session.apply(random_command(rng, session.clip.frames))
            except NoChange:
                continue
            del snapshots[position + 1 :]
            snapshots.append(session.clip.samples.copy())
            position += 1
        elif roll < 0.8 and position > 0:
            session.undo()
            position -= 1
        elif position < len(snapshots) - 1:
            session.redo()
            position += 1
        else:
            continue
        same(session.clip.samples, snapshots[position])
        assert session.history.undo_count == position
    assert session.history.evicted == 0
    assert session.history.spilled > 0
    while session.history.can_undo:
        session.undo()
        position -= 1
        same(session.clip.samples, snapshots[position])
    session.close()
    store.wait_idle()
    assert files(store) == [] and store.used_bytes == 0


def test_wait_pending_counts_writes_in_flight(store, monkeypatch):
    import threading

    gate = threading.Event()
    real = spill_module._write_array

    def slow(path, array):
        gate.wait(5)
        real(path, array)

    monkeypatch.setattr(spill_module, "_write_array", slow)
    a, b = Payload(noise(100)), Payload(noise(100, seed=1))
    assert a.spill(store) and b.spill(store)
    assert store.pending == 2
    assert not store.wait_pending(1, timeout=0.05)  # both still on their way
    gate.set()
    assert store.wait_pending(0, timeout=5) and store.pending == 0
    assert a.state == b.state == "disk"
