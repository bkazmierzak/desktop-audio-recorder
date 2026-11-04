"""editing/commands.py, editing/history.py and editing/session.py."""

import random

import numpy as np
import pytest

from desktop_audio_recorder.audio.clip import AudioClip
from desktop_audio_recorder.audio.envelope import EnvelopePyramid
from desktop_audio_recorder.editing import commands as cmd
from desktop_audio_recorder.editing import operations as ops
from desktop_audio_recorder.editing.history import Command, EditHistory, NoChange, ViewHint
from desktop_audio_recorder.editing.session import APPLY, REDO, UNDO, EditSession

RATE = 48000


def noise(frames=4800, seed=0, scale=1.2):
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, (frames, 2)).astype(np.float32)  # includes overs


def same(a, b):
    """Bit-exact equality (shape, dtype and every bit, so -0.0 != 0.0 and NaNs compare)."""
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


# ------------------------------------------------------------------------- commands


COMMANDS = {
    "delete": lambda: cmd.Delete(1000, 2000),
    "delete_head": lambda: cmd.Delete(0, 1),
    "delete_tail": lambda: cmd.Delete(4000, 4800),
    "crop": lambda: cmd.Crop(1000, 2000),
    "crop_head": lambda: cmd.Crop(0, 10),
    "crop_tail": lambda: cmd.Crop(4790, 4800),
    "insert": lambda: cmd.InsertSilence(1200, 480, RATE),
    "insert_start": lambda: cmd.InsertSilence(0, 1),
    "insert_end": lambda: cmd.InsertSilence(4800, 480),
    "silence": lambda: cmd.Silence(100, 900),
    "fade_in": lambda: cmd.FadeIn(0, 4800, selected=False),
    "fade_out": lambda: cmd.FadeOut(4000, 4800),
    "gain_up": lambda: cmd.Gain(0, 4800, 6.0, selected=False),
    "gain_down": lambda: cmd.Gain(10, 20, -20.0),
    "normalize": lambda: cmd.Normalize(0, 4800, selected=False),
    "normalize_sel": lambda: cmd.Normalize(2000, 2400, -3.0),
}

EXPECTED = {
    "delete": lambda x: ops.delete(x, 1000, 2000),
    "delete_head": lambda x: ops.delete(x, 0, 1),
    "delete_tail": lambda x: ops.delete(x, 4000, 4800),
    "crop": lambda x: ops.crop(x, 1000, 2000),
    "crop_head": lambda x: ops.crop(x, 0, 10),
    "crop_tail": lambda x: ops.crop(x, 4790, 4800),
    "insert": lambda x: ops.insert_silence(x, 1200, 480),
    "insert_start": lambda x: ops.insert_silence(x, 0, 1),
    "insert_end": lambda x: ops.insert_silence(x, 4800, 480),
    "silence": lambda x: ops.silence(x, 100, 900),
    "fade_in": lambda x: ops.fade_in(x, 0, 4800),
    "fade_out": lambda x: ops.fade_out(x, 4000, 4800),
    "gain_up": lambda x: ops.gain(x, 0, 4800, 6.0),
    "gain_down": lambda x: ops.gain(x, 10, 20, -20.0),
    "normalize": lambda x: ops.normalize(x, 0, 4800),
    "normalize_sel": lambda x: ops.normalize(x, 2000, 2400, -3.0),
}


@pytest.mark.parametrize("name", list(COMMANDS))
def test_command_matches_operation_and_round_trips(name):
    original = noise()
    samples = original.copy()
    c = COMMANDS[name]()
    applied = c.apply(samples)
    same(applied, EXPECTED[name](original))
    reverted = c.revert(applied)
    same(reverted, original)
    redone = c.apply(reverted)
    same(redone, EXPECTED[name](original))
    same(c.revert(redone), original)


@pytest.mark.parametrize("name", list(COMMANDS))
def test_command_undo_storage_is_minimal(name):
    samples = noise()
    c = COMMANDS[name]()
    assert c.nbytes == 0  # nothing stored before apply
    c.apply(samples)
    expected = {
        "delete": 1000 * 8,
        "delete_head": 8,
        "delete_tail": 800 * 8,
        "crop": 4800 * 8,  # the previous array itself (no copy)
        "crop_head": 4800 * 8,
        "crop_tail": 4800 * 8,
        "insert": 0,
        "insert_start": 0,
        "insert_end": 0,
        "silence": 800 * 8,
        "fade_in": 4800 * 8,
        "fade_out": 800 * 8,
        "gain_up": 4800 * 8,
        "gain_down": 10 * 8,
        "normalize": 4800 * 8,
        "normalize_sel": 400 * 8,
    }[name]
    assert c.nbytes == expected


def test_region_commands_work_in_place_and_structural_ones_do_not():
    samples = noise()
    for c in (cmd.Silence(0, 10), cmd.Gain(10, 20, 3), cmd.FadeIn(20, 30), cmd.Normalize(30, 40)):
        assert c.apply(samples) is samples
        assert not c.structural
        assert c.work_frames(4800) == 10
    for c in (cmd.Delete(0, 10), cmd.Crop(0, 10), cmd.InsertSilence(0, 10)):
        before = samples.copy()
        out = c.apply(samples)
        assert out is not samples
        same(samples, before)
        assert c.structural
        assert c.work_frames(4800) == 4800


def test_crop_revert_returns_the_previous_array():
    samples = noise()
    c = cmd.Crop(10, 20)
    cropped = c.apply(samples)
    assert c.revert(cropped) is samples


def test_overs_survive_round_trips():
    samples = noise(scale=1.5)
    assert ops.peak(samples) > 1.4
    c = cmd.Gain(0, 4800, -60.0, selected=False)
    same(c.revert(c.apply(samples.copy())), samples)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: cmd.Gain(0, 100, 0.0), "0 dB"),
        (lambda: cmd.Crop(0, 4800), "whole clip"),
    ],
)
def test_no_change_raises_before_touching_anything(factory, message):
    samples = noise()
    before = samples.copy()
    c = factory()
    with pytest.raises(NoChange, match=message):
        c.apply(samples)
    same(samples, before)
    assert c.nbytes == 0


def test_normalize_silence_and_already_normalized():
    zeros = np.zeros((100, 2), np.float32)
    with pytest.raises(NoChange, match="silent"):
        cmd.Normalize(0, 100).apply(zeros)
    samples = noise()
    first = cmd.Normalize(0, 4800)
    samples = first.apply(samples)
    assert ops.peak(samples) == pytest.approx(ops.db_to_gain(-1), rel=1e-6)
    with pytest.raises(NoChange, match="Already normalized"):
        cmd.Normalize(0, 4800).apply(samples)


def test_normalize_factor_is_kept_for_redo():
    samples = noise()
    c = cmd.Normalize(0, 4800)
    c.apply(samples)
    factor = c.factor
    c.revert(samples)
    c.apply(samples)
    assert c.factor == factor


def test_commands_reject_bad_ranges():
    for bad in (
        lambda: cmd.Delete(5, 5),
        lambda: cmd.Crop(-1, 3),
        lambda: cmd.Silence(9, 2),
        lambda: cmd.InsertSilence(0, 0),
    ):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(ValueError):
        cmd.Delete(4000, 5000).apply(noise())  # past the end


def test_revert_before_apply_is_an_error():
    for c in (cmd.Silence(0, 2), cmd.Delete(0, 2), cmd.Crop(0, 2)):
        with pytest.raises(RuntimeError):
            c.revert(noise())


def test_labels():
    assert cmd.Gain(0, 1, 3).label == "Gain +3.0 dB"
    assert cmd.Gain(0, 1, -2.25).label == "Gain -2.2 dB"
    assert cmd.Normalize(0, 1).label == "Normalize to -1 dBFS"
    assert cmd.InsertSilence(0, 24000, RATE).label == "Insert 0.5 s of silence"
    assert cmd.Delete(0, 1).label == "Delete"


def test_view_hints():
    assert cmd.Delete(10, 20).hint(True) == ViewHint(None, 10)
    assert cmd.Delete(10, 20).hint(False) == ViewHint((10, 20), 10)
    assert cmd.Crop(10, 20).hint(True) == ViewHint(None, 0, -10)
    assert cmd.Crop(10, 20).hint(False) == ViewHint((10, 20), 10, 10)
    assert cmd.InsertSilence(5, 3).hint(True) == ViewHint((5, 8), 5)
    assert cmd.InsertSilence(5, 3).hint(False) == ViewHint(None, 5)
    assert cmd.Silence(1, 2).hint(True) == ViewHint((1, 2))
    assert cmd.Gain(0, 9, 1, selected=False).hint(False) == ViewHint(None)


# ------------------------------------------------------------------------- history


class Sized(Command):
    """Test command: appends a marker frame; holds ``size`` bytes of fake undo data."""

    def __init__(self, size, label="x"):
        self.size = size
        self.label = label

    @property
    def nbytes(self):
        return self.size

    def apply(self, samples):
        return np.concatenate([samples, np.ones((1, 2), np.float32)])

    def revert(self, samples):
        return samples[:-1]


def test_undo_redo_stacks():
    h = EditHistory()
    x = noise(10)
    assert not h.can_undo and not h.can_redo
    assert h.undo_label is None
    x = h.execute(Sized(1, "a"), x)
    x = h.execute(Sized(1, "b"), x)
    assert (h.undo_count, h.redo_count, h.undo_label) == (2, 0, "b")
    x, c = h.undo(x)
    assert c.label == "b"
    assert (h.undo_label, h.redo_label) == ("a", "b")
    x, c = h.redo(x)
    assert c.label == "b"
    assert x.shape[0] == 12
    x, _ = h.undo(x)
    x = h.execute(Sized(1, "c"), x)  # a new edit drops the redo stack
    assert not h.can_redo
    assert [h.undo_label, h.undo_count] == ["c", 2]


def test_undo_redo_with_nothing_raise_no_change():
    h = EditHistory()
    with pytest.raises(NoChange, match="undo"):
        h.undo(noise(4))
    with pytest.raises(NoChange, match="redo"):
        h.redo(noise(4))


def test_failed_execute_or_revert_leaves_history_unchanged():
    class Boom(Sized):
        def revert(self, samples):
            raise MemoryError

    h = EditHistory()
    x = h.execute(Boom(1), noise(4))
    with pytest.raises(MemoryError):
        h.undo(x)
    assert (h.undo_count, h.redo_count) == (1, 0)
    with pytest.raises(NoChange):
        h.execute(cmd.Gain(0, 4, 0.0), x)
    assert h.undo_count == 1


def test_memory_cap_drops_oldest_entries():
    h = EditHistory(max_bytes=100)
    x = noise(1)
    for i in range(5):
        x = h.execute(Sized(30, f"e{i}"), x)
    assert h.undo_count == 3  # 90 bytes
    assert h.nbytes == 90
    assert h.evicted == 2
    # Undo all that is left; the evicted edits can't be undone.
    for expected in ("e4", "e3", "e2"):
        x, c = h.undo(x)
        assert c.label == expected
    assert not h.can_undo
    assert x.shape[0] == 3  # the frames from e0 and e1 stay


def test_newest_entry_is_kept_even_over_the_cap():
    h = EditHistory(max_bytes=100)
    x = h.execute(Sized(40), noise(1))
    x = h.execute(Sized(500, "huge"), x)
    assert h.undo_count == 1 and h.undo_label == "huge"
    assert h.nbytes == 500
    x = h.execute(Sized(10, "small"), x)
    assert h.undo_count == 1 and h.undo_label == "small"  # the huge one is dropped next


def test_redo_entries_count_towards_memory():
    h = EditHistory(max_bytes=100)
    x = h.execute(Sized(60), noise(1))
    x, _ = h.undo(x)
    assert h.nbytes == 60
    h.clear()
    assert h.nbytes == 0 and not h.can_redo


def test_real_commands_under_a_memory_cap():
    x = noise(48000)  # 384 kB
    h = EditHistory(max_bytes=200_000)
    x = h.execute(cmd.Silence(0, 12000), x)  # 96 kB
    x = h.execute(cmd.Gain(0, 12000, 3), x)  # 96 kB
    assert h.undo_count == 2
    x = h.execute(cmd.Delete(0, 6000), x)  # 48 kB: over the cap -> the silence is dropped
    assert h.undo_count == 2
    assert h.nbytes <= 200_000


# ------------------------------------------------------------------------- any-order round trips


def random_command(rng, frames):
    kind = rng.choice(
        ["delete", "crop", "insert", "silence", "fade_in", "fade_out", "gain", "normalize"]
    )
    a = rng.randrange(0, frames)
    b = rng.randrange(a + 1, frames + 1)
    if kind == "delete" and (a, b) != (0, frames):
        return cmd.Delete(a, b)
    if kind == "crop" and (a, b) != (0, frames):
        return cmd.Crop(a, b)
    if kind == "insert":
        return cmd.InsertSilence(rng.randrange(0, frames + 1), rng.randrange(1, 500))
    if kind == "silence":
        return cmd.Silence(a, b)
    if kind == "fade_in":
        return cmd.FadeIn(a, b)
    if kind == "fade_out":
        return cmd.FadeOut(a, b)
    if kind == "normalize":
        return cmd.Normalize(a, b)
    return cmd.Gain(a, b, rng.choice([-12.0, -3.0, 2.5, 6.0]))


@pytest.mark.parametrize("seed", range(12))
def test_random_edit_undo_redo_sequences_are_bit_exact(seed):
    """Model check: after any mix of edits, undos and redos the clip equals the snapshot the
    linear history says it should be."""
    rng = random.Random(seed)
    session = EditSession(AudioClip(noise(3000, seed=seed), RATE))
    snapshots = [session.clip.samples.copy()]  # state after each applied edit
    position = 0  # index into snapshots of the current state
    for _ in range(60):
        roll = rng.random()
        if roll < 0.5 or position == 0 and position == len(snapshots) - 1:
            command = random_command(rng, session.clip.frames)
            try:
                session.apply(command)
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
        assert session.history.redo_count == len(snapshots) - 1 - position
    # Unwind everything, then replay everything.
    while session.history.can_undo:
        session.undo()
        position -= 1
        same(session.clip.samples, snapshots[position])
    while session.history.can_redo:
        session.redo()
        position += 1
        same(session.clip.samples, snapshots[position])


# ------------------------------------------------------------------------- session


def test_session_region_edit_refreshes_the_pyramid_in_place():
    clip = AudioClip(noise(48000), RATE)
    session = EditSession(clip)
    pyramid = session.pyramid
    result = session.apply(cmd.Silence(10_000, 20_000))
    assert result.clip.samples is clip.samples
    assert result.pyramid is pyramid
    fresh = EnvelopePyramid(result.clip.samples)
    for (lo, hi), (flo, fhi) in zip(pyramid.levels, fresh.levels, strict=True):
        same(lo, flo)
        same(hi, fhi)


def test_session_structural_edit_rebuilds_the_pyramid():
    session = EditSession(AudioClip(noise(48000), RATE))
    old = session.pyramid
    result = session.apply(cmd.Delete(0, 24000))
    assert result.pyramid is not old
    assert result.pyramid.frames == 24000
    assert result.clip.frames == 24000
    assert result.clip.sample_rate == RATE


def test_session_results_and_messages():
    session = EditSession(AudioClip(noise(4800), RATE))
    r = session.apply(cmd.Gain(0, 4800, -6, selected=False))
    assert (r.action, r.message) == (APPLY, "Gain -6.0 dB")
    assert r.hint == ViewHint(None)
    r = session.undo()
    assert (r.action, r.message) == (UNDO, "Undid: Gain -6.0 dB")
    r = session.redo()
    assert (r.action, r.message) == (REDO, "Redid: Gain -6.0 dB")
    assert r.seconds >= 0
    with pytest.raises(NoChange):
        session.redo()
    with pytest.raises(ValueError):
        session.run("bogus")


def test_session_work_frames():
    session = EditSession(AudioClip(noise(4800), RATE))
    assert session.work_frames(UNDO) == 0
    assert session.work_frames(APPLY, cmd.Silence(0, 100)) == 100
    assert session.work_frames(APPLY, cmd.Delete(0, 100)) == 4800
    session.apply(cmd.Delete(0, 100))
    assert session.work_frames(UNDO) == 4700
    session.undo()
    assert session.work_frames(REDO) == 4800


def test_session_pyramid_tracks_peak_after_normalize():
    session = EditSession(AudioClip(noise(48000, scale=1.4), RATE))
    assert session.pyramid.peak > 1.3
    result = session.apply(cmd.Normalize(0, 48000, selected=False))
    assert result.pyramid.peak == pytest.approx(ops.db_to_gain(-1.0), rel=1e-6)
    result = session.undo()
    assert result.pyramid.peak > 1.3


# ------------------------------------------------------------------------- making room first


@pytest.mark.parametrize("name", list(COMMANDS))
def test_undo_bytes_predicts_nbytes(name):
    samples = noise()
    c = COMMANDS[name]()
    predicted = c.undo_bytes(samples)
    c.apply(samples)
    assert predicted == c.nbytes


def test_room_is_made_before_the_undo_copy():
    h = EditHistory(max_bytes=100_000)
    x = noise(4800)  # 38 400 bytes
    x = h.execute(cmd.Gain(0, 4800, 3, selected=False), x)
    x = h.execute(cmd.Gain(0, 4800, -3, selected=False), x)
    assert h.undo_count == 2  # 76 800 bytes
    seen = []

    class Spy(cmd.Gain):
        def apply(self, samples):
            seen.append(h.nbytes)  # what the history holds while this copies its region
            return super().apply(samples)

    x = h.execute(Spy(0, 4800, 1.0, selected=False), x)
    assert seen == [38_400]  # the oldest was dropped before the copy, not after
    assert h.undo_count == 2


def test_a_command_bigger_than_the_cap_clears_older_entries_first():
    h = EditHistory(max_bytes=10_000)
    x = h.execute(cmd.Silence(0, 100), noise(4800))
    x = h.execute(cmd.Normalize(0, 4800, selected=False), x)  # 38 400 bytes alone
    assert h.undo_count == 1 and h.undo_label.startswith("Normalize")
    assert h.evicted == 1


def test_no_change_evicts_nothing_and_keeps_redo():
    h = EditHistory(max_bytes=50_000)
    x = h.execute(cmd.Gain(0, 4800, 3, selected=False), noise(4800))
    x, _ = h.undo(x)
    with pytest.raises(NoChange):
        h.execute(cmd.Crop(0, 4800), x)
    with pytest.raises(NoChange):
        h.execute(cmd.Gain(0, 4800, 0.0), x)
    assert h.can_redo and h.evicted == 0


# ------------------------------------------------------------------------- ReplaceRegion


def test_replace_region_swaps_bit_exact_both_ways(monkeypatch):
    monkeypatch.setattr(cmd, "SWAP_CHUNK_FRAMES", 1000)  # several swap chunks
    samples = noise(10_000)
    original = samples.copy()
    replacement = noise(4_321, seed=7)
    expected = replacement.copy()
    command = cmd.ReplaceRegion(3_000, 7_321, replacement, "Noise reduction")
    assert command.nbytes == replacement.nbytes == command.undo_bytes(samples)
    assert command.region == (3_000, 7_321) and not command.structural
    out = command.apply(samples)
    assert out is samples
    same(samples[3_000:7_321], expected)
    same(samples[:3_000], original[:3_000])
    same(samples[7_321:], original[7_321:])
    assert command.nbytes == replacement.nbytes  # now it holds the originals
    command.revert(samples)
    same(samples, original)
    command.apply(samples)  # redo
    same(samples[3_000:7_321], expected)


def test_replace_region_rejects_mismatches_and_misuse():
    with pytest.raises(ValueError, match="doesn't fit"):
        cmd.ReplaceRegion(0, 100, np.zeros((99, 2), np.float32))
    samples = noise(1_000)
    command = cmd.ReplaceRegion(0, 100, np.zeros((100, 2), np.float64))
    with pytest.raises(ValueError, match="doesn't match"):
        EditHistory().execute(command, samples)
    command = cmd.ReplaceRegion(900, 1_100, np.zeros((200, 2), np.float32))
    with pytest.raises(ValueError):
        command.check(samples)  # past the end
    command = cmd.ReplaceRegion(0, 100, np.zeros((100, 2), np.float32))
    with pytest.raises(RuntimeError, match="revert before apply"):
        command.revert(samples)
    command.apply(samples)
    with pytest.raises(RuntimeError, match="applied twice"):
        command.apply(samples)


def test_replace_region_hints():
    selected = cmd.ReplaceRegion(10, 20, np.zeros((10, 2), np.float32))
    whole = cmd.ReplaceRegion(0, 10, np.zeros((10, 2), np.float32), selected=False)
    assert selected.hint(True) == ViewHint(selection=(10, 20))
    assert whole.hint(True) == ViewHint(selection=None)


def test_replace_region_in_a_session_with_undo_redo():
    clip = AudioClip(noise(48_000), RATE)
    original = clip.samples.copy()
    session = EditSession(clip)
    session.apply(cmd.Gain(0, 1000, 6.0))
    after_gain = session.clip.samples.copy()
    replacement = noise(20_000, seed=3)
    result = session.apply(cmd.ReplaceRegion(5_000, 25_000, replacement.copy(), "Noise reduction"))
    assert result.message == "Noise reduction"
    same(result.clip.samples[5_000:25_000], replacement)
    fresh = EnvelopePyramid(result.clip.samples)
    for (lo, hi), (flo, fhi) in zip(result.pyramid.levels, fresh.levels, strict=True):
        same(lo, flo)
        same(hi, fhi)
    assert session.history.undo_label == "Noise reduction"
    same(session.undo().clip.samples, after_gain)
    same(session.redo().clip.samples[5_000:25_000], replacement)
    session.undo()
    same(session.undo().clip.samples, original)


# ------------------------------------------------------------------------- position (saved state)


def gain(db, a=0, b=4800):
    return cmd.Gain(a, b, db)


def test_position_follows_undo_and_redo():
    h = EditHistory()
    x = noise()
    assert h.position == 0  # the original clip
    x = h.execute(gain(1.0), x)
    p1 = h.position
    x = h.execute(gain(2.0), x)
    p2 = h.position
    assert len({0, p1, p2}) == 3
    x, _ = h.undo(x)
    assert h.position == p1
    x, _ = h.undo(x)
    assert h.position == 0
    x, _ = h.redo(x)
    assert h.position == p1
    x, _ = h.redo(x)
    assert h.position == p2


def test_a_new_edit_after_undo_gets_a_new_position():
    h = EditHistory()
    x = h.execute(gain(1.0), noise())
    p1 = h.position
    x = h.execute(gain(2.0), x)
    p2 = h.position
    x, _ = h.undo(x)
    x = h.execute(gain(3.0), x)  # replaces the undone edit
    assert h.position not in (0, p1, p2)
    x, _ = h.undo(x)
    assert h.position == p1


def test_no_change_and_failed_edits_keep_the_position():
    h = EditHistory()
    x = h.execute(gain(1.0), noise())
    p = h.position
    with pytest.raises(NoChange):
        h.execute(gain(0.0), x)
    assert h.position == p


def test_position_survives_eviction():
    # Saved at the original; the oldest edits are evicted; undoing everything that is left
    # doesn't get back to the original, so the position must differ from 0.
    h = EditHistory(max_bytes=40_000)  # room for one 38 400-byte region copy
    x = noise()
    x = h.execute(gain(1.0), x)
    p1 = h.position
    x = h.execute(gain(2.0), x)  # evicts the first
    p2 = h.position
    assert h.evicted == 1
    x, _ = h.undo(x)
    assert not h.can_undo
    assert h.position == p1 != 0  # the content after the evicted edit
    x, _ = h.redo(x)
    assert h.position == p2


def test_clear_keeps_the_position():
    h = EditHistory()
    x = h.execute(gain(1.0), noise())
    p = h.position
    h.clear()
    assert h.position == p
    h.execute(gain(2.0), x)
    assert h.position not in (0, p)
