"""Editing in the main window: edit toolbar, shortcuts, undo/redo, playback after edits.

Offscreen, with the fake recorder (a 2.5 s, 48 kHz, 440 Hz sine at 0.5) and the fake player
backend.
"""

import threading

import numpy as np
import pytest
from conftest import FakeBackend, FakeDeviceManager, FakeRecorder
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from desktop_audio_recorder.audio.player import ClipPlayer
from desktop_audio_recorder.editing import commands
from desktop_audio_recorder.editing import operations as ops
from desktop_audio_recorder.ui.edit_toolbar import EditToolbar
from desktop_audio_recorder.ui.main_window import MainWindow
from desktop_audio_recorder.ui.state import AppState, EditControls
from desktop_audio_recorder.ui.transport_bar import format_elapsed

TIMEOUT = 3000
RATE = 48000
FRAMES = 120000


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def recorder(tmp_path):
    return FakeRecorder(tmp_path)


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def make_window(qtbot, recorder, backend):
    def make(**kwargs):
        kwargs.setdefault("recorder", recorder)
        kwargs.setdefault("device_manager", FakeDeviceManager())
        kwargs.setdefault("clock", Clock())
        kwargs.setdefault("player", ClipPlayer(backend))
        window = MainWindow(**kwargs)
        qtbot.addWidget(window)
        qtbot.waitUntil(lambda: window.device is not None, timeout=TIMEOUT)
        return window

    return make


def record(qtbot, window):
    qtbot.mouseClick(window.transport.record_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.state is AppState.RECORDING, timeout=TIMEOUT)
    qtbot.mouseClick(window.transport.stop_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)


@pytest.fixture
def window(make_window, qtbot):
    w = make_window()
    record(qtbot, w)
    return w


@pytest.fixture
def original(window):
    return window.clip.samples.copy()


def select(window, start, end):
    window.waveform.set_selection((start, end), emit=True)


def enabled(window):
    return {name: a.isEnabled() for name, a in window.edit_toolbar.edit_actions.items()}


def assert_consistent(window):
    """Window, player and waveform show the same clip."""
    assert window.player.clip is window.clip
    assert window.waveform.clip is window.clip
    assert window.session.clip is window.clip
    assert window.playback.position_label.text().endswith(format_elapsed(window.clip.duration))


def same(a, b):
    assert a.shape == b.shape and a.tobytes() == b.tobytes()


def show(qtbot, window):
    window.show()
    qtbot.waitExposed(window)
    window.activateWindow()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is window, timeout=TIMEOUT)


def play_and_pump(qtbot, window, backend, frames):
    qtbot.mouseClick(window.playback.play_button, Qt.MouseButton.LeftButton)
    assert window.state is AppState.PLAYING
    return backend.stream.pump(frames)


# ------------------------------------------------------------------------- toolbar widget


def test_toolbar_without_controls_is_all_disabled(qtbot):
    bar = EditToolbar()
    qtbot.addWidget(bar)
    assert not any(a.isEnabled() for a in bar.edit_actions.values())
    bar.set_controls(EditControls(*[True] * 11))
    assert all(a.isEnabled() for a in bar.edit_actions.values())
    bar.set_controls(None)
    assert not any(a.isEnabled() for a in bar.edit_actions.values())


def test_toolbar_shortcuts_and_tooltips(qtbot):
    bar = EditToolbar()
    qtbot.addWidget(bar)
    keys = {name: [k.toString() for k in a.shortcuts()] for name, a in bar.edit_actions.items()}
    assert keys["undo"] == ["Ctrl+Z"]
    assert keys["redo"] == ["Ctrl+Y", "Ctrl+Shift+Z"]
    assert keys["delete"] == ["Del"]
    assert bar.undo_action.toolTip() == "Nothing to undo (Ctrl+Z)"
    bar.set_history("Delete", "Gain +3.0 dB")
    assert bar.undo_action.toolTip() == "Undo: Delete (Ctrl+Z)"
    assert bar.redo_action.toolTip() == "Redo: Gain +3.0 dB (Ctrl+Y)"
    assert bar.delete_action.toolTip().endswith("(Del)")
    texts = [a.text() for a in bar.edit_actions.values()]
    assert all(t.isascii() or t.endswith("…") for t in texts)  # Segoe UI glyphs only


# ------------------------------------------------------------------------- availability


def test_toolbar_hidden_until_a_clip_is_loaded(make_window, qtbot):
    window = make_window()
    assert not window.edit_toolbar.isVisibleTo(window)
    assert not any(enabled(window).values())
    record(qtbot, window)
    assert window.edit_toolbar.isVisibleTo(window)


def test_actions_follow_selection_and_history(window):
    on = enabled(window)
    assert on == dict.fromkeys(on, False) | {
        "gain": True,
        "normalize": True,
        "insert_silence": True,
        "denoise": True,
    }
    select(window, 0.5, 1.0)
    on = enabled(window)
    assert on == dict.fromkeys(on, True) | {"undo": False, "redo": False}
    window.silence_selection()
    assert enabled(window)["undo"] and not enabled(window)["redo"]
    window.undo()
    assert not enabled(window)["undo"] and enabled(window)["redo"]
    assert window.edit_toolbar.redo_action.toolTip() == "Redo: Silence (Ctrl+Y)"
    window.waveform.clear_selection()
    assert not enabled(window)["delete"]


def test_editing_disabled_while_recording(window, qtbot):
    select(window, 0.5, 1.0)
    window.silence_selection()
    qtbot.mouseClick(window.transport.record_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.state is AppState.RECORDING, timeout=TIMEOUT)
    assert window.session is None  # the new recording discards the clip and its history
    assert not any(enabled(window).values())
    window.undo()  # e.g. from a stray shortcut: ignored
    assert window.state is AppState.RECORDING


# ------------------------------------------------------------------------- each operation


def test_delete_selection(window, original, backend, qtbot):
    select(window, 0.5, 1.0)
    window.edit_toolbar.delete_action.trigger()
    assert window.clip.frames == FRAMES - 24000
    same(window.clip.samples, ops.delete(original, 24000, 48000))
    assert_consistent(window)
    assert window.waveform.selection is None
    assert window.player.position == pytest.approx(0.5)
    assert window.waveform.playhead_position == pytest.approx(0.5)
    assert window.playback.position_label.text() == "00:00.5 / 00:02.0"
    assert window.status_text == "Delete"
    assert window.clip_info.text().startswith("Edited: 00:02.0")
    assert "not saved" in window.clip_info.text()
    # Playback continues from the cut straight into the audio that followed the selection.
    out = play_and_pump(qtbot, window, backend, 480)
    np.testing.assert_array_equal(out, original[48000:48480])


def test_delete_whole_clip_is_refused(window, original):
    select(window, 0.0, 2.5)
    window.delete_selection()
    assert window.clip.frames == FRAMES
    assert "whole clip" in window.status_text
    assert not window.session.history.can_undo


def test_crop_to_selection_keeps_the_view_on_the_audio(window, original, backend, qtbot):
    window.waveform.set_view_range(0.5, 1.5)
    select(window, 0.75, 1.25)
    window.crop_to_selection()
    assert window.clip.frames == 24000
    same(window.clip.samples, original[36000:60000])
    assert_consistent(window)
    assert window.waveform.view_range == pytest.approx((0.0, 0.5), abs=1e-6)
    assert window.player.position == 0.0
    out = play_and_pump(qtbot, window, backend, 480)
    np.testing.assert_array_equal(out, np.clip(original[36000:36480], -1, 1))
    window.pause_playback()
    window.undo()
    same(window.clip.samples, original)
    assert window.waveform.selection == pytest.approx((0.75, 1.25))
    # Shifted back so the audio stays put (the zoom was capped at the 0.5 s cropped clip).
    assert window.waveform.view_range == pytest.approx((0.75, 1.25), abs=1e-6)


def test_crop_whole_clip_changes_nothing(window):
    select(window, 0.0, 2.5)
    window.crop_to_selection()
    assert window.status_text == "The selection already covers the whole clip"
    assert not window.status_is_error
    assert not window.session.history.can_undo


def test_silence_selection_plays_silence(window, original, backend, qtbot):
    select(window, 0.5, 1.0)
    clip_before = window.clip
    window.silence_selection()
    assert window.clip is not clip_before  # a new clip object (sharing the edited array)
    assert not window.clip.samples[24000:48000].any()
    same(window.clip.samples[:24000], original[:24000])
    same(window.clip.samples[48000:], original[48000:])
    assert window.waveform.selection == pytest.approx((0.5, 1.0))  # kept
    assert window.player.selection == pytest.approx((0.5, 1.0))
    assert_consistent(window)
    # Waveform: the silenced part draws flat.
    lo, hi = window.session.pyramid.levels[0]
    assert not lo[94:187].any() and not hi[94:187].any()  # blocks inside 0.5-1.0 s
    assert hi[:93].max() == pytest.approx(0.5, abs=1e-3)
    # Play plays the (silenced) selection.
    out = play_and_pump(qtbot, window, backend, 4800)
    assert not out.any()


def test_fades(window, original):
    select(window, 0.0, 0.5)
    window.fade_in()
    same(window.clip.samples, ops.fade_in(original, 0, 24000))
    select(window, 2.0, 2.5)
    window.fade_out()
    same(window.clip.samples, ops.fade_out(ops.fade_in(original, 0, 24000), 96000, 120000))
    assert window.clip.samples[-1].tolist() == [0.0, 0.0]
    assert window.status_text == "Fade out"


def test_gain_on_selection_and_whole_clip(window, original):
    window.prompt_gain_db = lambda: 6.0
    window.edit_toolbar.gain_action.trigger()  # no selection: the whole clip
    same(window.clip.samples, ops.gain(original, 0, FRAMES, 6.0))
    assert window.waveform.selection is None
    assert window.status_text == "Gain +6.0 dB"
    assert window.session.pyramid.peak == pytest.approx(0.5 * ops.db_to_gain(6), rel=1e-4)
    select(window, 1.0, 2.0)
    window.apply_gain(-12.0)
    expected = ops.gain(ops.gain(original, 0, FRAMES, 6.0), 48000, 96000, -12.0)
    same(window.clip.samples, expected)
    assert window.waveform.selection == pytest.approx((1.0, 2.0))


def test_gain_prompt_cancelled_or_zero(window, original):
    window.prompt_gain_db = lambda: None
    window.apply_gain()
    window.apply_gain(0.0)
    same(window.clip.samples, original)
    assert window.status_text == "A gain of 0 dB changes nothing"
    assert not window.session.history.can_undo


def test_gain_can_push_over_full_scale_and_normalize_brings_it_back(window, original, backend):
    window.apply_gain(9.0)  # 0.5 -> 1.41
    assert window.session.pyramid.peak > 1.4
    assert window.waveform._vb.viewRange()[1][1] > 1.45  # the y range makes room for overs
    window.normalize()
    assert ops.peak(window.clip.samples) == pytest.approx(ops.db_to_gain(-1.0), rel=1e-5)
    assert window.status_text == "Normalize to -1 dBFS"
    assert window.waveform._vb.viewRange()[1][1] == pytest.approx(1.05)
    window.normalize()
    assert window.status_text == "Already normalized to -1 dBFS"


def test_normalize_selection_only(window, original):
    select(window, 0.0, 1.0)
    window.normalize()
    same(window.clip.samples, ops.normalize(original, 0, 48000))
    same(window.clip.samples[48000:], original[48000:])


def test_normalize_silence_does_nothing(window):
    select(window, 0.5, 1.0)
    window.silence_selection()
    window.normalize()
    assert window.status_text == "Nothing to normalize: the audio is silent"
    assert window.session.history.undo_count == 1


def test_insert_silence_at_cursor_and_at_selection(window, original):
    window.prompt_silence_seconds = lambda: 0.5
    window.seek(1.0)
    window.edit_toolbar.insert_silence_action.trigger()
    assert window.clip.frames == FRAMES + 24000
    same(window.clip.samples, ops.insert_silence(original, 48000, 24000))
    assert window.waveform.selection == pytest.approx((1.0, 1.5))  # the inserted silence
    assert window.status_text == "Insert 0.5 s of silence"
    select(window, 0.25, 0.5)
    window.insert_silence(0.25)
    assert window.clip.frames == FRAMES + 36000
    assert not window.clip.samples[12000:24000].any()
    window.prompt_silence_seconds = lambda: None
    window.insert_silence()
    assert window.clip.frames == FRAMES + 36000


# ------------------------------------------------------------------------- undo / redo


def test_undo_redo_in_any_order_restores_bit_exact(window, original):
    window.prompt_gain_db = lambda: 4.5
    states = [original.copy()]
    steps = [
        lambda: (select(window, 0.2, 0.6), window.fade_in()),
        lambda: (select(window, 1.0, 1.5), window.delete_selection()),
        lambda: window.apply_gain(),
        lambda: (select(window, 0.1, 1.9), window.crop_to_selection()),
        lambda: window.insert_silence(0.3),  # selects the inserted silence
        lambda: (window.waveform.clear_selection(), window.normalize()),
        lambda: (select(window, 0.3, 0.4), window.silence_selection()),
    ]
    for step in steps:
        step()
        states.append(window.clip.samples.copy())
    assert window.session.history.undo_count == len(steps)
    position = len(steps)
    for move in "uuuruuurrrrruuuuuuurrrrrrr":
        if move == "u":
            window.undo()
            position -= 1
        else:
            window.redo()
            position += 1
        same(window.clip.samples, states[position])
        assert_consistent(window)
        assert window.waveform.clip.frames == states[position].shape[0]
    assert position == len(steps)
    while window.session.history.can_undo:
        window.undo()
    same(window.clip.samples, original)
    assert window.status_text.startswith("Undid: ")
    assert window.clip_info.text().startswith("Recorded 00:02.5")


def test_undo_restores_selection_and_redo_reapplies(window, original):
    select(window, 0.5, 1.0)
    window.delete_selection()
    window.undo()
    assert window.status_text == "Undid: Delete"
    assert window.waveform.selection == pytest.approx((0.5, 1.0))
    assert window.player.selection == pytest.approx((0.5, 1.0))
    window.redo()
    assert window.status_text == "Redid: Delete"
    assert window.waveform.selection is None
    assert window.clip.frames == FRAMES - 24000


def test_new_edit_after_undo_clears_redo(window):
    select(window, 0.5, 1.0)
    window.silence_selection()
    window.undo()
    window.fade_in()
    assert not window.session.history.can_redo
    assert not enabled(window)["redo"]


def test_memory_cap_applies_to_the_window_history(make_window, qtbot):
    window = make_window(max_undo_bytes=300_000)  # 0.5 s of audio is 192 kB
    record(qtbot, window)
    for start in (0.0, 0.5, 1.0):
        select(window, start, start + 0.5)
        window.silence_selection()
    history = window.session.history
    assert history.undo_count == 1
    assert history.evicted == 2
    assert history.nbytes <= 300_000
    window.undo()
    assert not window.clip.samples[:48000].any()  # the evicted edits stay applied
    assert window.clip_info.text().startswith("Edited")  # still differs from the recording


# ------------------------------------------------------------------------- shortcuts


def test_shortcuts(window, original, qtbot):
    show(qtbot, window)
    select(window, 0.5, 1.0)
    qtbot.keyClick(window, Qt.Key.Key_Delete)
    assert window.clip.frames == FRAMES - 24000
    qtbot.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    same(window.clip.samples, original)
    qtbot.keyClick(window, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
    assert window.clip.frames == FRAMES - 24000
    qtbot.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClick(
        window,
        Qt.Key.Key_Z,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
    )
    assert window.clip.frames == FRAMES - 24000
    assert window.status_text == "Redid: Delete"


def test_delete_key_without_selection_does_nothing(window, qtbot):
    show(qtbot, window)
    qtbot.keyClick(window, Qt.Key.Key_Delete)
    assert window.clip.frames == FRAMES
    qtbot.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert window.clip.frames == FRAMES


# ------------------------------------------------------------------------- while playing


def test_edit_while_playing_pauses_first(window, original, backend, qtbot):
    select(window, 1.0, 2.0)
    qtbot.mouseClick(window.playback.play_button, Qt.MouseButton.LeftButton)
    stream = backend.stream
    stream.pump(4800)
    stream.time += 0.05
    assert enabled(window)["silence"]  # usable while playing
    window.silence_selection()
    assert window.state is AppState.STOPPED
    assert stream.closed
    assert window.player.position == pytest.approx(1.03, abs=1e-3)  # the paused position
    assert not window.clip.samples[48000:96000].any()
    # Resuming plays the edited audio from where it paused.
    out = play_and_pump(qtbot, window, backend, 480)
    assert not out.any()


def test_undo_while_playing(window, original, backend, qtbot):
    select(window, 0.5, 1.0)
    window.silence_selection()
    qtbot.mouseClick(window.playback.play_button, Qt.MouseButton.LeftButton)
    window.undo()
    assert window.state is AppState.STOPPED
    same(window.clip.samples, original)


# ------------------------------------------------------------------------- long clips


@pytest.fixture
def async_window(make_window, qtbot):
    w = make_window(sync_edit_frames=0)  # every edit runs on the loader thread
    record(qtbot, w)
    return w


def hold_loader(window):
    """Block the loader thread until the returned event is set."""
    gate = threading.Event()
    window._loader.submit(lambda: gate.wait(5))
    return gate


def test_long_edit_runs_off_the_gui_thread(async_window, qtbot, monkeypatch):
    window = async_window
    original = window.clip.samples.copy()
    threads = []
    real = commands.Delete.apply

    def spy(self, samples):
        threads.append(threading.current_thread())
        return real(self, samples)

    monkeypatch.setattr(commands.Delete, "apply", spy)
    gate = hold_loader(window)
    select(window, 0.5, 1.0)
    window.delete_selection()
    assert window.state is AppState.EDITING
    assert window.status_text == "Delete…"
    assert not any(enabled(window).values())
    assert not window.playback.play_button.isEnabled()
    assert not window.transport.record_button.isEnabled()
    assert window.transport.state_label.text() == "Editing…"
    assert window.clip.frames == FRAMES  # not committed yet
    window.toggle_playback()  # Space is ignored too
    window.undo()
    assert window.state is AppState.EDITING
    gate.set()
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)
    assert threads and threads[0] is not threading.current_thread()
    assert window.clip.frames == FRAMES - 24000
    assert_consistent(window)
    assert window.status_text == "Delete"
    window.undo()
    assert window.status_text == "Undo Delete…"
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)
    same(window.clip.samples, original)


def test_long_edit_no_change_and_failure(async_window, qtbot, monkeypatch):
    window = async_window
    window.apply_gain(0.0)
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)
    assert window.status_text == "A gain of 0 dB changes nothing"

    def boom(self, samples):
        raise MemoryError

    monkeypatch.setattr(commands.Delete, "apply", boom)
    select(window, 0.5, 1.0)
    window.delete_selection()
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)
    assert window.status_is_error
    assert window.status_text == "Could not delete: not enough memory"
    assert window.clip.frames == FRAMES
    assert not window.session.history.can_undo
    assert enabled(window)["delete"]


def test_sync_edit_failure_shows_error(window, monkeypatch):
    def boom(self, samples):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(commands.Crop, "apply", boom)
    select(window, 0.5, 1.0)
    window.crop_to_selection()
    assert window.status_text == "Could not crop: disk on fire"
    assert window.state is AppState.STOPPED


def test_close_during_a_long_edit(async_window, qtbot):
    window = async_window
    gate = hold_loader(window)
    select(window, 0.5, 1.0)
    window.delete_selection()
    assert window.state is AppState.EDITING
    path = window.result.path
    threading.Timer(0.05, gate.set).start()
    window.close()  # drains the loader; the late result is ignored
    assert not path.exists()
    assert window.state is AppState.EDITING  # never committed after shutdown


def test_threshold_picks_sync_or_async(make_window, qtbot):
    window = make_window(sync_edit_frames=50_000)
    record(qtbot, window)
    select(window, 0.0, 1.0)  # 48000 frames: in place, below the threshold
    window.silence_selection()
    assert window.state is AppState.STOPPED  # done synchronously
    window.delete_selection()  # structural: the whole clip (120000 frames) is rebuilt
    assert window.state is AppState.EDITING
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)
    assert window.clip.frames == 72000


# --------------------------------------------------------------------------- undo on disk


def test_undo_data_spills_to_the_temp_folder(make_window, qtbot, recorder, tmp_path):
    from desktop_audio_recorder.editing import spill

    # 120 000 frames = 960 kB per whole-clip undo copy; let 100 kB stay in memory and anything
    # from 1 byte up spill.
    w = make_window(max_undo_bytes=100_000)
    record(qtbot, w)
    store = w._spill_store()
    store.min_spill_bytes = 1
    store.min_free_bytes = 0
    original = w.clip.samples.copy()
    for db in (1.0, 2.0, 3.0):
        w.apply_gain(db)
        qtbot.waitUntil(lambda: w.state is AppState.STOPPED, timeout=TIMEOUT)
    assert store.wait_idle(5)
    history = w.session.history
    assert history.undo_count == 3 and history.evicted == 0
    assert history.disk_bytes == 2 * original.nbytes
    folder = store.folder
    assert folder.parent == recorder.temp_dir and len(list(folder.glob("*.f32"))) == 2
    # Undoing a spilled entry reads it back (in the background: it counts as a big job).
    for _ in range(3):
        w.undo()
        qtbot.waitUntil(lambda: w.state is AppState.STOPPED, timeout=TIMEOUT)
    np.testing.assert_array_equal(w.clip.samples, original)
    # A new recording drops the old history and its files; closing removes the folder.
    record(qtbot, w)
    assert store.wait_idle(5)
    assert not list(folder.glob("*.f32"))
    w.close()
    assert not folder.exists()
    assert spill.SpillStore.remove_stale(recorder.temp_dir) == 0


def test_stale_undo_folders_are_removed_at_start(make_window, recorder):
    stale = recorder.temp_dir / "undo-4242-0badf00d"
    stale.mkdir(parents=True)
    (stale / "000001.f32").write_bytes(b"\0" * 64)
    w = make_window()
    w.offer_recovery_later(delay_ms=60_000)
    assert not stale.exists()


def test_undo_spill_can_be_turned_off(make_window, qtbot):
    w = make_window(max_undo_bytes=100_000, max_undo_disk_bytes=0)
    record(qtbot, w)
    assert w._spill_store() is None and w.session.history.spill is None
