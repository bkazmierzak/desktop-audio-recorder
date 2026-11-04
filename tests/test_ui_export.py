"""Export in the main window: ExportDialog, the EXPORTING state, saved/unsaved tracking and the
Save / Discard / Cancel prompts before a new recording and on close.

Offscreen, with the fake recorder (a 2.5 s, 48 kHz, 440 Hz sine at 0.5) and the fake player
backend. Real files are written to ``tmp_path``.
"""

import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from conftest import FakeBackend, FakeDeviceManager, FakeRecorder
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from desktop_audio_recorder.audio.player import ClipPlayer
from desktop_audio_recorder.export import writers
from desktop_audio_recorder.export.writers import ExportFormat, ExportSettings
from desktop_audio_recorder.settings import Settings
from desktop_audio_recorder.ui import main_window
from desktop_audio_recorder.ui.export_dialog import (
    ExportDialog,
    ExportRequest,
    UnsavedChoice,
    UnsavedPrompt,
    ask_unsaved,
    format_size,
    with_extension,
)
from desktop_audio_recorder.ui.state import AppState

TIMEOUT = 5000
RATE = 48000
WAV16 = ExportSettings(ExportFormat.WAV, "PCM_16")
MP3 = ExportSettings(ExportFormat.MP3, mp3_bitrate=192)


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


# --------------------------------------------------------------------------- dialog


@pytest.fixture
def make_dialog(qtbot, tmp_path):
    def make(settings=WAV16, path=None, frames=RATE * 3, rate=RATE, has_overs=False):
        dialog = ExportDialog(
            frames=frames,
            rate=rate,
            settings=settings,
            path=path or tmp_path / "Recording.wav",
            has_overs=has_overs,
        )
        qtbot.addWidget(dialog)
        return dialog

    return make


def test_dialog_starts_from_the_given_settings(make_dialog, tmp_path):
    d = make_dialog(ExportSettings(ExportFormat.MP3, "PCM_24", 320), tmp_path / "x.wav")
    assert d.mp3_radio.isChecked()
    assert d.settings() == ExportSettings(ExportFormat.MP3, "PCM_24", 320)
    assert d.path() == tmp_path / "x.mp3"  # the extension follows the format
    assert d.mp3_combo.isVisibleTo(d) and not d.wav_combo.isVisibleTo(d)


def test_dither_checkbox_only_for_16_bit(make_dialog):
    d = make_dialog()
    assert d.dither_check.isChecked() and d.dither_check.isVisibleTo(d)
    d.wav_combo.setCurrentIndex(1)  # 24-bit
    assert not d.dither_check.isVisibleTo(d)
    assert d.settings().dither and not d.settings().dithers  # kept, but unused
    d.mp3_radio.setChecked(True)
    assert d.dither_check.isVisibleTo(d)
    d.dither_check.setChecked(False)
    assert d.settings() == ExportSettings(ExportFormat.MP3, "PCM_24", 192, dither=False)
    off = make_dialog(ExportSettings(dither=False))
    assert not off.dither_check.isChecked()


def test_switching_format_changes_the_extension(make_dialog, tmp_path):
    d = make_dialog()
    assert d.path_edit.text() == str(tmp_path / "Recording.wav")
    d.mp3_radio.setChecked(True)
    assert d.path_edit.text() == str(tmp_path / "Recording.mp3")
    assert d.wav_combo.isHidden() and not d.mp3_combo.isHidden()
    d.wav_radio.setChecked(True)
    assert d.path_edit.text() == str(tmp_path / "Recording.wav")


def test_typing_an_extension_picks_the_format(make_dialog, tmp_path):
    d = make_dialog()
    d.path_edit.setText(str(tmp_path / "song.MP3"))
    assert d.export_format is ExportFormat.MP3
    d.path_edit.setText(str(tmp_path / "song"))  # no extension: keep the format, append it
    assert d.export_format is ExportFormat.MP3
    assert d.path() == tmp_path / "song.mp3"
    d.path_edit.setText(str(tmp_path / "song.flac"))
    assert d.path() == tmp_path / "song.flac.mp3"


def test_summary_and_size(make_dialog):
    d = make_dialog(frames=RATE * 60)
    assert "01:00.0 · 48 kHz stereo · about 11.5 MB" in d.summary_label.text()
    d.mp3_radio.setChecked(True)
    d.mp3_combo.setCurrentIndex(writers.MP3_BITRATES.index(128))
    assert "about 961 KB" in d.summary_label.text()


def test_summary_mentions_resampling_for_mp3(make_dialog):
    d = make_dialog(settings=MP3, frames=96000 * 2, rate=96000)
    assert "48 kHz stereo (resampled from 96 kHz)" in d.summary_label.text()
    d.wav_radio.setChecked(True)
    assert "96 kHz stereo ·" in d.summary_label.text()


def test_overs_warning_only_for_clipping_formats(make_dialog):
    d = make_dialog(has_overs=True)
    assert not d.warning_label.isHidden()
    assert "clipped" in d.warning_label.text()
    d.wav_combo.setCurrentIndex(writers.WAV_SUBTYPES.index("FLOAT"))
    assert d.warning_label.isHidden()
    d.mp3_radio.setChecked(True)
    assert "clipped" in d.warning_label.text()
    assert make_dialog(has_overs=False).warning_label.isHidden()


def test_missing_folder_disables_export(make_dialog, tmp_path):
    d = make_dialog()
    d.path_edit.setText(str(tmp_path / "nope" / "a.wav"))
    assert not d.export_button.isEnabled()
    assert "doesn't exist" in d.warning_label.text()
    d.path_edit.setText("")
    assert not d.export_button.isEnabled()
    assert d.request() is None
    d.path_edit.setText(str(tmp_path / "a.wav"))
    assert d.export_button.isEnabled()


def test_browse_uses_the_save_as_dialog(make_dialog, tmp_path):
    d = make_dialog()
    asked = []

    def save_as(path, fmt):
        asked.append((path, fmt))
        return str(tmp_path / "picked"), ExportFormat.MP3  # MP3 filter, no extension typed

    d.get_save_path = save_as
    d.browse()
    assert asked == [(tmp_path / "Recording.wav", ExportFormat.WAV)]
    assert d.export_format is ExportFormat.MP3
    assert d.path() == tmp_path / "picked.mp3"
    # The Save As dialog already confirmed replacing it.
    (tmp_path / "picked.mp3").write_bytes(b"x")
    d.confirm_overwrite = lambda p: pytest.fail("asked twice")
    d.accept()
    assert d.result() == ExportDialog.DialogCode.Accepted
    assert d.request() == ExportRequest(tmp_path / "picked.mp3", MP3)


def test_browse_extension_beats_the_filter(make_dialog, tmp_path):
    d = make_dialog()
    d.get_save_path = lambda p, f: (str(tmp_path / "a.wav"), ExportFormat.MP3)
    d.browse()
    assert d.export_format is ExportFormat.WAV
    d.get_save_path = lambda p, f: None  # cancelled: nothing changes
    d.browse()
    assert d.path() == tmp_path / "a.wav"


def test_existing_file_is_confirmed(make_dialog, tmp_path):
    target = tmp_path / "Recording.wav"
    target.write_bytes(b"x")
    d = make_dialog()
    answers = []
    d.confirm_overwrite = lambda p: answers.append(p) or False
    d.accept()
    assert answers == [target]
    assert d.result() != ExportDialog.DialogCode.Accepted
    d.confirm_overwrite = lambda p: True
    d.accept()
    assert d.result() == ExportDialog.DialogCode.Accepted


def test_helpers():
    assert format_size(850_000) == "850 KB"
    assert format_size(2_345_678) == "2.3 MB"
    assert format_size(1_250_000_000) == "1.25 GB"
    assert format_size(10) == "1 KB"
    assert with_extension("a/b.wav", ExportFormat.MP3) == Path("a/b.mp3")
    assert with_extension("a/b.c", ExportFormat.WAV) == Path("a/b.c.wav")


@pytest.mark.parametrize(
    ("button", "expected"),
    [
        (QMessageBox.StandardButton.Save, UnsavedChoice.SAVE),
        (QMessageBox.StandardButton.Discard, UnsavedChoice.DISCARD),
        (QMessageBox.StandardButton.Cancel, UnsavedChoice.CANCEL),
    ],
)
def test_ask_unsaved_buttons(qtbot, button, expected):
    seen = {}

    def click():
        box = QApplication.activeModalWidget()
        seen["buttons"] = {b for b in QMessageBox.StandardButton if box.button(b) is not None}
        box.button(button).click()

    QTimer.singleShot(0, click)
    prompt = UnsavedPrompt(closing=True, can_save=True, text="Save?", detail="Lost otherwise.")
    assert ask_unsaved(None, prompt) is expected
    assert seen["buttons"] == {
        QMessageBox.StandardButton.Save,
        QMessageBox.StandardButton.Discard,
        QMessageBox.StandardButton.Cancel,
    }


def test_ask_unsaved_without_save(qtbot):
    seen = {}

    def click():
        box = QApplication.activeModalWidget()
        seen["save"] = box.button(QMessageBox.StandardButton.Save)
        box.button(QMessageBox.StandardButton.Cancel).click()

    QTimer.singleShot(0, click)
    prompt = UnsavedPrompt(closing=True, can_save=False, text="Busy", detail="Close anyway?")
    assert ask_unsaved(None, prompt) is UnsavedChoice.CANCEL
    assert seen["save"] is None


# --------------------------------------------------------------------------- window


@pytest.fixture
def recorder(tmp_path):
    return FakeRecorder(tmp_path)


@pytest.fixture
def ini(tmp_path):
    return str(tmp_path / "settings.ini")


@pytest.fixture
def out(tmp_path):
    folder = tmp_path / "exports"
    folder.mkdir()
    return folder


@pytest.fixture
def make_window(qtbot, recorder, ini):
    def make(**kwargs):
        kwargs.setdefault("recorder", recorder)
        kwargs.setdefault("device_manager", FakeDeviceManager())
        kwargs.setdefault("clock", Clock())
        kwargs.setdefault("player", ClipPlayer(FakeBackend()))
        kwargs.setdefault("settings", Settings(QSettings(ini, QSettings.Format.IniFormat)))
        window = main_window.MainWindow(**kwargs)
        window.prompts = []

        def prompt(p):
            window.prompts.append(p)
            return window.answers.pop(0) if window.answers else UnsavedChoice.DISCARD

        window.answers = []
        window.prompt_unsaved = prompt
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


def export_and_wait(qtbot, window, request=None, then=None):
    window.export_clip(request, then=then)
    qtbot.waitUntil(lambda: window.state is not AppState.EXPORTING, timeout=TIMEOUT)


def test_export_button_follows_the_clip(make_window, qtbot):
    w = make_window()
    assert w.export_button.isHidden()
    record(qtbot, w)
    assert not w.export_button.isHidden()
    assert w.export_button.isEnabled()
    assert w.has_unsaved_changes
    assert "not saved yet" in w.clip_info.text()


@pytest.mark.parametrize(
    ("settings", "subtype", "tolerance"),
    [
        (WAV16, "PCM_16", 1.5 / 32768),  # dithered (the default): TPDF + rounding
        (ExportSettings(ExportFormat.WAV, "PCM_16", dither=False), "PCM_16", 0.5 / 32768),
        (ExportSettings(ExportFormat.WAV, "FLOAT"), "FLOAT", 0.0),
    ],
)
def test_export_wav_reads_back(window, qtbot, out, settings, subtype, tolerance):
    path = out / "clip.wav"
    clip = window.clip
    export_and_wait(qtbot, window, ExportRequest(path, settings))
    assert window.state is AppState.STOPPED
    back, rate = sf.read(str(path), dtype="float32")
    assert sf.info(str(path)).subtype == subtype
    assert rate == clip.sample_rate
    assert back.shape == clip.samples.shape
    assert np.max(np.abs(back - clip.samples)) <= tolerance + 1e-9
    assert window.status_text.startswith("Exported clip.wav (")
    assert "00:02.5" in window.status_text
    assert not window.status_is_error
    assert not window.has_unsaved_changes
    assert window.saved_path == path
    assert "saved as clip.wav" in window.clip_info.text()
    assert list(out.iterdir()) == [path]  # no .part left


def test_export_mp3(window, qtbot, out):
    path = out / "clip.mp3"
    export_and_wait(qtbot, window, ExportRequest(path, MP3))
    back, rate = sf.read(str(path), dtype="float32")
    assert (rate, len(back)) == (RATE, window.clip.frames)
    assert "(MP3, 192 kbps)" in window.status_text


def test_progress_and_cancel(window, qtbot, out, monkeypatch):
    started = threading.Event()

    def slow_export(samples, rate, path, settings, progress, cancelled):
        progress(0.4)
        started.set()
        while not cancelled():
            threading.Event().wait(0.005)
        raise writers.Cancelled("Export cancelled")

    monkeypatch.setattr(main_window, "export", slow_export)
    window.export_clip(ExportRequest(out / "a.wav", WAV16))
    assert window.state is AppState.EXPORTING
    assert window.transport.state_label.text() == "Exporting…"
    assert window.progress_bar.isVisibleTo(window)
    assert window.cancel_button.isEnabled()
    assert "export" in window.cancel_button.toolTip()
    assert not window.export_button.isEnabled()
    assert not window.edit_toolbar.gain_action.isEnabled()
    assert not window.transport.record_button.isEnabled()
    started.wait(2)
    qtbot.waitUntil(lambda: window.progress_bar.value() == 40, timeout=TIMEOUT)
    window.cancel_processing()
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)
    assert window.status_text == "Export cancelled; no file was written"
    assert window.has_unsaved_changes
    assert window.saved_path is None
    assert not window.progress_bar.isVisibleTo(window)


def test_real_cancel_leaves_no_file(window, qtbot, out, monkeypatch):
    real = writers.export

    def cancel_at_once(samples, rate, path, settings, progress, cancelled):
        return real(samples, rate, path, settings, progress, lambda: True)

    monkeypatch.setattr(main_window, "export", cancel_at_once)
    export_and_wait(qtbot, window, ExportRequest(out / "a.mp3", MP3))
    assert list(out.iterdir()) == []
    assert "cancelled" in window.status_text


def test_failed_export_shows_an_error(window, qtbot, tmp_path):
    export_and_wait(qtbot, window, ExportRequest(tmp_path / "missing" / "a.mp3", MP3))
    assert window.state is AppState.STOPPED
    assert window.status_is_error
    assert window.status_text.startswith("Could not export a.mp3: ")
    assert window.has_unsaved_changes


def test_file_in_use_message(window, qtbot, out, monkeypatch):
    def denied(*args):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(main_window, "export", denied)
    export_and_wait(qtbot, window, ExportRequest(out / "a.wav", WAV16))
    assert window.status_text == (
        "Could not export a.wav: access denied (is the file open in another program?)"
    )


def test_overs_are_reported(make_window, recorder, qtbot, out):
    t = np.arange(RATE) / RATE
    x = (1.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    recorder.samples = np.stack([x, x], axis=1)
    w = make_window()
    record(qtbot, w)
    export_and_wait(qtbot, w, ExportRequest(out / "a.wav", WAV16))
    clipped = int(np.count_nonzero(np.abs(recorder.samples) > 1.0))
    assert f"; {clipped:,} samples above 0 dBFS were clipped" in w.status_text


def test_edits_after_export_are_unsaved_until_undone(window, qtbot, out):
    export_and_wait(qtbot, window, ExportRequest(out / "a.wav", WAV16))
    assert not window.has_unsaved_changes
    window.apply_gain(-3.0)
    assert window.has_unsaved_changes
    assert "changed since saved as a.wav" in window.clip_info.text()
    window.undo()
    assert not window.has_unsaved_changes
    assert "saved as a.wav" in window.clip_info.text()
    window.redo()
    assert window.has_unsaved_changes
    export_and_wait(qtbot, window, ExportRequest(out / "b.mp3", MP3))
    assert not window.has_unsaved_changes
    window.undo()  # back to what a.wav holds, but the clip was last saved as b.mp3
    assert window.has_unsaved_changes


def test_export_pauses_playback(window, qtbot, out):
    window.play()
    assert window.state is AppState.PLAYING
    export_and_wait(qtbot, window, ExportRequest(out / "a.wav", WAV16))
    assert (out / "a.wav").exists()
    assert window.player.state.value == "stopped"


def test_dialog_choices_are_remembered(make_window, qtbot, out, ini):
    w = make_window()
    record(qtbot, w)
    asked = []

    def prompt():
        asked.append(w.suggested_export_path())
        return ExportRequest(out / "take.mp3", ExportSettings(ExportFormat.MP3, mp3_bitrate=320))

    w.prompt_export = prompt
    export_and_wait(qtbot, w)
    assert asked[0].name.startswith("Recording ") and asked[0].suffix == ".wav"
    assert (out / "take.mp3").exists()
    # The same clip again: suggest its last file.
    assert w.suggested_export_path() == out / "take.mp3"
    # A new window (next run) starts from the remembered folder and format.
    w2 = make_window(settings=Settings(QSettings(ini, QSettings.Format.IniFormat)))
    record(qtbot, w2)
    assert w2.settings.export_settings == ExportSettings(ExportFormat.MP3, mp3_bitrate=320)
    suggested = w2.suggested_export_path()
    assert suggested.parent == out
    assert suggested.suffix == ".mp3"
    assert suggested.stem.startswith("Recording 20")


def test_cancelled_dialog_exports_nothing(window, qtbot):
    window.prompt_export = lambda: None
    window.export_clip()
    assert window.state is AppState.STOPPED
    assert window.has_unsaved_changes


def test_ctrl_s_opens_the_export_dialog(window, qtbot):
    asked = []
    window.prompt_export = lambda: asked.append(1)
    window.show()
    qtbot.waitExposed(window)
    window.activateWindow()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is window, timeout=TIMEOUT)
    qtbot.keyClick(window, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
    assert asked == [1]
    asked.clear()
    qtbot.mouseClick(window.export_button, Qt.MouseButton.LeftButton)
    assert asked == [1]


# --------------------------------------------------------------------------- prompts


def test_new_recording_cancel_keeps_the_clip(window, recorder, qtbot):
    path = window.result.path
    window.answers = [UnsavedChoice.CANCEL]
    window.start_recording()
    assert window.state is AppState.STOPPED
    assert recorder.calls.count("start") == 1
    assert path.exists()
    (prompt,) = window.prompts
    assert not prompt.closing and prompt.can_save


def test_new_recording_discard_replaces_the_clip(window, recorder, qtbot):
    path = window.result.path
    window.answers = [UnsavedChoice.DISCARD]
    window.start_recording()
    qtbot.waitUntil(lambda: window.state is AppState.RECORDING, timeout=TIMEOUT)
    assert not path.exists()
    assert window.clip is None


def test_new_recording_save_exports_then_records(window, recorder, qtbot, out):
    old = window.result.path
    window.answers = [UnsavedChoice.SAVE]
    window.prompt_export = lambda: ExportRequest(out / "first.wav", WAV16)
    window.start_recording()
    qtbot.waitUntil(lambda: window.state is AppState.RECORDING, timeout=TIMEOUT)
    assert (out / "first.wav").exists()
    assert sf.info(str(out / "first.wav")).frames == 120000
    assert not old.exists()  # the temp WAV goes once the new recording has started
    assert len(window.prompts) == 1


def test_new_recording_save_but_dialog_cancelled(window, recorder, qtbot):
    window.answers = [UnsavedChoice.SAVE]
    window.prompt_export = lambda: None
    window.start_recording()
    assert window.state is AppState.STOPPED
    assert recorder.calls.count("start") == 1


def test_new_recording_save_fails_then_no_recording(window, recorder, qtbot, tmp_path):
    window.answers = [UnsavedChoice.SAVE]
    window.prompt_export = lambda: ExportRequest(tmp_path / "missing" / "a.wav", WAV16)
    window.start_recording()
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)
    qtbot.wait(50)
    assert window.state is AppState.STOPPED
    assert window.status_is_error
    assert recorder.calls.count("start") == 1


def test_no_prompt_once_saved(window, recorder, qtbot, out):
    export_and_wait(qtbot, window, ExportRequest(out / "a.wav", WAV16))
    window.start_recording()
    qtbot.waitUntil(lambda: window.state is AppState.RECORDING, timeout=TIMEOUT)
    assert window.prompts == []
    assert (out / "a.wav").exists()  # the export stays; only the temp WAV goes


def test_close_cancel_keeps_the_window(window, qtbot):
    path = window.result.path
    window.answers = [UnsavedChoice.CANCEL]
    assert window.close() is False
    assert path.exists()
    assert window.state is AppState.STOPPED
    (prompt,) = window.prompts
    assert prompt.closing and prompt.can_save
    window.answers = [UnsavedChoice.DISCARD]
    assert window.close() is True
    assert not path.exists()


def test_close_save_exports_then_closes(window, qtbot, out):
    path = window.result.path
    window.answers = [UnsavedChoice.SAVE]
    window.prompt_export = lambda: ExportRequest(out / "last.mp3", MP3)
    window.show()
    qtbot.waitExposed(window)
    assert window.close() is False  # not yet: the export runs first
    assert window.state is AppState.EXPORTING
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=TIMEOUT)
    assert (out / "last.mp3").exists()
    assert not path.exists()
    assert len(window.prompts) == 1


def test_close_save_dialog_cancelled_keeps_the_window(window, qtbot):
    window.answers = [UnsavedChoice.SAVE]
    window.prompt_export = lambda: None
    assert window.close() is False
    assert window.state is AppState.STOPPED


def test_close_without_changes_does_not_ask(make_window, qtbot, out):
    w = make_window()
    assert w.close() is True  # nothing recorded
    w = make_window()
    record(qtbot, w)
    export_and_wait(qtbot, w, ExportRequest(out / "a.wav", WAV16))
    assert w.close() is True
    assert w.prompts == []


def test_close_while_recording_asks_without_save(make_window, recorder, qtbot):
    w = make_window()
    qtbot.mouseClick(w.transport.record_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: w.state is AppState.RECORDING, timeout=TIMEOUT)
    w.answers = [UnsavedChoice.CANCEL]
    assert w.close() is False
    assert w.state is AppState.RECORDING
    (prompt,) = w.prompts
    assert prompt == UnsavedPrompt(
        closing=True,
        can_save=False,
        text="A recording is in progress.",
        detail="Close anyway? Unsaved audio will be lost.",
    )
    w.answers = [UnsavedChoice.DISCARD]
    assert w.close() is True
    assert recorder.calls[-1] == "stop"


def test_close_during_export_cancels_it(window, qtbot, out, monkeypatch):
    def slow_export(samples, rate, path, settings, progress, cancelled):
        while not cancelled():
            threading.Event().wait(0.005)
        raise writers.Cancelled("Export cancelled")

    monkeypatch.setattr(main_window, "export", slow_export)
    window.export_clip(ExportRequest(out / "a.wav", WAV16))
    job = window.job
    window.answers = [UnsavedChoice.CANCEL]
    assert window.close() is False
    assert window.prompts[-1].text == "The clip is being exported."
    assert not job.is_cancelled()
    window.answers = [UnsavedChoice.DISCARD]
    assert window.close() is True  # shutdown cancels the job and waits for the loader
    assert job.is_cancelled()
    assert list(out.iterdir()) == []
