"""File > Open in the main window: loading WAV/MP3/FLAC files, the saved state of an opened clip,
the prompts before replacing an unsaved clip, failures, drops and the command-line argument.

Offscreen, with the fake recorder (a 2.5 s, 48 kHz, 440 Hz sine at 0.5) and the fake player
backend. Real files are written to ``tmp_path``.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from conftest import FakeBackend, FakeDeviceManager, FakeRecorder
from PySide6.QtCore import QMimeData, QSettings, Qt, QUrl

from desktop_audio_recorder import app
from desktop_audio_recorder.audio.player import ClipPlayer
from desktop_audio_recorder.export.writers import ExportFormat, ExportSettings, write_mp3
from desktop_audio_recorder.settings import Settings
from desktop_audio_recorder.ui import main_window
from desktop_audio_recorder.ui.export_dialog import ExportRequest, UnsavedChoice
from desktop_audio_recorder.ui.state import AppState, can_open

TIMEOUT = 5000


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


def tone(seconds, rate, channels=2, hz=1000.0, amplitude=0.25):
    t = np.arange(round(seconds * rate)) / rate
    mono = (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    return np.repeat(mono[:, None], channels, axis=1) if channels > 1 else mono


@pytest.fixture
def recorder(tmp_path):
    return FakeRecorder(tmp_path / "temp")


@pytest.fixture
def ini(tmp_path):
    return str(tmp_path / "settings.ini")


@pytest.fixture
def files(tmp_path):
    folder = tmp_path / "music"
    folder.mkdir()
    return folder


@pytest.fixture
def make_window(qtbot, recorder, ini):
    recorder.temp_dir.mkdir(parents=True, exist_ok=True)

    def make(**kwargs):
        kwargs.setdefault("recorder", recorder)
        kwargs.setdefault("device_manager", FakeDeviceManager())
        kwargs.setdefault("clock", Clock())
        kwargs.setdefault("player", ClipPlayer(FakeBackend()))
        kwargs.setdefault("settings", Settings(QSettings(ini, QSettings.Format.IniFormat)))
        window = main_window.MainWindow(**kwargs)
        window.prompts = []
        window.answers = []

        def prompt(p):
            window.prompts.append(p)
            return window.answers.pop(0) if window.answers else UnsavedChoice.DISCARD

        window.prompt_unsaved = prompt
        qtbot.addWidget(window)
        qtbot.waitUntil(lambda: window.device is not None, timeout=TIMEOUT)
        return window

    return make


@pytest.fixture
def window(make_window):
    return make_window()


def record(qtbot, window):
    qtbot.mouseClick(window.transport.record_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.state is AppState.RECORDING, timeout=TIMEOUT)
    qtbot.mouseClick(window.transport.stop_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)


def open_and_wait(qtbot, window, path=None):
    window.open_file(path)
    qtbot.waitUntil(lambda: window.state is not AppState.LOADING, timeout=TIMEOUT)


def wav(folder, name="talk.wav", seconds=1.5, rate=44100, channels=2, subtype="PCM_16"):
    path = folder / name
    sf.write(str(path), tone(seconds, rate, channels), rate, subtype=subtype)
    return path


# --------------------------------------------------------------------------- opening


def test_open_a_wav(window, qtbot, files, ini):
    path = wav(files, rate=44100)
    open_and_wait(qtbot, window, path)
    assert window.state is AppState.STOPPED
    clip = window.clip
    expected, _ = sf.read(str(path), dtype="float32")
    np.testing.assert_array_equal(clip.samples, expected)
    assert clip.sample_rate == 44100 and clip.source == path
    opened = window.opened_file
    assert (opened.path, opened.channels, opened.format) == (path, 2, "WAV")
    assert window.result is None
    assert not window.has_unsaved_changes  # nothing to save yet
    assert window.clip_info.text() == "Opened talk.wav · 00:01.5 · 44.1 kHz stereo · unchanged"
    assert window.status_text == "Opened talk.wav: 00:01.5 · 44.1 kHz stereo"
    assert window.transport.elapsed_label.text() == "00:01.5"
    assert window.player.clip is clip and window.waveform.isVisible() is not None
    assert Settings(QSettings(ini, QSettings.Format.IniFormat)).open_folder == str(files)


def test_open_mono_flac_and_surround(window, qtbot, files):
    mono = files / "voice.flac"
    sf.write(str(mono), tone(1.0, 22050, channels=1), 22050, format="FLAC")
    open_and_wait(qtbot, window, mono)
    assert window.opened_file.layout == "stereo (from mono)"
    assert window.opened_file.format == "FLAC"
    np.testing.assert_allclose(window.clip.samples[:, 0], window.clip.samples[:, 1])
    assert "22.05 kHz stereo (from mono)" in window.clip_info.text()
    surround = wav(files, "movie.wav", rate=48000, channels=6)
    open_and_wait(qtbot, window, surround)
    assert window.opened_file.layout == "stereo (from 6 ch)"
    assert window.clip.samples.shape == (72000, 2)


def test_open_an_mp3_keeps_its_exact_length(window, qtbot, files):
    path = files / "song.mp3"
    write_mp3(tone(2.0, 44100), 44100, path, 192)
    open_and_wait(qtbot, window, path)
    assert window.state is AppState.STOPPED
    assert window.clip.frames == 88200 and window.clip.sample_rate == 44100
    assert window.opened_file.format == "MP3"


def test_ctrl_o_asks_for_the_file(window, qtbot, files):
    path = wav(files)
    asked = []
    window.prompt_open_path = lambda: asked.append(1) or path
    assert window.open_action.shortcut().toString() == "Ctrl+O"
    window.open_action.trigger()
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)
    assert asked == [1] and window.opened_file.path == path
    window.prompt_open_path = lambda: None  # cancelled: nothing happens
    window.open_action.trigger()
    assert window.state is AppState.STOPPED and window.opened_file.path == path


def test_menu_follows_the_state(window, qtbot):
    assert window.open_action.isEnabled() and not window.export_action.isEnabled()
    qtbot.mouseClick(window.transport.record_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.state is AppState.RECORDING, timeout=TIMEOUT)
    assert not window.open_action.isEnabled()
    window.open_file(Path("whatever.wav"))  # ignored while recording
    assert window.state is AppState.RECORDING
    qtbot.mouseClick(window.transport.stop_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)
    assert window.open_action.isEnabled() and window.export_action.isEnabled()
    assert [can_open(s) for s in (AppState.IDLE, AppState.PLAYING, AppState.EDITING)] == [
        True,
        True,
        False,
    ]


def test_open_while_playing_stops_playback(window, qtbot, files):
    record(qtbot, window)
    window.play()
    assert window.state is AppState.PLAYING
    open_and_wait(qtbot, window, wav(files))
    assert window.state is AppState.STOPPED and window.opened_file is not None


# --------------------------------------------------------------------------- saved state


def test_edits_and_export_of_an_opened_file(window, qtbot, files):
    path = wav(files, rate=48000)
    source = path.read_bytes()
    open_and_wait(qtbot, window, path)
    window.apply_gain(-6.0)
    assert window.has_unsaved_changes
    assert window.clip_info.text() == (
        "Edited: 00:01.5 · 48 kHz stereo · opened talk.wav (00:01.5) · not saved yet"
    )
    # Export suggests a new name next to the file, never the file itself.
    assert window.suggested_export_path() == files / "talk (edited).wav"
    window.settings.export_settings = ExportSettings(ExportFormat.MP3)
    assert window.suggested_export_path() == files / "talk.mp3"
    window.settings.export_settings = ExportSettings()
    target = window.suggested_export_path()
    window.export_clip(ExportRequest(target, ExportSettings()))
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)
    assert target.exists() and path.read_bytes() == source
    assert not window.has_unsaved_changes
    assert window.clip_info.text().endswith("saved as talk (edited).wav")
    window.undo()
    assert window.clip_info.text().startswith("Opened talk.wav")
    assert window.clip_info.text().endswith("changed since saved as talk (edited).wav")


def test_undoing_back_counts_as_unchanged(window, qtbot, files):
    open_and_wait(qtbot, window, wav(files))
    window.apply_gain(3.0)
    assert window.has_unsaved_changes
    window.undo()
    assert not window.has_unsaved_changes
    assert window.clip_info.text().endswith("· unchanged")


def test_opened_files_are_never_deleted(make_window, qtbot, files):
    path = wav(files)
    w = make_window()
    open_and_wait(qtbot, w, path)
    record(qtbot, w)  # unchanged: no prompt, and the file stays
    assert w.prompts == [] and path.exists()
    assert w.opened_file is None and w.result is not None
    open_and_wait(qtbot, w, path)
    w.close()
    assert path.exists()


# --------------------------------------------------------------------------- replacing a clip


def test_opening_over_an_unsaved_recording_asks(window, qtbot, files, recorder):
    path = wav(files)
    record(qtbot, window)
    temp = window.result.path
    original = window.clip
    window.answers = [UnsavedChoice.CANCEL]
    window.open_file(path)
    assert window.state is AppState.STOPPED and window.clip is original
    assert window.prompts[-1].text == "Save the current clip before opening talk.wav?"
    assert window.prompts[-1].can_save and not window.prompts[-1].closing
    window.answers = [UnsavedChoice.DISCARD]
    open_and_wait(qtbot, window, path)
    assert window.opened_file.path == path
    assert not temp.exists()  # the replaced recording's temp file went


def test_save_first_then_open(window, qtbot, files, tmp_path):
    path = wav(files)
    record(qtbot, window)
    target = tmp_path / "kept.wav"
    window.prompt_export = lambda: ExportRequest(target, ExportSettings())
    window.answers = [UnsavedChoice.SAVE]
    window.open_file(path)
    qtbot.waitUntil(lambda: window.opened_file is not None, timeout=TIMEOUT)
    assert window.state is AppState.STOPPED
    assert target.exists() and window.opened_file.path == path


# --------------------------------------------------------------------------- failures


def test_not_an_audio_file_keeps_the_previous_clip(window, qtbot, files):
    record(qtbot, window)
    clip, temp = window.clip, window.result.path
    bad = files / "notes.wav"
    bad.write_text("not audio")
    window.answers = [UnsavedChoice.DISCARD]
    open_and_wait(qtbot, window, bad)
    assert window.state is AppState.STOPPED
    assert window.clip is clip and window.result is not None and temp.exists()
    assert window.status_is_error
    assert window.status_text == (
        "Could not open notes.wav: it isn't an audio file this app can read, or it is damaged"
    )
    assert window.clip_info.text().startswith("Recorded ")


def test_missing_and_empty_files_from_idle(window, qtbot, files):
    open_and_wait(qtbot, window, files / "gone.wav")
    assert window.state is AppState.IDLE
    assert window.status_text == "Could not open gone.wav: the file doesn't exist"
    empty = files / "empty.wav"
    sf.write(str(empty), np.zeros((0, 2), np.float32), 48000)
    open_and_wait(qtbot, window, empty)
    assert window.state is AppState.IDLE and window.clip is None
    assert window.status_text == "Could not open empty.wav: the file has no audio"


# --------------------------------------------------------------------------- drops, command line


class FakeDrop:
    def __init__(self, *paths, remote=False):
        self._mime = QMimeData()
        urls = [QUrl(p) if remote else QUrl.fromLocalFile(str(p)) for p in paths]
        self._mime.setUrls(urls)

    def mimeData(self):  # noqa: N802 (Qt's name)
        return self._mime


def test_dropped_files(tmp_path):
    dropped = main_window.MainWindow._dropped_file
    assert dropped(FakeDrop(tmp_path / "a.MP3")) == tmp_path / "a.MP3"
    assert dropped(FakeDrop(tmp_path / "a.flac")) == tmp_path / "a.flac"
    assert dropped(FakeDrop(tmp_path / "a.txt")) is None
    assert dropped(FakeDrop(tmp_path / "a.wav", tmp_path / "b.wav")) is None  # one at a time
    assert dropped(FakeDrop("https://example.com/a.wav", remote=True)) is None


def test_command_line_file_argument(tmp_path):
    args = app._parse_args([str(tmp_path / "x.wav")])
    assert args.file == tmp_path / "x.wav" and args.smoke_test is None
    assert app._parse_args([]).file is None
    args = app._parse_args(["--smoke-test", str(tmp_path)])
    assert args.smoke_test == tmp_path and args.file is None


def test_open_folder_setting_falls_back(ini, tmp_path):
    s = Settings(QSettings(ini, QSettings.Format.IniFormat))
    assert s.open_folder == s.export_folder
    s.open_folder = tmp_path
    assert s.open_folder == str(tmp_path)
    s.open_folder = tmp_path / "gone"
    assert s.open_folder == s.export_folder
