"""Phase 7 window behaviour: persisted settings, crash recovery and session end.

(``QApplication.quit()`` only acts inside ``app.exec()``, so it is checked by the real-window
verification script, not here: in Qt 6 it sends close events, and a refused close cancels it.)"""

import shutil

import numpy as np
import pytest
import soundfile as sf
from conftest import FakeBackend, FakeDeviceManager, FakeRecorder
from PySide6.QtCore import QSettings, Qt

from desktop_audio_recorder.audio.player import ClipPlayer
from desktop_audio_recorder.audio.recorder import TEMP_PREFIX
from desktop_audio_recorder.processing.denoise import DenoiseSettings, NoiseMode
from desktop_audio_recorder.settings import Settings
from desktop_audio_recorder.ui import main_window as mw
from desktop_audio_recorder.ui.denoise_dialog import DenoiseRequest
from desktop_audio_recorder.ui.export_dialog import RecoveryChoice, UnsavedChoice
from desktop_audio_recorder.ui.main_window import MainWindow
from desktop_audio_recorder.ui.state import AppState

TIMEOUT = 3000
RATE = 48000


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def ini(tmp_path):
    return str(tmp_path / "settings.ini")


@pytest.fixture
def temp(tmp_path):
    folder = tmp_path / "temp"
    folder.mkdir()
    return folder


@pytest.fixture
def make_window(qtbot, ini, temp):
    def make(**kwargs):
        kwargs.setdefault("recorder", FakeRecorder(temp))
        kwargs.setdefault("device_manager", FakeDeviceManager())
        kwargs.setdefault("clock", Clock())
        kwargs.setdefault("player", ClipPlayer(FakeBackend()))
        kwargs.setdefault("settings", Settings(QSettings(ini, QSettings.Format.IniFormat)))
        window = MainWindow(**kwargs)
        qtbot.addWidget(window)
        window.prompts = []
        window.recovery_prompts = []
        window.unsaved_answer = UnsavedChoice.DISCARD
        window.recovery_answer = RecoveryChoice.LATER

        def unsaved(prompt):
            window.prompts.append(prompt)
            return window.unsaved_answer

        def recover(prompt):
            window.recovery_prompts.append(prompt)
            return window.recovery_answer

        window.prompt_unsaved = unsaved
        window.prompt_recovery = recover
        return window

    return make


def record(qtbot, window):
    qtbot.mouseClick(window.transport.record_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.state is AppState.RECORDING, timeout=TIMEOUT)
    qtbot.mouseClick(window.transport.stop_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)


# --------------------------------------------------------------------------- settings


def test_loop_is_remembered(make_window):
    first = make_window()
    assert not first.playback.loop
    first.playback.loop_button.setChecked(True)
    assert first.player.loop
    first.close()
    second = make_window()
    assert second.playback.loop
    assert second.player.loop


def test_denoise_choices_are_remembered(make_window, qtbot):
    first = make_window()
    assert first.denoise_settings == DenoiseSettings()
    record(qtbot, first)
    chosen = DenoiseSettings(NoiseMode.NON_STATIONARY, 0.6, None, normalize=True)
    first.prompt_denoise = lambda: DenoiseRequest(chosen, profile_from_selection=False)
    first.reduce_noise()
    qtbot.waitUntil(lambda: first.state is AppState.STOPPED, timeout=10_000)
    first.close()
    second = make_window()
    assert second.denoise_settings == chosen
    assert second.denoise_profile_from_selection is False


def test_window_geometry_is_remembered(make_window, ini):
    first = make_window()
    first.show()
    first.setGeometry(40, 50, 640, 470)  # the offscreen screen is 800 x 600
    first.close()
    assert Settings(QSettings(ini, QSettings.Format.IniFormat)).window_geometry is not None
    second = make_window()
    assert (second.width(), second.height()) == (640, 470)


def test_geometry_is_saved_once_even_if_shutdown_runs_twice(make_window, ini):
    window = make_window()
    window.show()
    window.resize(700, 500)
    window.close()
    window.resize(900, 900)  # after closing: not saved again
    window.shutdown()
    again = make_window()
    assert (again.width(), again.height()) == (700, 500)


# --------------------------------------------------------------------------- crash recovery


def leave_crashed_recording(folder, name="recording-20261103-101500-x.wav", seconds=1.5):
    frames = round(seconds * RATE)
    data = np.full((frames, 2), 0.25, dtype=np.float32)
    live = folder / "live.tmp"
    f = sf.SoundFile(live, mode="w", samplerate=RATE, channels=2, subtype="FLOAT", format="WAV")
    f.write(data)
    f.flush()
    shutil.copy(live, folder / name)
    f.close()
    live.unlink()
    return folder / name


def test_no_leftovers_no_prompt(make_window):
    window = make_window()
    window.offer_recovery()
    assert window.recovery_prompts == []


def test_recover_a_crashed_recording(make_window, qtbot, temp):
    path = leave_crashed_recording(temp)
    window = make_window()
    window.recovery_answer = RecoveryChoice.RECOVER
    window.offer_recovery()
    (prompt,) = window.recovery_prompts
    assert "00:01.5" in prompt.text and "wasn't saved" in prompt.text
    assert prompt.count == 1
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)
    assert window.clip.frames == round(1.5 * RATE)
    assert np.allclose(window.clip.samples, 0.25)
    assert window.clip_info.text().startswith("Recovered recording 00:01.5")
    assert window.status_text.startswith("Recovered the recording from")
    assert window.has_unsaved_changes
    assert sf.info(path).frames == round(1.5 * RATE)  # the header was repaired
    window.close()  # Discard
    assert not path.exists()


def test_delete_all_leftovers(make_window, temp):
    a = leave_crashed_recording(temp, f"{TEMP_PREFIX}a.wav")
    b = leave_crashed_recording(temp, f"{TEMP_PREFIX}b.wav")
    window = make_window()
    window.recovery_answer = RecoveryChoice.DELETE
    window.offer_recovery()
    assert window.recovery_prompts[0].count == 2
    assert "1 older unsaved recording" in window.recovery_prompts[0].detail
    assert not a.exists() and not b.exists()
    assert window.status_text == "Deleted 2 unsaved recording(s)"
    assert window.state is AppState.IDLE


def test_not_now_keeps_them(make_window, temp):
    path = leave_crashed_recording(temp)
    window = make_window()
    window.offer_recovery()
    assert len(window.recovery_prompts) == 1
    assert path.exists()
    assert window.state is AppState.IDLE
    window.close()
    assert path.exists()


def test_empty_leftovers_are_deleted_without_asking(make_window, temp):
    empty = leave_crashed_recording(temp, seconds=0)
    window = make_window()
    window.offer_recovery()
    assert window.recovery_prompts == []
    assert not empty.exists()


def test_not_offered_with_a_clip_loaded(make_window, qtbot, temp):
    window = make_window()
    record(qtbot, window)
    leave_crashed_recording(temp)
    window.offer_recovery()
    assert window.recovery_prompts == []


def test_a_failed_recovery_keeps_the_file(make_window, qtbot, temp, monkeypatch):
    path = leave_crashed_recording(temp)

    def broken(p):
        raise OSError("unreadable")

    monkeypatch.setattr(mw, "load_clip", broken)
    window = make_window()
    window.recovery_answer = RecoveryChoice.RECOVER
    window.offer_recovery()
    qtbot.waitUntil(
        lambda: window.state is AppState.IDLE and window.status_is_error, timeout=TIMEOUT
    )
    assert "Could not recover" in window.status_text
    assert str(temp) in window.status_text
    window.close()
    assert path.exists()


def test_offer_recovery_later_runs_from_the_event_loop(make_window, qtbot, temp):
    leave_crashed_recording(temp)
    window = make_window()
    window.offer_recovery_later(delay_ms=0)
    assert window.recovery_prompts == []
    qtbot.waitUntil(lambda: len(window.recovery_prompts) == 1, timeout=TIMEOUT)


# --------------------------------------------------------------------------- session end / quit


class FakeSessionManager:
    def __init__(self, allows=True):
        self.allows = allows
        self.cancelled = False
        self.released = False

    def allowsInteraction(self):  # noqa: N802 - Qt's name
        return self.allows

    def cancel(self):
        self.cancelled = True

    def release(self):
        self.released = True


def test_session_end_without_a_clip_asks_nothing(make_window):
    window = make_window()
    manager = FakeSessionManager()
    window.commit_session_data(manager)
    assert window.prompts == []
    assert not manager.cancelled


def test_session_end_discard_lets_windows_sign_out(make_window, qtbot):
    window = make_window()
    record(qtbot, window)
    manager = FakeSessionManager()
    window.commit_session_data(manager)
    assert len(window.prompts) == 1
    assert window.prompts[0].can_save and "signs out" in window.prompts[0].text
    assert manager.released and not manager.cancelled
    window.close()  # Qt closes the windows next: no second prompt
    assert len(window.prompts) == 1


def test_session_end_cancel_cancels_the_sign_out(make_window, qtbot):
    window = make_window()
    record(qtbot, window)
    window.unsaved_answer = UnsavedChoice.CANCEL
    manager = FakeSessionManager()
    window.commit_session_data(manager)
    assert manager.cancelled and manager.released
    assert window.state is AppState.STOPPED  # nothing was closed or discarded


def test_session_end_save_cancels_and_exports(make_window, qtbot):
    window = make_window()
    record(qtbot, window)
    window.unsaved_answer = UnsavedChoice.SAVE
    asked = []
    window.prompt_export = lambda: asked.append(True)  # cancelled export dialog
    manager = FakeSessionManager()
    window.commit_session_data(manager)
    assert manager.cancelled
    assert asked == [True]


def test_session_end_while_recording_offers_close_anyway(make_window, qtbot):
    window = make_window()
    qtbot.mouseClick(window.transport.record_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.state is AppState.RECORDING, timeout=TIMEOUT)
    manager = FakeSessionManager()
    window.commit_session_data(manager)
    (prompt,) = window.prompts
    assert not prompt.can_save and "recording is in progress" in prompt.text
    assert not manager.cancelled


def test_session_end_without_interaction_keeps_the_recording_for_recovery(make_window, qtbot, temp):
    window = make_window()
    record(qtbot, window)
    path = window.result.path
    window.commit_session_data(FakeSessionManager(allows=False))
    assert window.prompts == []
    window.close()
    assert path.exists()  # offered for recovery at the next start
    # FakeRecorder names its files differently; rename like the real recorder would.
    path.rename(temp / f"{TEMP_PREFIX}kept.wav")
    nxt = make_window()
    nxt.offer_recovery()
    assert len(nxt.recovery_prompts) == 1
