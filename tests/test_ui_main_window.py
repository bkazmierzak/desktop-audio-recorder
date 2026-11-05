"""Main window with a fake recorder and device manager (offscreen, real background thread)."""

import threading

import numpy as np
import pytest
from conftest import FakeBackend, FakeDeviceManager, FakeRecorder, make_loopback
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication

from desktop_audio_recorder.audio.devices import DeviceError
from desktop_audio_recorder.audio.player import ClipPlayer
from desktop_audio_recorder.audio.recorder import RecorderError, RecorderState
from desktop_audio_recorder.ui.level_meter import level_to_fraction
from desktop_audio_recorder.ui.main_window import MainWindow, device_summary
from desktop_audio_recorder.ui.state import AppState

TIMEOUT = 3000


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def recorder(tmp_path):
    return FakeRecorder(tmp_path)


@pytest.fixture
def dm():
    return FakeDeviceManager()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def make_window(qtbot, recorder, dm, clock, backend):
    def make(**kwargs):
        kwargs.setdefault("recorder", recorder)
        kwargs.setdefault("device_manager", dm)
        kwargs.setdefault("clock", clock)
        kwargs.setdefault("player", ClipPlayer(backend))
        window = MainWindow(**kwargs)
        qtbot.addWidget(window)
        return window

    return make


@pytest.fixture
def window(make_window, qtbot):
    w = make_window()
    qtbot.waitUntil(lambda: w.device is not None, timeout=TIMEOUT)
    return w


def click(qtbot, button):
    assert button.isEnabled(), f"{button.text()} is disabled"
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)


def wait_state(qtbot, window, state):
    qtbot.waitUntil(lambda: window.state is state, timeout=TIMEOUT)


def buttons(window):
    t = window.transport
    return (
        t.record_button.isEnabled(),
        t.pause_button.isEnabled(),
        t.stop_button.isEnabled(),
        window.refresh_button.isEnabled(),
    )


def start(qtbot, window):
    click(qtbot, window.transport.record_button)
    wait_state(qtbot, window, AppState.RECORDING)


def stop(qtbot, window):
    click(qtbot, window.transport.stop_button)
    wait_state(qtbot, window, AppState.STOPPED)


# ------------------------------------------------------------------------- device line


def test_initial_state_shows_default_device(window, dm):
    assert window.state is AppState.IDLE
    assert dm.calls == 1
    assert window.device_label.text() == "Recording from: Speakers (INZONE H9 II - Game)"
    assert window.device_detail_label.text() == "48 kHz · 8 ch → stereo"
    assert buttons(window) == (True, False, False, True)
    assert window.transport.elapsed_label.text() == "00:00.0"
    assert window.status_text == "Ready"
    assert not window.status_is_error


@pytest.mark.parametrize(
    ("channels", "rate", "text"),
    [(2, 48000, "48 kHz · stereo"), (1, 44100, "44.1 kHz · mono → stereo")],
)
def test_device_summary(channels, rate, text):
    assert device_summary(make_loopback(channels=channels, rate=rate)) == text


def test_refresh_button_picks_up_new_default(window, dm, qtbot):
    dm.device = make_loopback("Headphones (USB Audio)", channels=2)
    click(qtbot, window.refresh_button)
    qtbot.waitUntil(lambda: "Headphones" in window.device_label.text(), timeout=TIMEOUT)
    assert window.device_detail_label.text() == "48 kHz · stereo"
    assert dm.calls == 2


def test_no_device_at_startup_shows_error(make_window, dm, qtbot):
    dm.error = DeviceError("No default playback device is set in Windows.")
    window = make_window()
    qtbot.waitUntil(lambda: window.status_is_error, timeout=TIMEOUT)
    assert window.status_text == "No playback device: No default playback device is set in Windows."
    assert window.device_label.text() == "Recording from: no playback device found"
    assert window.transport.record_button.isEnabled()  # start() re-resolves anyway

    # The device comes back: Refresh clears the error.
    dm.error = None
    click(qtbot, window.refresh_button)
    qtbot.waitUntil(lambda: not window.status_is_error, timeout=TIMEOUT)
    assert window.status_text == "Ready"
    assert "INZONE" in window.device_label.text()


def test_activation_refresh_is_throttled(window, dm, clock, qtbot):
    window.refresh_device()  # same instant as the startup refresh: skipped
    clock.now += 0.5
    window.refresh_device()
    assert dm.calls == 1
    clock.now += 1.0
    window.refresh_device()
    qtbot.waitUntil(lambda: dm.calls == 2, timeout=TIMEOUT)


def test_no_refresh_while_recording(window, dm, clock, qtbot):
    start(qtbot, window)
    clock.now += 10
    window.refresh_device(force=True)
    qtbot.wait(50)
    assert dm.calls == 1
    assert not window.refresh_button.isEnabled()


# ------------------------------------------------------------------------- transport


def test_record_pause_resume_stop_cycle(window, recorder, qtbot):
    start(qtbot, window)
    assert buttons(window) == (False, True, True, False)
    assert window.transport.pause_button.text() == "Pause"
    assert window.status_text == "Recording from Speakers (INZONE H9 II - Game)"
    assert window.windowTitle().startswith("● ")

    click(qtbot, window.transport.pause_button)
    assert window.state is AppState.PAUSED
    assert recorder.state is RecorderState.PAUSED
    assert buttons(window) == (False, True, True, False)
    assert window.transport.pause_button.text() == "Resume"

    click(qtbot, window.transport.pause_button)
    assert window.state is AppState.RECORDING
    assert recorder.state is RecorderState.RECORDING

    stop(qtbot, window)
    assert buttons(window) == (True, False, False, True)
    assert recorder.calls == ["start", "pause", "resume", "stop"]
    assert window.result is not None
    assert window.result.duration == pytest.approx(2.5)
    assert window.transport.elapsed_label.text() == "00:02.5"
    assert window.status_text == "Recorded 00:02.5 from Speakers (INZONE H9 II - Game)"
    assert "not saved" in window.clip_info.text()


def test_stop_from_paused(window, recorder, qtbot):
    start(qtbot, window)
    click(qtbot, window.transport.pause_button)
    stop(qtbot, window)
    assert recorder.calls == ["start", "pause", "stop"]


def test_start_and_stop_run_off_the_gui_thread(window, recorder, qtbot):
    start(qtbot, window)
    click(qtbot, window.transport.pause_button)
    click(qtbot, window.transport.pause_button)
    stop(qtbot, window)
    gui = threading.current_thread().name
    threads = dict(recorder.threads)
    assert threads["start"] != gui
    assert threads["stop"] != gui
    assert threads["start"] == threads["stop"]  # one audio thread for all PortAudio work
    assert threads["pause"] == gui


def test_buttons_disabled_and_gui_responsive_while_starting(window, recorder, qtbot):
    recorder.start_gate = threading.Event()
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(10)

    click(qtbot, window.transport.record_button)
    assert window.state is AppState.STARTING
    assert buttons(window) == (False, False, False, False)
    qtbot.wait(100)  # start() is blocked on the audio thread; the event loop keeps running
    assert len(ticks) >= 3
    assert window.state is AppState.STARTING

    recorder.start_gate.set()
    wait_state(qtbot, window, AppState.RECORDING)
    timer.stop()


def test_buttons_disabled_while_stopping(window, recorder, qtbot):
    start(qtbot, window)
    recorder.stop_gate = threading.Event()
    click(qtbot, window.transport.stop_button)
    assert window.state is AppState.STOPPING
    assert buttons(window) == (False, False, False, False)
    recorder.stop_gate.set()
    wait_state(qtbot, window, AppState.STOPPED)


def test_elapsed_timer_text(window, recorder, clock, qtbot):
    start(qtbot, window)
    assert window.meter_running
    recorder.elapsed = 12.34
    qtbot.waitUntil(lambda: window.transport.elapsed_label.text() == "00:12.3", timeout=TIMEOUT)
    recorder.elapsed = 3725.5
    qtbot.waitUntil(lambda: window.transport.elapsed_label.text() == "1:02:05.5", timeout=TIMEOUT)


def test_meter_follows_recorder_levels(window, recorder, clock, qtbot):
    start(qtbot, window)
    recorder.levels = (0.1, 1.5)
    qtbot.waitUntil(lambda: window.meter.clipped == (False, True), timeout=TIMEOUT)
    left, right = window.meter.fractions
    assert left == pytest.approx(level_to_fraction(0.1))
    assert right == 1.0


def test_meter_decays_and_timer_stops_after_stop(window, recorder, clock, qtbot):
    start(qtbot, window)
    recorder.levels = (0.5, 0.5)
    qtbot.waitUntil(lambda: window.meter.fractions[0] > 0.8, timeout=TIMEOUT)
    stop(qtbot, window)
    clock.now += 60  # next tick: a minute of decay
    qtbot.waitUntil(lambda: not window.meter_running, timeout=TIMEOUT)
    assert window.meter.fractions == (0.0, 0.0)


# ------------------------------------------------------------------------- errors


def test_start_device_error_shown_in_status_bar(window, recorder, qtbot):
    recorder.start_error = DeviceError("No default playback device is set in Windows.")
    click(qtbot, window.transport.record_button)
    qtbot.waitUntil(lambda: window.status_is_error, timeout=TIMEOUT)
    assert window.state is AppState.IDLE
    assert window.status_text == "Cannot record: No default playback device is set in Windows."
    assert window.device_label.text() == "Recording from: no playback device found"
    assert buttons(window) == (True, False, False, True)


def test_start_portaudio_error_shown_in_status_bar(window, recorder, qtbot):
    recorder.start_error = OSError("[Errno -9996] Invalid device")
    click(qtbot, window.transport.record_button)
    qtbot.waitUntil(lambda: window.status_is_error, timeout=TIMEOUT)
    assert window.state is AppState.IDLE
    assert window.status_text == "Could not start recording: [Errno -9996] Invalid device"


def test_device_refresh_keeps_an_unrelated_error(window, dm, qtbot):
    # Activating the window refreshes the device; that must not hide, say, an export error.
    window.show_error("Could not export take.wav: disk full")
    dm.device = make_loopback("Headphones (USB Audio)", channels=2)
    click(qtbot, window.refresh_button)
    qtbot.waitUntil(lambda: "Headphones" in window.device_label.text(), timeout=TIMEOUT)
    assert window.status_is_error
    assert window.status_text == "Could not export take.wav: disk full"


def test_device_refresh_clears_a_cannot_record_error(window, recorder, dm, qtbot):
    recorder.start_error = DeviceError("No default playback device is set in Windows.")
    click(qtbot, window.transport.record_button)
    qtbot.waitUntil(lambda: window.status_is_error, timeout=TIMEOUT)
    click(qtbot, window.refresh_button)
    qtbot.waitUntil(lambda: not window.status_is_error, timeout=TIMEOUT)
    assert window.status_text == "Ready"
    assert "INZONE" in window.device_label.text()


def test_stop_failure_shown_in_status_bar(window, recorder, qtbot):
    start(qtbot, window)
    error = RecorderError("recording failed")
    error.__cause__ = OSError("disk full")
    recorder.stop_error = error
    click(qtbot, window.transport.stop_button)
    wait_state(qtbot, window, AppState.IDLE)
    assert window.status_is_error
    assert window.status_text == "Recording failed: recording failed: disk full"
    assert window.result is None
    assert buttons(window) == (True, False, False, True)


def test_pause_error_shown_and_state_kept(window, recorder, qtbot):
    start(qtbot, window)
    recorder.pause_error = RecorderError("cannot pause while idle")
    click(qtbot, window.transport.pause_button)
    assert window.state is AppState.RECORDING
    assert window.status_is_error
    assert window.status_text == "Could not pause: cannot pause while idle"


def test_error_cleared_by_next_successful_action(window, recorder, qtbot):
    recorder.start_error = OSError("busy")
    click(qtbot, window.transport.record_button)
    qtbot.waitUntil(lambda: window.status_is_error, timeout=TIMEOUT)
    recorder.start_error = None
    start(qtbot, window)
    assert not window.status_is_error


# ------------------------------------------------------------------------- temp files


def test_new_recording_deletes_previous_temp_file(window, recorder, qtbot):
    start(qtbot, window)
    stop(qtbot, window)
    first = window.result.path
    assert first.exists()
    start(qtbot, window)
    assert not first.exists()
    assert window.result is None
    stop(qtbot, window)
    assert window.result.path.exists()
    assert window.result.path != first


def test_failed_start_keeps_previous_clip(window, recorder, qtbot):
    start(qtbot, window)
    stop(qtbot, window)
    first = window.result
    recorder.start_error = DeviceError("No default playback device is set in Windows.")
    click(qtbot, window.transport.record_button)
    qtbot.waitUntil(lambda: window.status_is_error, timeout=TIMEOUT)
    assert window.state is AppState.STOPPED
    assert window.result is first
    assert first.path.exists()


def test_close_while_recording_stops_and_cleans_up(window, recorder, tmp_path, qtbot):
    start(qtbot, window)
    window.close()
    assert recorder.state is RecorderState.IDLE
    assert recorder.calls[-1] == "stop"
    assert list(tmp_path.glob("fake-recording-*.wav")) == []


def test_close_while_stopping_cleans_up(window, recorder, tmp_path, qtbot):
    start(qtbot, window)
    recorder.stop_gate = threading.Event()
    click(qtbot, window.transport.stop_button)
    threading.Timer(0.1, recorder.stop_gate.set).start()
    window.close()
    assert recorder.calls.count("stop") == 1
    assert list(tmp_path.glob("fake-recording-*.wav")) == []


def test_close_with_unsaved_clip_deletes_it(window, recorder, qtbot):
    start(qtbot, window)
    stop(qtbot, window)
    path = window.result.path
    window.close()
    assert not path.exists()


# ------------------------------------------------------------------------- clip & playback


@pytest.fixture
def stopped(window, qtbot):
    """A window with a recorded (2.5 s, 48 kHz) clip loaded."""
    start(qtbot, window)
    stop(qtbot, window)
    return window


def play(qtbot, window):
    click(qtbot, window.playback.play_button)
    assert window.state is AppState.PLAYING


def playback_buttons(window):
    p = window.playback
    return (p.play_button.isEnabled(), p.stop_button.isEnabled(), p.loop_button.isEnabled())


def show(qtbot, window):
    """Show and activate the window (shortcuts only fire in the active window; activation is
    asynchronous)."""
    window.show()
    qtbot.waitExposed(window)
    window.activateWindow()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is window, timeout=TIMEOUT)


def test_waveform_appears_after_stop(stopped):
    window = stopped
    assert window.clip is not None
    assert window.clip.frames == 120000
    assert window.clip.sample_rate == 48000
    assert window.waveform.isVisibleTo(window)
    assert window.waveform.clip is window.clip
    assert window.waveform.view_range[1] == pytest.approx(2.5, abs=1e-6)
    assert window.player.clip is window.clip
    assert playback_buttons(window) == (True, False, True)
    assert window.playback.position_label.text() == "00:00.0 / 00:02.5"
    assert "not saved" in window.clip_info.text()


def test_no_waveform_before_a_recording(window):
    assert window.clip is None
    assert not window.waveform.isVisibleTo(window)
    assert playback_buttons(window) == (False, False, False)


def test_clip_loads_off_the_gui_thread(window, qtbot, monkeypatch):
    import desktop_audio_recorder.ui.main_window as mw

    threads = []
    gate = threading.Event()
    real = mw.load_clip

    def spy(path):
        threads.append(threading.current_thread().name)
        gate.wait(5)
        return real(path)

    monkeypatch.setattr(mw, "load_clip", spy)
    start(qtbot, window)
    click(qtbot, window.transport.stop_button)
    wait_state(qtbot, window, AppState.LOADING)
    assert buttons(window) == (False, False, False, False)
    assert playback_buttons(window) == (False, False, False)
    assert window.clip_info.text() == "Loading the recording…"
    gate.set()
    wait_state(qtbot, window, AppState.STOPPED)
    assert threads and threads[0] != threading.current_thread().name


def test_load_failure_shows_error(window, recorder, qtbot):
    recorder.write_invalid_file = True
    start(qtbot, window)
    click(qtbot, window.transport.stop_button)
    qtbot.waitUntil(lambda: window.status_is_error, timeout=TIMEOUT)
    assert window.state is AppState.IDLE
    assert window.status_text.startswith("Could not load the recording:")
    assert window.clip is None
    assert buttons(window) == (True, False, False, True)
    # The unusable temp file is still cleaned up.
    path = window.result.path
    window.close()
    assert not path.exists()


def test_play_pause_stop_buttons(stopped, backend, qtbot):
    window = stopped
    play(qtbot, window)
    stream = backend.stream
    assert stream.started
    assert stream.samplerate == 48000
    assert playback_buttons(window) == (True, True, True)
    assert window.playback.play_button.text() == "Pause"
    assert buttons(window) == (False, False, False, True)  # no recording while playing
    assert window.status_text == "Playing on Fake Speakers"
    assert window.meter_running

    stream.pump(4800)
    stream.time += 0.05
    click(qtbot, window.playback.play_button)  # pause
    assert window.state is AppState.STOPPED
    assert stream.closed
    assert window.player.position == pytest.approx(0.03, abs=1e-3)
    assert window.playback.position_label.text() == "00:00.0 / 00:02.5"
    assert window.status_text == "Playback paused"

    play(qtbot, window)
    assert len(backend.streams) == 2
    click(qtbot, window.playback.stop_button)
    assert window.state is AppState.STOPPED
    assert window.player.position == 0.0
    assert window.waveform.playhead_position == 0.0
    assert window.status_text == "Playback stopped"
    qtbot.waitUntil(lambda: not window.meter_running, timeout=TIMEOUT)


def test_playhead_follows_heard_position(stopped, backend, qtbot):
    window = stopped
    play(qtbot, window)
    s = backend.stream
    s.pump(48000)  # one second written; heard from time + 0.02
    s.time += 0.02 + 0.625
    qtbot.waitUntil(lambda: abs(window.waveform.playhead_position - 0.625) < 1e-3, timeout=TIMEOUT)
    assert window.playback.position_label.text() == "00:00.6 / 00:02.5"


def test_end_of_clip_returns_to_stopped(stopped, backend, qtbot):
    window = stopped
    window.seek(2.4)
    play(qtbot, window)
    s = backend.stream
    s.pump(9600)  # 0.1 s of clip, then silence
    qtbot.wait(100)
    assert window.state is AppState.PLAYING  # the tail hasn't been heard yet
    s.time += 0.2
    wait_state(qtbot, window, AppState.STOPPED)
    assert window.status_text == "Playback finished"
    assert window.player.position == pytest.approx(2.5)
    assert window.playback.position_label.text() == "00:02.5 / 00:02.5"


def test_lost_device_shows_error(stopped, backend, qtbot):
    window = stopped
    play(qtbot, window)
    backend.stream.pump(480)
    backend.stream.finish()
    wait_state(qtbot, window, AppState.STOPPED)
    assert window.status_is_error
    assert "unexpectedly" in window.status_text


def test_meter_follows_playback(stopped, backend, qtbot, clock):
    window = stopped  # a 440 Hz sine at 0.5
    window.meter.reset()
    play(qtbot, window)
    window._meter_timer.stop()  # drive the ticks by hand
    stream = backend.stream
    window._tick()
    assert window.meter.fractions == (0.0, 0.0)  # nothing heard yet
    stream.pump(4800)
    stream.time += 0.05
    clock.now += 0.03
    window._tick()
    assert window.meter.channels[0].raw_peak == pytest.approx(0.5, abs=0.01)
    assert window.meter.fractions[0] == pytest.approx(level_to_fraction(0.5), abs=0.01)


def test_play_error_shows_status(stopped, backend, qtbot):
    window = stopped
    backend.open_error = OSError("[Errno -9996] Invalid device")
    click(qtbot, window.playback.play_button)
    assert window.state is AppState.STOPPED
    assert window.status_is_error
    assert window.status_text == "Could not open the playback device: [Errno -9996] Invalid device"


def test_seek_from_waveform(stopped):
    window = stopped
    window.waveform.seek_requested.emit(1.0)
    assert window.player.position == pytest.approx(1.0)
    assert window.waveform.playhead_position == pytest.approx(1.0)
    assert window.playback.position_label.text() == "00:01.0 / 00:02.5"


def test_seek_while_playing(stopped, backend, qtbot):
    window = stopped
    play(qtbot, window)
    backend.stream.pump(480)
    window.seek(2.0)
    out = backend.stream.pump(480)
    np.testing.assert_array_equal(out, window.clip.samples[96000:96480])


def test_selection_is_passed_to_player(stopped):
    window = stopped
    window.waveform.set_selection((0.5, 1.0), emit=True)
    assert window.player.selection == pytest.approx((0.5, 1.0))
    assert window.playback.selection_label.text() == "Selection 00:00.5 – 00:01.0 (0.5 s)"
    window.waveform.clear_selection()
    assert window.player.selection is None
    assert window.playback.selection_label.text() == "No selection"


def test_loop_toggle(stopped, qtbot):
    window = stopped
    click(qtbot, window.playback.loop_button)
    assert window.player.loop
    click(qtbot, window.playback.loop_button)
    assert not window.player.loop


def test_space_toggles_playback_even_with_a_button_focused(stopped, backend, qtbot):
    window = stopped
    show(qtbot, window)
    window.transport.record_button.setFocus()
    qtbot.keyClick(window.transport.record_button, Qt.Key.Key_Space)
    assert window.state is AppState.PLAYING  # not a new recording
    qtbot.keyClick(window, Qt.Key.Key_Space)
    assert window.state is AppState.STOPPED
    assert len(backend.streams) == 1


def test_space_does_nothing_while_recording(window, recorder, qtbot):
    show(qtbot, window)
    start(qtbot, window)
    qtbot.keyClick(window, Qt.Key.Key_Space)
    assert window.state is AppState.RECORDING
    assert recorder.calls == ["start"]


def test_home_goes_to_start(stopped, backend, qtbot):
    window = stopped
    show(qtbot, window)
    window.seek(1.5)
    qtbot.keyClick(window, Qt.Key.Key_Home)
    assert window.player.position == 0.0
    assert window.waveform.playhead_position == 0.0
    # While playing, Home jumps back to the start and keeps playing.
    play(qtbot, window)
    backend.stream.pump(4800)
    qtbot.keyClick(window, Qt.Key.Key_Home)
    assert window.state is AppState.PLAYING
    out = backend.stream.pump(10)
    np.testing.assert_array_equal(out, window.clip.samples[:10])


def test_start_recording_while_playing_stops_playback_first(stopped, backend, qtbot):
    window = stopped
    play(qtbot, window)
    assert not window.transport.record_button.isEnabled()
    window.start_recording()  # e.g. a future shortcut; the button itself is disabled
    wait_state(qtbot, window, AppState.RECORDING)
    assert backend.stream.closed
    assert window.clip is None
    assert window.player.clip is None
    assert not window.waveform.isVisibleTo(window)
    assert playback_buttons(window) == (False, False, False)


def test_failed_start_keeps_clip_playable(stopped, recorder, qtbot):
    window = stopped
    clip = window.clip
    recorder.start_error = OSError("busy")
    click(qtbot, window.transport.record_button)
    qtbot.waitUntil(lambda: window.status_is_error, timeout=TIMEOUT)
    assert window.state is AppState.STOPPED
    assert window.clip is clip
    assert window.player.clip is clip
    assert playback_buttons(window) == (True, False, True)


def test_close_while_playing_closes_stream(stopped, backend, qtbot):
    window = stopped
    play(qtbot, window)
    path = window.result.path
    window.close()
    assert backend.stream.closed
    assert not path.exists()
