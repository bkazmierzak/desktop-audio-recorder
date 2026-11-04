"""Device changes while recording (DeviceWatch + FakeEndpointMonitor), recorder health, and the
real Core Audio endpoint monitor (skipped without it)."""

import threading

import pytest
from conftest import FakeBackend, FakeDeviceManager, FakeRecorder, make_loopback
from PySide6.QtCore import Qt

from desktop_audio_recorder.audio import endpoints
from desktop_audio_recorder.audio.endpoints import (
    DEVICE_STATE_ACTIVE,
    DEVICE_STATE_DISABLED,
    DEVICE_STATE_UNPLUGGED,
    EndpointError,
    EndpointEvent,
    EventKind,
)
from desktop_audio_recorder.audio.player import ClipPlayer
from desktop_audio_recorder.ui.device_watch import DeviceWatch
from desktop_audio_recorder.ui.main_window import MainWindow
from desktop_audio_recorder.ui.state import AppState

TIMEOUT = 3000
INZONE, REALTEK = "{inzone}", "{realtek}"


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def recorder(tmp_path):
    return FakeRecorder(tmp_path)


@pytest.fixture
def dm():
    return FakeDeviceManager()


@pytest.fixture
def window(qtbot, recorder, dm):
    w = MainWindow(
        recorder=recorder, device_manager=dm, clock=Clock(), player=ClipPlayer(FakeBackend())
    )
    qtbot.addWidget(w)
    qtbot.waitUntil(lambda: w.device is not None, timeout=TIMEOUT)
    return w


@pytest.fixture
def warn_only(window):
    """Recording > Follow the default playback device turned off."""
    window.follow_action.setChecked(False)
    return window


def monitor(window):
    return window.device_watch._monitor


def record(qtbot, window):
    qtbot.mouseClick(window.transport.record_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.state is AppState.RECORDING, timeout=TIMEOUT)


def stop(qtbot, window):
    qtbot.mouseClick(window.transport.stop_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)


def default_changed(device_id):
    return EndpointEvent(EventKind.DEFAULT_CHANGED, device_id)


def test_the_watch_is_running(window):
    assert window.device_watch.available
    assert monitor(window).started


@pytest.mark.usefixtures("warn_only")
def test_default_change_while_recording_warns_and_keeps_recording(window, qtbot, recorder):
    record(qtbot, window)
    assert window.warning_text == ""
    assert not window.warning_label.isVisibleTo(window)
    monitor(window).default_id = REALTEK
    monitor(window).fire(default_changed(REALTEK))
    qtbot.waitUntil(lambda: window.warning_text != "", timeout=TIMEOUT)
    text = window.warning_text
    assert "Speakers (Realtek(R) Audio)" in text
    assert "continues from Speakers (INZONE H9 II - Game)" in text
    assert window.warning_label.isVisibleTo(window)
    assert window.state is AppState.RECORDING
    assert "stop" not in recorder.calls


@pytest.mark.usefixtures("warn_only")
def test_the_warning_clears_when_the_default_switches_back(window, qtbot):
    record(qtbot, window)
    monitor(window).fire(default_changed(REALTEK))
    qtbot.waitUntil(lambda: window.warning_text != "", timeout=TIMEOUT)
    monitor(window).fire(default_changed(INZONE))
    qtbot.waitUntil(lambda: window.warning_text == "", timeout=TIMEOUT)
    assert not window.warning_label.isVisibleTo(window)


@pytest.mark.usefixtures("warn_only")
def test_events_from_a_system_thread_arrive_on_the_gui_thread(window, qtbot):
    seen = []
    window.device_watch.changed.connect(lambda e: seen.append(threading.current_thread()))
    record(qtbot, window)
    monitor(window).fire(default_changed(REALTEK), thread=True)
    qtbot.waitUntil(lambda: window.warning_text != "", timeout=TIMEOUT)
    assert seen == [threading.main_thread()]


@pytest.mark.usefixtures("warn_only")
def test_no_default_device_at_all(window, qtbot):
    record(qtbot, window)
    monitor(window).fire(default_changed(""))
    qtbot.waitUntil(lambda: window.warning_text != "", timeout=TIMEOUT)
    assert "no default playback device" in window.warning_text


@pytest.mark.usefixtures("warn_only")
def test_stopping_clears_the_warning_and_refreshes_the_device(window, qtbot, dm):
    record(qtbot, window)
    monitor(window).fire(default_changed(REALTEK))
    qtbot.waitUntil(lambda: window.warning_text != "", timeout=TIMEOUT)
    dm.device = make_loopback("Speakers (Realtek(R) Audio)", channels=2)
    stop(qtbot, window)
    assert window.warning_text == ""
    qtbot.waitUntil(
        lambda: window.device_label.text() == "Recording from: Speakers (Realtek(R) Audio)",
        timeout=TIMEOUT,
    )
    assert "INZONE" in window.status_text  # the clip came from the old device


@pytest.mark.usefixtures("warn_only")
def test_default_changed_just_before_the_recording_started(window, qtbot):
    monitor(window).default_id = REALTEK  # the recorder still resolved INZONE
    record(qtbot, window)
    assert "Speakers (Realtek(R) Audio)" in window.warning_text


@pytest.mark.parametrize(
    "event",
    [
        EndpointEvent(EventKind.STATE_CHANGED, INZONE, DEVICE_STATE_UNPLUGGED),
        EndpointEvent(EventKind.STATE_CHANGED, INZONE, DEVICE_STATE_DISABLED),
        EndpointEvent(EventKind.REMOVED, INZONE),
    ],
    ids=["unplugged", "disabled", "removed"],
)
@pytest.mark.parametrize("follow", [True, False], ids=["following", "warn-only"])
def test_the_recorded_device_going_away_stops_and_keeps_the_clip(
    window, qtbot, recorder, event, follow
):
    # Following, it waits for Windows to choose another default first; none comes here.
    window.follow_action.setChecked(follow)
    record(qtbot, window)
    monitor(window).fire(event)
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)
    assert window.clip is not None
    assert window.status_is_error
    assert "Speakers (INZONE H9 II - Game) was disconnected" in window.status_text
    assert "Kept 00:02.5" in window.status_text


def test_other_devices_changing_are_ignored(window, qtbot, recorder):
    record(qtbot, window)
    monitor(window).fire(EndpointEvent(EventKind.STATE_CHANGED, REALTEK, DEVICE_STATE_UNPLUGGED))
    monitor(window).fire(EndpointEvent(EventKind.STATE_CHANGED, INZONE, DEVICE_STATE_ACTIVE))
    monitor(window).fire(EndpointEvent(EventKind.ADDED, "{new}"))
    qtbot.wait(50)
    assert window.state is AppState.RECORDING
    assert window.warning_text == ""


def test_default_change_while_idle_refreshes_the_device_line(window, qtbot, dm):
    calls = dm.calls
    dm.device = make_loopback("Speakers (Realtek(R) Audio)", channels=2)
    monitor(window).fire(default_changed(REALTEK))
    qtbot.waitUntil(
        lambda: window.device_label.text() == "Recording from: Speakers (Realtek(R) Audio)",
        timeout=TIMEOUT,
    )
    assert dm.calls == calls + 1
    assert window.warning_text == ""


def test_a_change_during_a_running_refresh_refreshes_again(window, qtbot, dm):
    gate = threading.Event()
    original = dm.resolve_default_loopback

    def slow(pa=None):
        gate.wait(2)
        return original(pa)

    dm.resolve_default_loopback = slow
    window.refresh_device(force=True)
    calls = dm.calls
    dm.device = make_loopback("Speakers (Realtek(R) Audio)", channels=2)
    monitor(window).fire(default_changed(REALTEK))  # the running refresh may read the old one
    gate.set()
    qtbot.waitUntil(lambda: dm.calls == calls + 2, timeout=TIMEOUT)
    qtbot.waitUntil(lambda: "Realtek" in window.device_label.text(), timeout=TIMEOUT)


def test_writer_failure_stops_the_recording_and_keeps_the_audio(window, qtbot, recorder):
    record(qtbot, window)
    recorder.result_error = "the disk is full"
    recorder.failure = "the disk is full"
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)
    assert recorder.calls.count("stop") == 1
    assert window.clip is not None
    assert window.status_is_error
    assert window.status_text.startswith("Recording stopped early: the disk is full. Kept")


def test_no_audio_arriving_warns_until_it_resumes(window, qtbot, recorder):
    record(qtbot, window)
    recorder.data_gap = 3.2
    qtbot.waitUntil(lambda: "No audio has arrived" in window.warning_text, timeout=TIMEOUT)
    assert "for 3 s" in window.warning_text
    assert window.state is AppState.RECORDING
    recorder.data_gap = 0.0
    qtbot.waitUntil(lambda: window.warning_text == "", timeout=TIMEOUT)


def test_no_stall_warning_without_keepalive(window, qtbot, recorder):
    recorder.keepalive_active = False  # silence then legitimately delivers nothing
    record(qtbot, window)
    recorder.data_gap = 10.0
    qtbot.wait(100)
    assert window.warning_text == ""


def test_without_core_audio_the_app_still_records(qtbot, recorder, dm):
    def broken(callback):
        raise EndpointError("no audio service")

    watch = DeviceWatch(monitor_factory=broken)
    assert not watch.available
    assert watch.default_render() is None and watch.find_render("x") is None
    w = MainWindow(
        recorder=recorder,
        device_manager=dm,
        clock=Clock(),
        player=ClipPlayer(FakeBackend()),
        device_watch=watch,
    )
    qtbot.addWidget(w)
    record(qtbot, w)
    stop(qtbot, w)
    assert w.clip is not None


def test_closing_stops_the_watch(window):
    fake = monitor(window)
    window.close()
    assert fake.closed
    assert not window.device_watch.available


def test_events_after_close_are_dropped(window, qtbot):
    fake = monitor(window)
    received = []
    window.device_watch.changed.connect(received.append)
    window.device_watch.close()
    fake.started = True
    fake.fire(default_changed(REALTEK))
    qtbot.wait(20)
    assert received == []


# --------------------------------------------------------------------------- real Core Audio


@pytest.fixture
def real_monitor():
    events = []
    try:
        m = endpoints.EndpointMonitor(events.append)
        if m.default_render() is None:
            pytest.skip("no default playback device")
    except EndpointError as exc:
        pytest.skip(f"Core Audio unavailable: {exc}")
    m.events = events
    yield m
    m.close()


def test_real_default_matches_portaudio(real_monitor):
    from desktop_audio_recorder.audio.devices import DeviceError, DeviceManager

    try:
        device = DeviceManager().resolve_default_loopback()
    except DeviceError as exc:
        pytest.skip(str(exc))
    default = real_monitor.default_render()
    assert default.active
    assert default.name == device.output_name
    assert real_monitor.find_render(device.output_name) == default
    assert real_monitor.endpoint(default.id) == default
    assert real_monitor.endpoint("{not-a-device}") is None
    assert default in real_monitor.render_endpoints()


def test_real_notification_client_dispatches_through_com(real_monitor):
    """Call the registered COM object through its interface pointer (the vtable Core Audio
    uses); only multimedia render default changes are passed on."""
    real_monitor.start()
    assert real_monitor.watching
    client = real_monitor._client_ptr
    client.OnDefaultDeviceChanged(endpoints.E_RENDER, endpoints.E_CONSOLE, "{console}")
    client.OnDefaultDeviceChanged(endpoints.E_CAPTURE, endpoints.E_MULTIMEDIA, "{mic}")
    client.OnDefaultDeviceChanged(endpoints.E_RENDER, endpoints.E_MULTIMEDIA, "{new}")
    client.OnDeviceStateChanged("{dev}", DEVICE_STATE_UNPLUGGED)
    client.OnDeviceRemoved("{gone}")
    client.OnDeviceAdded("{added}")
    events = [
        e for e in real_monitor.events if e.device_id.startswith("{") and "-" not in e.device_id
    ]
    assert events == [
        EndpointEvent(EventKind.DEFAULT_CHANGED, "{new}"),
        EndpointEvent(EventKind.STATE_CHANGED, "{dev}", DEVICE_STATE_UNPLUGGED),
        EndpointEvent(EventKind.REMOVED, "{gone}"),
        EndpointEvent(EventKind.ADDED, "{added}"),
    ]
    real_monitor.stop()
    assert not real_monitor.watching


def test_describe_state():
    assert endpoints.describe_state(DEVICE_STATE_UNPLUGGED) == "unplugged"
    assert endpoints.describe_state(0x40) == "state 0x40"


# --------------------------------------------------------------------------- following the default


REALTEK_DEVICE = make_loopback("Speakers (Realtek(R) Audio)", channels=2)


def switch_default(window, device_id):
    monitor(window).default_id = device_id
    monitor(window).fire(default_changed(device_id))


def test_following_is_on_by_default_and_remembered(window):
    assert window.follow_action.isChecked() and window.follow_action.isCheckable()
    window.follow_action.setChecked(False)
    assert window.settings.follow_default_device is False
    window.follow_action.setChecked(True)
    assert window.settings.follow_default_device is True


def test_the_recording_moves_to_the_new_default(window, qtbot, recorder):
    recorder.switch_to = REALTEK_DEVICE
    record(qtbot, window)
    recorder.elapsed = 4.2
    switch_default(window, REALTEK)
    qtbot.waitUntil(lambda: window.status_text.startswith("Now recording"), timeout=TIMEOUT)
    assert window.status_text == (
        "Now recording from Speakers (Realtek(R) Audio)"
        " (moved from Speakers (INZONE H9 II - Game) at 00:04.2)"
    )
    (thread,) = [t for name, t in recorder.threads if name == "switch"]
    assert thread == next(t for name, t in recorder.threads if name == "start")  # audio thread
    assert window.warning_text == ""
    assert window.state is AppState.RECORDING
    assert "Speakers (Realtek(R) Audio)" in window.device_label.text()
    # The new device is the one watched now: unplugging the old one changes nothing.
    monitor(window).fire(EndpointEvent(EventKind.STATE_CHANGED, INZONE, DEVICE_STATE_UNPLUGGED))
    qtbot.wait(50)
    assert window.state is AppState.RECORDING and window.warning_text == ""
    stop(qtbot, window)
    assert window.status_text.startswith(
        "Recorded 00:02.5 from Speakers (INZONE H9 II - Game), then Speakers (Realtek(R) Audio)"
    )


def test_a_burst_of_changes_that_ends_where_it_started_switches_nothing(window, qtbot, recorder):
    record(qtbot, window)
    switch_default(window, REALTEK)
    switch_default(window, INZONE)
    qtbot.wait(500)  # longer than SWITCH_DELAY_MS
    assert "switch" not in recorder.calls
    assert window.warning_text == ""


def test_a_burst_of_changes_switches_once(window, qtbot, recorder):
    recorder.switch_to = REALTEK_DEVICE
    record(qtbot, window)
    for target in (REALTEK, INZONE, REALTEK):
        switch_default(window, target)
    qtbot.waitUntil(lambda: window.status_text.startswith("Now recording"), timeout=TIMEOUT)
    qtbot.wait(400)
    assert recorder.calls.count("switch") == 1


def test_a_device_that_cant_be_opened_keeps_the_old_one(window, qtbot, recorder):
    recorder.switch_fallback = "No loopback capture device found"
    record(qtbot, window)
    switch_default(window, REALTEK)
    qtbot.waitUntil(lambda: window.warning_text != "", timeout=TIMEOUT)
    assert window.warning_text.startswith("Couldn't move the recording")
    assert "It continues from Speakers (INZONE H9 II - Game)" in window.warning_text
    assert window.state is AppState.RECORDING
    qtbot.wait(400)
    assert recorder.calls.count("switch") == 1  # no retry loop


def test_no_device_left_stops_and_keeps_the_clip(window, qtbot, recorder):
    from desktop_audio_recorder.audio.recorder import RecorderError

    recorder.switch_error = RecorderError("no device to record from")
    record(qtbot, window)
    switch_default(window, REALTEK)
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)
    assert window.clip is not None and window.status_is_error
    assert window.status_text.startswith(
        "Could not record from the new default playback device (no device to record from),"
        " so the recording stopped. Kept 00:02.5"
    )


def test_unplugging_moves_to_the_new_default(window, qtbot, recorder):
    recorder.switch_to = REALTEK_DEVICE
    record(qtbot, window)
    # Windows announces the lost device first and the new default a moment later.
    monitor(window).set_state(INZONE, DEVICE_STATE_UNPLUGGED)
    monitor(window).fire(EndpointEvent(EventKind.STATE_CHANGED, INZONE, DEVICE_STATE_UNPLUGGED))
    qtbot.waitUntil(lambda: "Moving the recording" in window.warning_text, timeout=TIMEOUT)
    qtbot.wait(200)
    assert "switch" not in recorder.calls and window.state is AppState.RECORDING
    switch_default(window, REALTEK)
    qtbot.waitUntil(lambda: window.status_text.startswith("Now recording"), timeout=TIMEOUT)
    assert window.state is AppState.RECORDING and window.warning_text == ""
    qtbot.wait(1700)  # past LOST_GRACE_MS: nothing stops it
    assert window.state is AppState.RECORDING


def test_unplugging_after_the_default_already_moved(window, qtbot, recorder):
    recorder.switch_to = REALTEK_DEVICE
    record(qtbot, window)
    monitor(window).default_id = REALTEK  # Windows switched first; its event is still to come
    monitor(window).set_state(INZONE, DEVICE_STATE_UNPLUGGED)
    monitor(window).fire(EndpointEvent(EventKind.STATE_CHANGED, INZONE, DEVICE_STATE_UNPLUGGED))
    qtbot.waitUntil(lambda: window.status_text.startswith("Now recording"), timeout=TIMEOUT)
    assert recorder.calls.count("switch") == 1


def test_turning_following_on_mid_recording_switches(warn_only, qtbot, recorder):
    window = warn_only
    recorder.switch_to = REALTEK_DEVICE
    record(qtbot, window)
    switch_default(window, REALTEK)
    qtbot.waitUntil(lambda: window.warning_text != "", timeout=TIMEOUT)
    assert "Follow the default playback device" in window.warning_text
    window.follow_action.setChecked(True)
    qtbot.waitUntil(lambda: window.status_text.startswith("Now recording"), timeout=TIMEOUT)
    assert window.warning_text == ""
