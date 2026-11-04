import os
from types import SimpleNamespace

import pytest

# Let Qt widgets be created in tests without showing windows.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "hardware: uses the real default playback device (plays a quiet test tone)"
    )


# --------------------------------------------------------------------------- window safety


@pytest.fixture(autouse=True)
def _no_real_settings_or_prompts(monkeypatch, tmp_path_factory):
    """Keep tests away from the user's settings (the registry) and from real modal dialogs.

    ``MainWindow`` without an explicit ``settings=`` gets an INI file in a temp folder, and its
    unsaved-changes prompt answers Discard (a real ``QMessageBox`` would block a closing window
    forever offscreen). Tests that care replace ``window.prompt_unsaved`` themselves. The real
    export dialog fails the test instead of blocking.
    """
    try:
        from PySide6.QtCore import QSettings

        from desktop_audio_recorder import settings as settings_module
        from desktop_audio_recorder.ui import main_window
        from desktop_audio_recorder.ui.export_dialog import UnsavedChoice
    except ImportError:  # pragma: no cover - environment test reports this
        return
    folder = tmp_path_factory.mktemp("settings")

    def temp_settings():
        store = QSettings(str(folder / "settings.ini"), QSettings.Format.IniFormat)
        return settings_module.Settings(store)

    def no_export_dialog(*args, **kwargs):
        pytest.fail("the real export dialog was opened; set window.prompt_export")

    monkeypatch.setattr(main_window, "Settings", temp_settings)
    monkeypatch.setattr(main_window, "ask_unsaved", lambda parent, prompt: UnsavedChoice.DISCARD)
    monkeypatch.setattr(main_window, "ask_export", no_export_dialog)

    def no_recovery_dialog(*args, **kwargs):
        pytest.fail("the real recovery prompt was opened; set window.prompt_recovery")

    monkeypatch.setattr(main_window, "ask_recovery", no_recovery_dialog)

    def no_open_dialog(*args, **kwargs):
        pytest.fail("the real Open dialog was opened; pass a path or set window.prompt_open_path")

    monkeypatch.setattr(main_window, "ask_open_path", no_open_dialog)


@pytest.fixture(autouse=True)
def _no_real_device_watch(monkeypatch):
    """``MainWindow`` without an explicit ``device_watch=`` gets a ``DeviceWatch`` over a
    ``FakeEndpointMonitor`` (never real Core Audio events). ``window._watch`` is the watch and
    ``window._watch._monitor`` the fake, whose ``fire()`` sends events."""
    try:
        from desktop_audio_recorder.ui import device_watch, main_window
    except ImportError:  # pragma: no cover
        return

    def fake_watch(parent=None):
        return device_watch.DeviceWatch(parent, monitor_factory=FakeEndpointMonitor)

    monkeypatch.setattr(main_window, "DeviceWatch", fake_watch)


class FakeEndpointMonitor:
    """Stands in for ``audio.endpoints.EndpointMonitor``. ``endpoints`` maps id -> Endpoint;
    ``default_id`` is the default render endpoint; ``fire(event)`` calls the callback (from the
    calling thread, or a new thread with ``thread=True``, like Core Audio does)."""

    def __init__(self, on_event=None):
        from desktop_audio_recorder.audio.endpoints import DEVICE_STATE_ACTIVE, Endpoint

        self.on_event = on_event
        self.started = False
        self.closed = False
        inzone = Endpoint("{inzone}", "Speakers (INZONE H9 II - Game)", DEVICE_STATE_ACTIVE)
        realtek = Endpoint("{realtek}", "Speakers (Realtek(R) Audio)", DEVICE_STATE_ACTIVE)
        self.endpoints = {e.id: e for e in (inzone, realtek)}
        self.default_id = inzone.id

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True
        self.started = False

    def default_render(self):
        return self.endpoints.get(self.default_id)

    def endpoint(self, device_id):
        return self.endpoints.get(device_id)

    def find_render(self, name):
        for endpoint in self.endpoints.values():
            if endpoint.name == name and endpoint.active:
                return endpoint
        return None

    def set_state(self, device_id, state):
        import dataclasses

        self.endpoints[device_id] = dataclasses.replace(self.endpoints[device_id], state=state)

    def fire(self, event, thread=False):
        if not self.started:
            return
        if thread:
            import threading

            worker = threading.Thread(target=self.on_event, args=(event,))
            worker.start()
            worker.join()
        else:
            self.on_event(event)


# --------------------------------------------------------------------------- fake PyAudio

WASAPI_HOST_INDEX = 2
MME_HOST_INDEX = 0


def output_device(index, name, channels=2, rate=48000.0, host=WASAPI_HOST_INDEX):
    return {
        "index": index,
        "name": name,
        "hostApi": host,
        "maxInputChannels": 0,
        "maxOutputChannels": channels,
        "defaultSampleRate": rate,
        "isLoopbackDevice": False,
    }


def input_device(index, name, channels=2, rate=48000.0, host=WASAPI_HOST_INDEX):
    return {
        "index": index,
        "name": name,
        "hostApi": host,
        "maxInputChannels": channels,
        "maxOutputChannels": 0,
        "defaultSampleRate": rate,
        "isLoopbackDevice": False,
    }


def loopback_device(index, output_name, channels=2, rate=48000.0):
    return {
        "index": index,
        "name": f"{output_name} [Loopback]",
        "hostApi": WASAPI_HOST_INDEX,
        "maxInputChannels": channels,
        "maxOutputChannels": 0,
        "defaultSampleRate": rate,
        "isLoopbackDevice": True,
    }


def machine_devices():
    """Mirror of the real device list on the dev machine (see PROGRESS.md)."""
    return [
        input_device(1, "Microphone (Realtek(R) Audio)", host=MME_HOST_INDEX),
        output_device(3, "Speakers (INZONE H9 II - Game)", 8, host=MME_HOST_INDEX),
        output_device(10, "Speakers (INZONE H9 II - Game)", 8),
        output_device(11, "Speakers (Realtek(R) Audio)", 2),
        input_device(12, "Microphone (Realtek(R) Audio)", 2),
        loopback_device(13, "Speakers (INZONE H9 II - Game)", 8),
        loopback_device(14, "Speakers (Realtek(R) Audio)", 2),
    ]


class FakeStream:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.started = kwargs.get("start", True)
        self.stopped = False
        self.closed = False

    @property
    def callback(self):
        return self.kwargs["stream_callback"]

    @property
    def is_output(self):
        return bool(self.kwargs.get("output"))

    def start_stream(self):
        self.started = True

    def stop_stream(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakePyAudio:
    def __init__(self, module):
        self._module = module
        # Snapshot like PortAudio does on initialise.
        self._devices = {d["index"]: dict(d) for d in module.devices}
        self._default_output = module.default_output
        self.streams = []
        self.terminated = False
        module.instances.append(self)

    def get_host_api_info_by_type(self, api_type):
        if not self._module.has_wasapi or api_type != self._module.paWASAPI:
            raise OSError("host API not found")
        return {
            "index": WASAPI_HOST_INDEX,
            "type": self._module.paWASAPI,
            "name": "Windows WASAPI",
            "defaultOutputDevice": self._default_output,
        }

    def get_device_count(self):
        return max(self._devices, default=-1) + 1

    def get_device_info_by_index(self, index):
        try:
            return dict(self._devices[index])
        except KeyError:
            # Gaps in the fake index space look like non-WASAPI devices.
            return {
                "index": index,
                "name": f"other {index}",
                "hostApi": 99,
                "maxInputChannels": 0,
                "maxOutputChannels": 0,
                "defaultSampleRate": 44100.0,
            }

    def open(self, **kwargs):
        if kwargs.get("output") and self._module.fail_output_open:
            raise OSError("cannot open output stream")
        stream = FakeStream(kwargs)
        self.streams.append(stream)
        return stream

    def terminate(self):
        self.terminated = True


def make_fake_pyaudio(devices=None, default_output=10, has_wasapi=True):
    module = SimpleNamespace(
        paWASAPI=13,
        paFloat32=1,
        paContinue=0,
        devices=machine_devices() if devices is None else devices,
        default_output=default_output,
        has_wasapi=has_wasapi,
        fail_output_open=False,
        instances=[],
    )
    module.PyAudio = lambda: FakePyAudio(module)
    return module


@pytest.fixture
def fake_pyaudio():
    return make_fake_pyaudio()


# --------------------------------------------------------------------------- fake recorder (UI)


def make_loopback(output_name="Speakers (INZONE H9 II - Game)", channels=8, rate=48000):
    from desktop_audio_recorder.audio.devices import LoopbackDevice

    return LoopbackDevice(
        index=13,
        name=f"{output_name} [Loopback]",
        sample_rate=rate,
        channels=channels,
        output_index=10,
        output_name=output_name,
        output_channels=channels,
    )


class FakeDeviceManager:
    """Stands in for ``DeviceManager`` in widget tests."""

    def __init__(self, device=None, error=None):
        self.device = device if device is not None else make_loopback()
        self.error = error
        self.calls = 0

    def resolve_default_loopback(self, pa=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.device


class FakeRecorder:
    """Stands in for ``LoopbackRecorder`` in widget tests.

    ``start_gate`` / ``stop_gate`` (``threading.Event``) make ``start()``/``stop()`` block until
    set, to observe the transient STARTING/STOPPING states. ``stop()`` writes a real float32 WAV
    (a 440 Hz sine of ``duration`` seconds) into ``temp_dir``, which the window then loads.
    """

    def __init__(self, temp_dir, device=None):
        import threading

        from desktop_audio_recorder.audio.recorder import RecorderState

        self.temp_dir = temp_dir
        self.device = device if device is not None else make_loopback()
        self.state = RecorderState.IDLE
        self.elapsed = 0.0
        self.levels = (0.0, 0.0)
        self.failure = None  # LoopbackRecorder.failure (a live writer error)
        self.data_gap = 0.0
        self.keepalive_active = True
        self.result_error = None  # RecordingResult.error of the next stop()
        self.start_error = None
        self.stop_error = None
        self.pause_error = None
        self.start_gate = None
        self.stop_gate = None
        self.switch_gate = None
        self.switch_to = None  # the device switch_device() moves to (else it stays)
        self.switch_error = None  # switch_device() raises this (no device left)
        self.switch_fallback = None  # SwitchResult.error: stayed on the previous device
        self.devices_used = []
        self.duration = 2.5
        self.samples = None  # [frames, 2] float32 to write instead of the sine
        self.write_invalid_file = False  # makes loading the result fail
        self.calls = []
        self.threads = []
        self._threading = threading
        self._count = 0

    def _record(self, name):
        self.calls.append(name)
        self.threads.append((name, self._threading.current_thread().name))

    def start(self):
        from desktop_audio_recorder.audio.recorder import RecorderError, RecorderState

        self._record("start")
        if self.start_gate is not None:
            self.start_gate.wait(5)
        if self.start_error is not None:
            raise self.start_error
        if self.state is not RecorderState.IDLE:
            raise RecorderError(f"cannot start while {self.state.value}")
        self.state = RecorderState.RECORDING
        self.devices_used = [self.device]
        return self.device

    def switch_device(self):
        from desktop_audio_recorder.audio.recorder import (
            RecorderError,
            RecorderState,
            SwitchResult,
        )

        self._record("switch")
        if self.switch_gate is not None:
            self.switch_gate.wait(5)
        if self.state is RecorderState.IDLE:
            raise RecorderError("cannot switch devices: not recording")
        if self.switch_error is not None:
            raise self.switch_error
        previous = self.device
        if self.switch_fallback is not None:
            return SwitchResult(previous, previous, self.elapsed, self.switch_fallback)
        new = self.switch_to or previous
        self.device = new
        self.devices_used.append(new)
        return SwitchResult(new, previous, self.elapsed)

    def pause(self):
        from desktop_audio_recorder.audio.recorder import RecorderError, RecorderState

        self._record("pause")
        if self.pause_error is not None:
            raise self.pause_error
        if self.state is not RecorderState.RECORDING:
            raise RecorderError(f"cannot pause while {self.state.value}")
        self.state = RecorderState.PAUSED

    def resume(self):
        from desktop_audio_recorder.audio.recorder import RecorderError, RecorderState

        self._record("resume")
        if self.state is not RecorderState.PAUSED:
            raise RecorderError(f"cannot resume while {self.state.value}")
        self.state = RecorderState.RECORDING

    def stop(self):
        from desktop_audio_recorder.audio.recorder import (
            RecorderError,
            RecorderState,
            RecordingResult,
        )

        self._record("stop")
        if self.stop_gate is not None:
            self.stop_gate.wait(5)
        if self.state is RecorderState.IDLE:
            raise RecorderError("cannot stop: not recording")
        self.state = RecorderState.IDLE
        if self.stop_error is not None:
            raise self.stop_error
        self._count += 1
        path = self.temp_dir / f"fake-recording-{self._count}.wav"
        frames = int(self.duration * self.device.sample_rate)
        if self.write_invalid_file:
            path.write_bytes(b"RIFF")
        elif self.samples is not None:
            import soundfile as sf

            sf.write(str(path), self.samples, self.device.sample_rate, subtype="FLOAT")
            frames = len(self.samples)
        else:
            write_test_wav(path, self.duration, self.device.sample_rate)
        return RecordingResult(
            path=path,
            sample_rate=self.device.sample_rate,
            channels=2,
            frames=frames,
            zero_filled_frames=0,
            input_overflows=0,
            wall_clock_seconds=self.duration,
            keepalive_active=True,
            device=self.devices_used[0] if self.devices_used else self.device,
            error=self.result_error,
            devices=tuple(self.devices_used),
        )


def write_test_wav(path, seconds, rate=48000, hz=440.0, amplitude=0.5, channels=2):
    """A float32 WAV holding a sine (the same on every channel)."""
    import numpy as np
    import soundfile as sf

    t = np.arange(round(seconds * rate)) / rate
    x = (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    sf.write(str(path), np.repeat(x[:, None], channels, axis=1), rate, subtype="FLOAT")
    return path


def speech_like(seconds, rate=48000, seed=2, pauses=True, peak=0.3):
    """A voice-like test signal: harmonic "syllables" (random pitch 100-250 Hz with a glide,
    harmonics up to 4 kHz falling off as 1/k, 0.12-0.3 s each) separated by short gaps and,
    with ``pauses``, longer pauses between "words"; without, the syllables overlap (no gaps).

    Returns ``(mono float32 signal, boolean mask of the frames inside a syllable)``.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n = round(seconds * rate)
    out = np.zeros(n)
    active = np.zeros(n, dtype=bool)
    pos = round(0.3 * rate) if pauses else 0
    while pos < n:
        dur = round(rng.uniform(0.12, 0.3) * rate)
        f0, glide = rng.uniform(100, 250), rng.uniform(-0.3, 0.3)
        m = min(dur, n - pos)
        t = np.arange(m) / rate
        phase = 2 * np.pi * np.cumsum(f0 * (1 + glide * t / (dur / rate))) / rate
        x = sum(np.sin(k * phase) / k for k in range(1, 25) if k * f0 < 4000)
        out[pos : pos + m] += x * np.sin(np.pi * np.arange(m) / dur) ** 0.5
        active[pos : pos + m] = True
        if pauses:
            gap = rng.choice([rng.uniform(0.03, 0.1), rng.uniform(0.3, 0.6)], p=[0.7, 0.3])
        else:
            gap = -0.05
        pos += dur + round(gap * rate)
    return (peak * out / np.max(np.abs(out))).astype(np.float32), active


# --------------------------------------------------------------------------- fake playback stream


class FakeOutputStream:
    """Stands in for ``sounddevice.OutputStream``. Tests drive it: set ``time`` (the stream
    clock) and call ``pump(frames)`` to run one callback, ``finish()`` for the finished
    callback."""

    def __init__(self, samplerate, channels, callback, finished_callback, latency=0.02):
        self.samplerate = samplerate
        self.channels = channels
        self.callback = callback
        self.finished_callback = finished_callback
        self.latency = latency
        self.time = 100.0
        self.started = False
        self.aborted = False
        self.closed = False
        self.completed = False  # the callback raised CallbackStop
        self.start_error = None
        self.dac_offset = latency  # outputBufferDacTime - currentTime

    def start(self):
        if self.start_error is not None:
            raise self.start_error
        self.started = True

    def abort(self):
        self.aborted = True

    def close(self):
        self.closed = True

    def pump(self, frames, status=0):
        import numpy as np
        import sounddevice as sd

        out = np.full((frames, self.channels), np.nan, dtype=np.float32)
        info = SimpleNamespace(
            outputBufferDacTime=self.time + self.dac_offset if self.dac_offset is not None else 0,
            currentTime=self.time,
        )
        try:
            self.callback(out, frames, info, status)
        except sd.CallbackStop:
            self.completed = True
        return out

    def finish(self):
        self.finished_callback()


class FakeBackend:
    """``StreamBackend`` that hands out ``FakeOutputStream``s."""

    def __init__(self):
        self.streams = []
        self.device_name = "Fake Speakers"
        self.open_error = None
        self.start_error = None

    @property
    def stream(self):
        return self.streams[-1] if self.streams else None

    def open_stream(self, *, samplerate, channels, callback, finished_callback):
        if self.open_error is not None:
            raise self.open_error
        stream = FakeOutputStream(samplerate, channels, callback, finished_callback)
        stream.start_error = self.start_error
        self.streams.append(stream)
        return stream


@pytest.fixture
def fake_backend():
    return FakeBackend()
