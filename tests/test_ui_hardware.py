"""Record through the real main window on the real default device (plays a quiet tone).

Checks that the GUI thread never stalls during start/pause/resume/stop and that the meter moves.
Skipped without WASAPI or a default device; deselect with ``-m "not hardware"``.
"""

import time

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import Qt, QTimer

from desktop_audio_recorder.audio.devices import DeviceError, DeviceManager
from desktop_audio_recorder.audio.recorder import LoopbackRecorder
from desktop_audio_recorder.ui.main_window import MainWindow
from desktop_audio_recorder.ui.state import AppState

pytestmark = pytest.mark.hardware

RATE = 48000
AMPLITUDE = 0.05  # about -26 dBFS
MAX_GUI_STALL_S = 0.1


@pytest.fixture(scope="module")
def sd():
    sounddevice = pytest.importorskip("sounddevice")
    try:
        DeviceManager().resolve_default_loopback()
        sounddevice.query_devices(kind="output")
    except (DeviceError, OSError, ValueError, sounddevice.PortAudioError) as exc:
        pytest.skip(f"no WASAPI loopback / default output device: {exc}")
    return sounddevice


def tone(seconds, hz=1000.0):
    t = np.arange(int(seconds * RATE)) / RATE
    x = AMPLITUDE * np.sin(2 * np.pi * hz * t)
    ramp = int(0.005 * RATE)
    x[:ramp] *= np.linspace(0, 1, ramp)
    x[-ramp:] *= np.linspace(1, 0, ramp)
    return np.column_stack([x, x]).astype(np.float32)


def test_record_pause_stop_through_window(sd, qtbot, tmp_path):
    recorder = LoopbackRecorder(temp_dir=tmp_path)
    window = MainWindow(recorder=recorder)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(lambda: window.device is not None, timeout=5000)

    # Heartbeat on the GUI thread: the largest gap between ticks is the worst stall.
    beats = []
    meter_peak = [0.0]

    def beat():
        beats.append(time.perf_counter())
        meter_peak[0] = max(meter_peak[0], *window.meter.fractions)

    heartbeat = QTimer()
    heartbeat.setTimerType(Qt.TimerType.PreciseTimer)
    heartbeat.timeout.connect(beat)
    heartbeat.start(5)
    qtbot.wait(100)

    qtbot.mouseClick(window.transport.record_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.state is AppState.RECORDING, timeout=5000)
    sd.play(tone(1.0), RATE)
    qtbot.wait(1200)
    qtbot.mouseClick(window.transport.pause_button, Qt.MouseButton.LeftButton)
    assert window.state is AppState.PAUSED
    qtbot.wait(400)
    qtbot.mouseClick(window.transport.pause_button, Qt.MouseButton.LeftButton)
    assert window.state is AppState.RECORDING
    qtbot.wait(400)
    elapsed_before_stop = recorder.elapsed
    qtbot.mouseClick(window.transport.stop_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=5000)
    qtbot.wait(50)
    heartbeat.stop()
    sd.stop()

    gaps = np.diff(beats)
    print(
        f"heartbeat: {len(beats)} beats, max gap {gaps.max() * 1000:.1f} ms, "
        f"p99 {np.percentile(gaps, 99) * 1000:.1f} ms; meter peak fraction {meter_peak[0]:.2f}; "
        f"elapsed {elapsed_before_stop:.3f} s"
    )
    assert gaps.max() < MAX_GUI_STALL_S

    # -26 dBFS tone -> fraction ~0.57 (other apps playing can only raise it).
    assert meter_peak[0] > 0.5

    result = window.result
    assert result is not None
    assert result.path.exists()
    assert result.duration == pytest.approx(elapsed_before_stop, abs=0.1)
    assert result.duration == pytest.approx(1.6, abs=0.15)  # 1.2 s + 0.4 s, pause excluded
    info = sf.info(result.path)
    assert (info.samplerate, info.channels) == (result.sample_rate, 2)
    assert window.transport.elapsed_label.text().startswith("00:01.")

    window.close()
    assert not result.path.exists()  # unsaved clip discarded on close (until Phase 6)
