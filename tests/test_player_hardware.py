"""ClipPlayer on the real default output (plays a quiet chirp).

1. The position advances at real time and playback finishes on its own.
2. The playhead matches what is rendered: the default device's WASAPI loopback is captured
   while the clip plays; cross-correlation finds where the clip sits in the capture, and every
   polled ``(wall time, position)`` is compared with the wall time at which that clip frame
   reached the loopback. sounddevice's stream clock, PortAudio's and ``time.perf_counter`` are
   all QueryPerformanceCounter on Windows, so the times are directly comparable.

Skipped without WASAPI or a default device; deselect with ``-m "not hardware"``.
"""

import threading
import time

import numpy as np
import pytest
from scipy.signal import chirp, fftconvolve

from desktop_audio_recorder.audio.clip import AudioClip
from desktop_audio_recorder.audio.devices import DeviceError, DeviceManager
from desktop_audio_recorder.audio.downmix import interleaved_to_stereo
from desktop_audio_recorder.audio.player import ClipPlayer, PlaybackState

pytestmark = pytest.mark.hardware

RATE = 48000
AMPLITUDE = 0.05  # about -26 dBFS


@pytest.fixture(scope="module")
def sd():
    sounddevice = pytest.importorskip("sounddevice")
    try:
        DeviceManager().resolve_default_loopback()
        sounddevice.query_devices(kind="output")
    except (DeviceError, OSError, ValueError, sounddevice.PortAudioError) as exc:
        pytest.skip(f"no WASAPI loopback / default output device: {exc}")
    return sounddevice


def chirp_clip(seconds):
    t = np.arange(int(seconds * RATE)) / RATE
    x = AMPLITUDE * chirp(t, f0=200, t1=seconds, f1=6000, method="logarithmic")
    ramp = int(0.01 * RATE)
    x[:ramp] *= np.linspace(0, 1, ramp)
    x[-ramp:] *= np.linspace(1, 0, ramp)
    return AudioClip(np.column_stack([x, x]).astype(np.float32), RATE)


def poll(player, until, interval=0.005):
    samples = []
    while time.perf_counter() < until:
        samples.append((time.perf_counter(), player.position))
        if player.update():
            break
        time.sleep(interval)
    return samples


def test_position_advances_at_real_time(sd):
    clip = chirp_clip(1.5)
    player = ClipPlayer()
    player.set_clip(clip)
    started = time.perf_counter()
    player.play()
    try:
        samples = poll(player, started + 3.0)
    finally:
        player.close()
    finished = time.perf_counter()
    assert player.state is PlaybackState.STOPPED
    assert player.last_error is None
    assert player.position == pytest.approx(clip.duration)
    # Finishes on its own about one clip length (plus output latency) after starting.
    assert 1.5 <= finished - started < 1.5 + 0.3

    moving = [(w, p) for w, p in samples if 0.1 < p < 1.4]
    w, p = np.array(moving).T
    slope, offset = np.polyfit(w - started, p, 1)
    assert slope == pytest.approx(1.0, abs=0.02)
    # Position lags the wall clock by the output latency (tens of ms), never leads it.
    lag = (w - started) - p
    assert 0.0 <= np.median(lag) < 0.15
    # Monotonic: the playhead never jumps backwards.
    assert np.all(np.diff(np.array([p for _, p in samples])) >= 0)


def test_playhead_matches_rendered_audio(sd):
    import pyaudiowpatch as pyaudio

    clip = chirp_clip(2.0)
    pa = pyaudio.PyAudio()
    try:
        device = DeviceManager(pyaudio).resolve_default_loopback(pa)
        blocks = []  # (callback wall time, raw bytes)
        lock = threading.Lock()

        def on_input(data, frame_count, time_info, status):
            now = time.perf_counter()
            with lock:
                blocks.append((now, data))
            return (None, pyaudio.paContinue)

        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=device.channels,
            rate=device.sample_rate,
            input=True,
            input_device_index=device.index,
            frames_per_buffer=480,
            stream_callback=on_input,
        )
        stream.start_stream()
        try:
            time.sleep(0.3)
            player = ClipPlayer()
            player.set_clip(clip)
            player.play()
            try:
                samples = poll(player, time.perf_counter() + 3.0)
            finally:
                player.close()
            time.sleep(0.3)
        finally:
            stream.stop_stream()
            stream.close()
    finally:
        pa.terminate()

    if device.sample_rate != RATE:
        pytest.skip(f"loopback runs at {device.sample_rate} Hz; this check assumes {RATE}")

    # Capture frame k reached us at about: callback time - (frames delivered so far - k) / rate.
    captured = []
    offsets = []
    delivered = 0
    for wall, data in blocks:
        block = interleaved_to_stereo(data, device.channels)
        captured.append(block)
        delivered += len(block)
        offsets.append(wall - delivered / RATE)
    capture = np.concatenate(captured).mean(axis=1)
    capture_t0 = float(np.median(offsets))  # wall time of capture frame 0 (jitter-robust)

    reference = clip.samples[:, 0]
    corr = fftconvolve(capture, reference[::-1], mode="valid")
    lag = int(np.argmax(np.abs(corr)))
    # The match must stand out from the rest of the correlation (other apps may be playing).
    assert np.abs(corr[lag]) > 5 * np.median(np.abs(corr))

    moving = [(w, p) for w, p in samples if 0.1 < p < 1.9]
    w, p = np.array(moving).T
    rendered_at = capture_t0 + (lag + p * RATE) / RATE  # when that clip frame hit the loopback
    error = w - rendered_at  # > 0: the playhead is behind what has been rendered
    print(
        f"playhead vs loopback: median {np.median(error) * 1000:.1f} ms, "
        f"p5 {np.percentile(error, 5) * 1000:.1f} ms, p95 {np.percentile(error, 95) * 1000:.1f} ms"
    )
    assert abs(np.median(error)) < 0.05
    assert np.percentile(np.abs(error), 95) < 0.08
