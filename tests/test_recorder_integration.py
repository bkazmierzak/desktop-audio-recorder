"""Real-device capture tests: play a quiet known signal with sounddevice while recording loopback.

Skipped when WASAPI or a default playback device isn't available. Deselect with
``uv run pytest -m "not hardware"`` to keep the speakers quiet.

Other apps may be playing loud audio at the same time (the dev machine usually has music or a
game running), so the analysis doesn't assume silence: signals are located by cross-correlating
with exactly what was played, and the tone's frequency is the one whose power rose most compared
with the rest of the recording.
"""

import time

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from desktop_audio_recorder.audio.devices import DeviceError, DeviceManager, resolve_loopback
from desktop_audio_recorder.audio.recorder import LoopbackRecorder

pytestmark = pytest.mark.hardware

TONE_HZ = 1000.0
AMPLITUDE = 0.05  # about -26 dBFS: audible but quiet
RATE = 48000


@pytest.fixture(scope="module")
def sd():
    sounddevice = pytest.importorskip("sounddevice")
    try:
        DeviceManager().resolve_default_loopback()
        sounddevice.query_devices(kind="output")
    except (DeviceError, OSError, ValueError, sounddevice.PortAudioError) as exc:
        pytest.skip(f"no WASAPI loopback / default output device: {exc}")
    return sounddevice


def _ramped(x):
    ramp = int(0.005 * RATE)
    x[:ramp] *= np.linspace(0, 1, ramp)
    x[-ramp:] *= np.linspace(1, 0, ramp)
    return x


def sine(seconds, hz=TONE_HZ):
    t = np.arange(int(seconds * RATE)) / RATE
    return _ramped(AMPLITUDE * np.sin(2 * np.pi * hz * t))


def chirp(seconds, f0=500.0, f1=4000.0):
    t = np.arange(int(seconds * RATE)) / RATE
    return _ramped(AMPLITUDE * signal.chirp(t, f0, seconds, f1))


def play(sd, mono):
    """Play ``mono`` on both channels of the default output, blocking."""
    sd.play(np.column_stack([mono, mono]).astype(np.float32), RATE, blocking=True)


def find_signal(recording, template, count, min_separation):
    """Sample offsets of the ``count`` strongest matches of ``template`` in ``recording``."""
    corr = np.abs(signal.correlate(recording, template, mode="valid", method="fft"))
    peaks = []
    for _ in range(count):
        i = int(np.argmax(corr))
        peaks.append(i)
        corr[max(0, i - min_separation) : i + min_separation] = 0
    return sorted(peaks)


def added_tone_frequency(recording, rate, start, length):
    """Frequency whose power rose most in ``[start, start+length)`` vs the rest."""
    margin = int(0.1 * rate)
    seg = recording[start : start + length]
    background = np.concatenate(
        [recording[: max(0, start - margin)], recording[start + length + margin :]]
    )
    f, p_seg = signal.welch(seg, rate, nperseg=4800)
    _, p_bg = signal.welch(background, rate, nperseg=4800)
    # Absolute excess power, not a ratio: a ratio spikes wherever the background happens to be
    # near-empty (e.g. a hi-hat in the segment at 14 kHz).
    band = (f > 200) & (f < 15000)
    excess = p_seg[band] - p_bg[band]
    return f[band][np.argmax(excess)]


def record(tmp_path, body, **kwargs):
    """Record while ``body()`` runs. ``wall`` is the time from ``start()`` returning to
    ``stop()`` being called; the stream opening/closing latency (driver-dependent: 30-170 ms
    seen on the INZONE device) is outside it."""
    recorder = LoopbackRecorder(temp_dir=tmp_path, **kwargs)
    recorder.start()
    t0 = time.monotonic()
    try:
        marks = body()
    finally:
        wall = time.monotonic() - t0
        result = recorder.stop()
    data, rate = sf.read(result.path, dtype="float32", always_2d=True)
    return result, data, rate, wall, marks


def test_tone_is_captured_with_correct_duration_and_frequency(sd, tmp_path):
    tone = sine(1.0)

    def body():
        time.sleep(0.5)
        play(sd, tone)
        time.sleep(0.5)

    result, data, rate, wall, _ = record(tmp_path, body)
    assert rate == result.sample_rate == RATE
    assert data.shape[1] == 2
    # Length matches the wall clock (start/stop overhead is a few ms).
    assert result.duration == pytest.approx(result.wall_clock_seconds, abs=0.05)
    assert wall - 0.05 < result.duration < wall + 0.15

    mono = data.mean(axis=1)
    (start,) = find_signal(mono, tone, 1, len(tone))
    assert 0.4 < start / rate < 1.0  # played ~0.5 s in, plus output latency
    assert added_tone_frequency(mono, rate, start, len(tone)) == pytest.approx(TONE_HZ, abs=10)


@pytest.mark.parametrize("keepalive", [True, False], ids=["keepalive", "zero-fill-only"])
def test_silent_gap_keeps_wall_clock_length(sd, tmp_path, keepalive):
    gap = 2.0
    marker = chirp(0.5)

    def body():
        time.sleep(0.3)
        first = time.monotonic()
        play(sd, marker)
        time.sleep(gap)  # we play nothing
        second = time.monotonic()
        play(sd, marker)
        time.sleep(0.3)
        return first, second

    result, data, rate, wall, (first, second) = record(tmp_path, body, keepalive=keepalive)
    assert result.keepalive_active is keepalive
    assert result.duration == pytest.approx(result.wall_clock_seconds, abs=0.05)
    assert wall - 0.05 < result.duration < wall + 0.15
    assert result.duration > 0.3 + 0.5 + gap + 0.5

    # The second marker sits as far after the first in the file as it did on the wall clock,
    # so the stretch where we played nothing wasn't collapsed.
    a, b = find_signal(data.mean(axis=1), marker, 2, rate)
    assert (b - a) / rate == pytest.approx(second - first, abs=0.05)


class _FixedOutputManager(DeviceManager):
    """Records a chosen output instead of the default one."""

    def __init__(self, output_index):
        super().__init__()
        self._output_index = output_index

    def resolve_default_loopback(self, pa=None):
        if pa is not None:
            return resolve_loopback(pa, self._output_index, self.module)
        with self.session() as fresh:
            return resolve_loopback(fresh, self._output_index, self.module)


@pytest.fixture
def idle_output():
    """A non-default WASAPI output that (hopefully) has nothing playing on it."""
    try:
        candidates = [
            d
            for d in DeviceManager().list_output_devices()
            if not d.is_default and d.loopback_index is not None
        ]
    except DeviceError as exc:
        pytest.skip(str(exc))
    if not candidates:
        pytest.skip("needs a second WASAPI output device")
    return candidates[0]


def test_idle_endpoint_stall_is_handled(tmp_path, idle_output):
    """On an idle endpoint WASAPI loopback delivers nothing at all.

    Without keep-alive the zero-fill fallback has to supply the whole timeline; with keep-alive
    real (silent) data flows and nothing needs filling. Either way the length is the wall clock.
    Nothing audible is played.
    """
    results = {}
    for keepalive in (False, True):
        recorder = LoopbackRecorder(
            _FixedOutputManager(idle_output.index), temp_dir=tmp_path, keepalive=keepalive
        )
        recorder.start()
        time.sleep(1.5)
        results[keepalive] = recorder.stop()

    bare, kept = results[False], results[True]
    if bare.zero_filled_frames < 0.5 * bare.frames:
        pytest.skip(f"{idle_output.name!r} isn't idle (something is playing on it)")
    for result in (bare, kept):
        assert result.duration == pytest.approx(result.wall_clock_seconds, abs=0.05)
        assert result.duration == pytest.approx(1.5, abs=0.1)
    assert kept.keepalive_active
    assert kept.zero_filled_frames == 0


class _NamedOutputManager(DeviceManager):
    """Treats the output called ``name`` as the default (Windows' own default is untouched)."""

    def __init__(self, name):
        super().__init__()
        self.name = name

    def resolve_default_loopback(self, pa=None):
        if pa is not None:
            return self.resolve_named_loopback(pa, self.name)
        with self.session() as fresh:
            return self.resolve_named_loopback(fresh, self.name)


def test_switching_devices_mid_recording(sd, tmp_path, idle_output):
    """Start on the default device, then move the recording to an idle endpoint (as when the
    Windows default changes). A tone played on the default before the switch is in the file; a
    tone played on it after the switch isn't; the length is still the wall clock. Nothing is
    played on the idle endpoint."""
    default = DeviceManager().resolve_default_loopback()
    manager = _NamedOutputManager(default.output_name)
    recorder = LoopbackRecorder(manager, temp_dir=tmp_path)
    before, after = sine(0.6), sine(0.6, hz=440.0)
    recorder.start()
    try:
        time.sleep(0.3)
        play(sd, before)
        time.sleep(0.2)
        manager.name = idle_output.name
        t = time.monotonic()
        switch = recorder.switch_device()
        switch_seconds = time.monotonic() - t
        time.sleep(0.3)
        play(sd, after)  # on the default device, which isn't recorded any more
        time.sleep(0.3)
    finally:
        result = recorder.stop()
    data, rate = sf.read(result.path, dtype="float32", always_2d=True)
    assert switch.switched, switch.error
    assert switch.device.output_name == idle_output.name
    assert [d.output_name for d in result.devices] == [default.output_name, idle_output.name]
    assert switch_seconds < 0.5
    assert result.duration == pytest.approx(result.wall_clock_seconds, abs=0.05)
    mono = data.mean(axis=1)
    cut = int(switch.at * rate)
    (start,) = find_signal(mono[:cut], before, 1, len(before))
    assert added_tone_frequency(mono[:cut], rate, start, len(before)) == pytest.approx(
        TONE_HZ, abs=10
    )
    tail = data[cut + int(0.05 * rate) :]
    if np.max(np.abs(tail)) > 1e-4:
        pytest.skip(f"{idle_output.name!r} isn't idle (something is playing on it)")
    # Silence from the idle endpoint (keep-alive data), not the 440 Hz tone.
    assert result.keepalive_active
