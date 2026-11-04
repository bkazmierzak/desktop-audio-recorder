"""LoopbackRecorder with mocked PyAudio and a controllable clock."""

import time

import numpy as np
import pytest
import soundfile as sf
from conftest import loopback_device, make_fake_pyaudio, output_device

from desktop_audio_recorder.audio.devices import DeviceManager
from desktop_audio_recorder.audio.downmix import MINUS_3DB
from desktop_audio_recorder.audio.recorder import (
    ActiveClock,
    LoopbackRecorder,
    RecorderError,
    RecorderState,
)

SR = 48000
BLOCK = 480  # 10 ms


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class Harness:
    """A recorder wired to a fake PyAudio module and fake clock."""

    def __init__(self, tmp_path, module=None, **kwargs):
        self.module = module or make_fake_pyaudio()
        self.clock = FakeClock()
        self.levels = []
        self.recorder = LoopbackRecorder(
            DeviceManager(self.module),
            temp_dir=tmp_path,
            clock=self.clock,
            on_level=lambda left, right: self.levels.append((left, right)),
            **kwargs,
        )

    def start(self):
        device = self.recorder.start()
        self.pa = self.module.instances[-1]
        self.capture = next(s for s in self.pa.streams if s.kwargs.get("input"))
        self.channels = device.channels
        return device

    def feed(self, seconds, value=0.25, channel=0):
        """Deliver ``seconds`` of audio in BLOCK-sized callbacks, advancing the clock."""
        for _ in range(int(round(seconds * SR / BLOCK))):
            self.clock.advance(BLOCK / SR)
            block = np.zeros((BLOCK, self.channels), dtype=np.float32)
            block[:, channel] = value
            self.capture.callback(block.tobytes(), BLOCK, {}, 0)

    def stop(self):
        result = self.recorder.stop()
        data, rate = sf.read(result.path, dtype="float32", always_2d=True)
        assert rate == SR
        assert len(data) == result.frames
        return result, data


@pytest.fixture
def h(tmp_path):
    return Harness(tmp_path)


def test_continuous_recording_length_and_format(h):
    h.start()
    assert h.capture.started
    assert h.capture.kwargs["channels"] == 8
    assert h.capture.kwargs["rate"] == SR
    assert h.capture.kwargs["input_device_index"] == 13
    h.feed(2.0)
    result, data = h.stop()
    assert result.frames == 2 * SR
    assert result.zero_filled_frames == 0
    assert result.channels == 2 and data.shape == (2 * SR, 2)
    assert result.wall_clock_seconds == pytest.approx(2.0)
    assert result.duration == pytest.approx(2.0)
    assert sf.info(result.path).subtype == "FLOAT"
    assert result.path.parent == h.recorder._temp_dir


def test_8ch_input_is_downmixed(h):
    h.start()
    h.feed(0.1, value=0.5, channel=2)  # centre channel
    _, data = h.stop()
    np.testing.assert_allclose(data[0], [0.5 * MINUS_3DB] * 2, rtol=1e-6)


def test_mono_device_is_duplicated(tmp_path):
    module = make_fake_pyaudio(
        devices=[output_device(0, "Mono out", 1), loopback_device(1, "Mono out", 1)],
        default_output=0,
    )
    h = Harness(tmp_path, module)
    device = h.start()
    assert device.channels == 1
    h.feed(0.1, value=0.3)
    _, data = h.stop()
    np.testing.assert_allclose(data, np.full((int(0.1 * SR), 2), 0.3), rtol=1e-6)


def test_gap_in_the_middle_is_zero_filled(h):
    h.start()
    h.feed(1.0, value=0.25)
    h.clock.advance(2.0)  # loopback delivers nothing for 2 s
    time.sleep(0.2)  # let the writer's idle ticks run
    h.feed(1.0, value=0.5)
    result, data = h.stop()
    assert result.frames == 4 * SR
    assert result.zero_filled_frames == 2 * SR
    assert result.wall_clock_seconds == pytest.approx(4.0)
    # Data stays where the wall clock puts it.
    assert np.all(data[:SR, 0] == 0.25)
    assert np.all(data[SR : 3 * SR] == 0)
    assert np.all(data[3 * SR :, 0] == 0.5)


def test_gap_at_end_is_zero_filled(h):
    h.start()
    h.feed(1.0)
    h.clock.advance(1.5)
    result, data = h.stop()
    assert result.frames == int(2.5 * SR)
    assert result.zero_filled_frames == int(1.5 * SR)
    assert np.all(data[SR:] == 0)


def test_gap_at_start_is_zero_filled(h):
    h.start()
    h.clock.advance(0.5)
    h.feed(0.5, value=0.25)
    result, data = h.stop()
    assert result.frames == SR
    assert result.zero_filled_frames == SR // 2
    assert np.all(data[: SR // 2] == 0)
    assert np.all(data[SR // 2 :, 0] == 0.25)


def test_no_data_at_all_is_all_zeros(h):
    h.start()
    h.clock.advance(1.5)
    time.sleep(0.15)
    result, data = h.stop()
    assert result.frames == int(1.5 * SR)
    assert result.zero_filled_frames == result.frames
    assert not data.any()


def test_short_jitter_is_not_zero_filled(h):
    h.start()
    h.feed(0.5)
    h.clock.advance(0.2)  # below the 0.25 s gap threshold
    h.feed(0.5)
    result, _ = h.stop()
    assert result.zero_filled_frames == 0
    assert result.frames == SR


def test_clock_drift_never_inserts_zeros(h):
    """Audio clock 0.1 % slower than the wall clock: no glitches, just a slightly short file."""
    h.start()
    for _ in range(1000):  # 10 s
        h.clock.advance(BLOCK / SR * 1.001)
        h.capture.callback(np.zeros((BLOCK, 8), np.float32).tobytes(), BLOCK, {}, 0)
    result, _ = h.stop()
    assert result.zero_filled_frames == 0
    assert result.frames == 10 * SR


def test_zero_fill_can_be_disabled(tmp_path):
    h = Harness(tmp_path, zero_fill=False)
    h.start()
    h.feed(1.0)
    h.clock.advance(2.0)
    time.sleep(0.15)
    h.feed(1.0)
    h.clock.advance(1.0)
    result, _ = h.stop()
    assert result.frames == 2 * SR
    assert result.zero_filled_frames == 0


def test_pause_drops_audio_and_time(h):
    h.start()
    h.feed(1.0, value=0.25)
    h.recorder.pause()
    assert h.recorder.state is RecorderState.PAUSED
    h.feed(1.0, value=0.9)  # discarded
    h.clock.advance(5.0)  # long silent pause: must not be zero-filled
    time.sleep(0.15)
    h.recorder.resume()
    assert h.recorder.state is RecorderState.RECORDING
    h.feed(1.0, value=0.5)
    assert h.recorder.elapsed == pytest.approx(2.0)
    result, data = h.stop()
    assert result.frames == 2 * SR
    assert result.zero_filled_frames == 0
    assert result.wall_clock_seconds == pytest.approx(2.0)
    assert not np.any(data == 0.9)


def test_gap_after_resume_still_zero_filled(h):
    h.start()
    h.feed(0.5)
    h.recorder.pause()
    h.clock.advance(3.0)
    h.recorder.resume()
    h.clock.advance(1.0)  # silent after resume
    h.feed(0.5)
    result, _ = h.stop()
    assert result.frames == 2 * SR
    assert result.zero_filled_frames == SR


def test_level_callback_reports_downmixed_peaks(h):
    h.start()
    h.feed(0.05, value=-0.6, channel=1)  # FR
    h.stop()
    assert h.levels
    left, right = h.levels[-1]
    assert left == 0.0
    assert right == pytest.approx(0.6)
    assert h.recorder.levels == h.levels[-1]


def test_keepalive_stream_plays_silence_on_output_device(h):
    h.start()
    assert h.recorder.keepalive_active
    out = next(s for s in h.pa.streams if s.is_output)
    assert out.kwargs["output_device_index"] == 10
    assert out.kwargs["channels"] == 8
    assert out.kwargs["rate"] == SR
    data, flag = out.callback(None, 256, {}, 0)
    assert data == bytes(256 * 8 * 4)
    assert flag == h.module.paContinue
    result, _ = h.stop()
    assert result.keepalive_active
    assert out.stopped and out.closed


def test_keepalive_failure_falls_back_to_zero_fill(tmp_path):
    module = make_fake_pyaudio()
    module.fail_output_open = True
    h = Harness(tmp_path, module)
    h.start()
    assert not h.recorder.keepalive_active
    h.feed(0.5)
    h.clock.advance(1.0)
    result, _ = h.stop()
    assert not result.keepalive_active
    assert result.frames == int(1.5 * SR)


def test_keepalive_can_be_disabled(tmp_path):
    h = Harness(tmp_path, keepalive=False)
    h.start()
    assert not any(s.is_output for s in h.pa.streams)
    h.stop()


def test_streams_and_pyaudio_cleaned_up(h):
    h.start()
    h.feed(0.1)
    h.stop()
    assert all(s.stopped and s.closed for s in h.pa.streams)
    assert h.pa.terminated
    assert h.recorder.state is RecorderState.IDLE


def test_each_recording_resolves_current_default_device(h):
    h.start()
    h.feed(0.1)
    first, _ = h.stop()
    h.module.default_output = 11  # user switched to the Realtek speakers
    device = h.start()
    h.feed(0.1)
    second, _ = h.stop()
    assert first.device.output_index == 10
    assert device.output_index == 11 and device.channels == 2
    assert second.device.index == 14
    assert first.path != second.path
    assert len(h.module.instances) == 2


def test_input_overflow_counted(h):
    h.start()
    h.clock.advance(BLOCK / SR)
    h.capture.callback(np.zeros((BLOCK, 8), np.float32).tobytes(), BLOCK, {}, 0x2)
    result, _ = h.stop()
    assert result.input_overflows == 1


@pytest.mark.parametrize("action", ["pause", "resume", "stop"])
def test_invalid_transitions_when_idle(h, action):
    with pytest.raises(RecorderError):
        getattr(h.recorder, action)()


def test_invalid_transitions_while_recording(h):
    h.start()
    with pytest.raises(RecorderError):
        h.recorder.start()
    with pytest.raises(RecorderError):
        h.recorder.resume()
    h.recorder.pause()
    with pytest.raises(RecorderError):
        h.recorder.pause()
    h.recorder.stop()


def test_start_failure_cleans_up(tmp_path):
    module = make_fake_pyaudio(default_output=-1)
    h = Harness(tmp_path, module)
    with pytest.raises(Exception, match="No default playback"):
        h.recorder.start()
    assert h.recorder.state is RecorderState.IDLE
    assert module.instances[-1].terminated


def test_active_clock():
    clock = ActiveClock(10.0)
    assert clock.active(12.0) == pytest.approx(2.0)
    clock.pause(12.0)
    assert not clock.is_active_at(13.0)
    assert clock.active(15.0) == pytest.approx(2.0)
    clock.resume(15.0)
    assert clock.is_active_at(16.0)
    assert clock.active(16.5) == pytest.approx(3.5)
    assert clock.active(11.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------- live health


def wait_for(predicate, timeout=3.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def failing_writes(recorder, after_blocks, error):
    original = recorder._write_block
    calls = {"n": 0}

    def write(block):
        calls["n"] += 1
        if calls["n"] > after_blocks:
            raise error
        original(block)

    recorder._write_block = write


def test_writer_failure_is_reported_live_and_the_audio_before_it_kept(h):
    h.start()
    failing_writes(h.recorder, 10, OSError(28, "No space left on device"))
    assert h.recorder.failure is None
    h.feed(0.5)
    assert wait_for(lambda: h.recorder.failure is not None)
    assert h.recorder.failure == "the disk is full"
    result, data = h.stop()
    assert result.error == "the disk is full"
    assert result.frames == 10 * BLOCK
    assert np.allclose(data[:, 0], 0.25)


def test_writer_failure_before_any_audio_raises_on_stop(h):
    h.start()
    failing_writes(h.recorder, 0, RuntimeError("cannot write"))
    h.feed(0.1)
    assert wait_for(lambda: h.recorder.failure == "cannot write")
    with pytest.raises(RecorderError) as info:
        h.recorder.stop()
    assert isinstance(info.value.__cause__, RuntimeError)
    assert h.recorder.state is RecorderState.IDLE


def test_successful_recording_has_no_error(h):
    h.start()
    h.feed(0.2)
    result, _ = h.stop()
    assert result.error is None
    assert h.recorder.failure is None


def test_data_gap_tracks_time_without_audio(h):
    assert h.recorder.data_gap == 0.0
    h.start()
    h.feed(0.5)
    assert wait_for(lambda: h.recorder.data_gap < 0.05)
    h.clock.advance(1.5)  # nothing delivered
    assert h.recorder.data_gap == pytest.approx(1.5, abs=0.05)
    h.feed(0.1)
    assert wait_for(lambda: h.recorder.data_gap < 0.05)
    h.stop()
    assert h.recorder.data_gap == 0.0


def test_data_gap_does_not_grow_while_paused(h):
    h.start()
    h.feed(0.3)
    assert wait_for(lambda: h.recorder.data_gap < 0.05)
    h.recorder.pause()
    h.clock.advance(5.0)
    assert h.recorder.data_gap < 0.05
    h.stop()


@pytest.mark.parametrize(
    "exc, text",
    [
        (MemoryError(), "out of memory"),
        (OSError(28, "No space left on device"), "the disk is full"),
        (RuntimeError("Error: disk full"), "the disk is full"),
        (ValueError("weird"), "weird"),
        (ValueError(), "ValueError"),
    ],
)
def test_describe_failure(exc, text):
    from desktop_audio_recorder.audio.recorder import describe_failure

    assert describe_failure(exc) == text


# --------------------------------------------------------------------------- following the default


def usb_devices():
    """The machine's devices plus a 44.1 kHz USB headset."""
    from conftest import machine_devices

    return machine_devices() + [
        output_device(15, "Headphones (USB Audio)", 2, rate=44100.0),
        loopback_device(16, "Headphones (USB Audio)", 2, rate=44100.0),
    ]


def switch(h, default_output):
    h.module.default_output = default_output
    result = h.recorder.switch_device()
    h.pa = h.module.instances[-1]
    h.capture = next(s for s in h.pa.streams if s.kwargs.get("input"))
    h.channels = result.device.channels
    return result


def test_switch_to_the_new_default_device(h):
    h.start()
    old_pa, old_capture = h.pa, h.capture
    old_keepalive = next(s for s in old_pa.streams if s.is_output)
    h.feed(1.0, value=0.25)
    h.clock.advance(0.1)  # the switch takes a while
    result = switch(h, 11)
    assert result.switched and result.error is None
    assert result.previous.output_name == "Speakers (INZONE H9 II - Game)"
    assert result.device.output_name == "Speakers (Realtek(R) Audio)"
    assert result.at == pytest.approx(1.1)
    assert old_capture.closed and old_keepalive.closed and old_pa.terminated
    assert h.pa is not old_pa and h.capture.kwargs["channels"] == 2 and h.capture.started
    keepalive = next(s for s in h.pa.streams if s.is_output)
    assert keepalive.kwargs["output_device_index"] == 11 and keepalive.kwargs["channels"] == 2
    out, _ = keepalive.callback(None, 256, {}, 0)
    assert out == bytes(256 * 2 * 4)
    assert h.recorder.device.output_name == "Speakers (Realtek(R) Audio)"
    h.feed(1.0, value=0.5)
    result, data = h.stop()
    assert result.device.output_name == "Speakers (INZONE H9 II - Game)"  # the first one
    assert [d.output_name for d in result.devices] == [
        "Speakers (INZONE H9 II - Game)",
        "Speakers (Realtek(R) Audio)",
    ]
    assert result.device_names == "Speakers (INZONE H9 II - Game), then Speakers (Realtek(R) Audio)"
    # 1 s of the first device, the 0.1 s switch gap as silence, 1 s of the second.
    assert result.frames == pytest.approx(2.1 * SR, abs=BLOCK)
    assert np.all(data[: SR - BLOCK, 0] == 0.25)
    gap = data[SR + BLOCK : int(1.1 * SR) - BLOCK]
    assert len(gap) and not gap.any()
    assert np.all(data[int(1.1 * SR) + BLOCK :, 0] == 0.5)
    assert result.zero_filled_frames == pytest.approx(0.1 * SR, abs=BLOCK)


def test_switch_to_a_device_at_another_rate(tmp_path):
    h = Harness(tmp_path, make_fake_pyaudio(usb_devices()))
    h.start()
    h.feed(0.5, value=0.25)
    switch(h, 15)
    assert h.capture.kwargs["rate"] == 44100
    rate = 44100
    block = 441
    t = np.arange(rate) / rate
    tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    for a in range(0, rate, block):  # 1 s of a 1 kHz tone from the USB device
        h.clock.advance(block / rate)
        h.capture.callback(np.repeat(tone[a : a + block, None], 2, 1).tobytes(), block, {}, 0)
    result, data = h.stop()  # read back at 48 kHz (Harness checks the rate)
    assert result.sample_rate == SR
    assert result.frames == pytest.approx(1.5 * SR, abs=2 * BLOCK)
    second = data[int(0.55 * SR) : int(1.45 * SR), 0]
    spectrum = np.abs(np.fft.rfft(second * np.hanning(len(second))))
    assert np.argmax(spectrum) * SR / len(second) == pytest.approx(1000, abs=2)
    assert np.max(np.abs(second)) == pytest.approx(0.5, abs=0.01)


def test_switch_falls_back_to_the_previous_device(h):
    h.start()
    h.feed(0.3)
    h.module.devices = [d for d in h.module.devices if d["index"] != 14]  # no Realtek loopback
    result = switch(h, 11)
    assert not result.switched and "No loopback" in result.error
    assert result.device.output_name == "Speakers (INZONE H9 II - Game)"
    assert h.capture.kwargs["input_device_index"] == 13 and h.capture.started
    h.feed(0.3)
    result, data = h.stop()
    assert result.frames == pytest.approx(0.6 * SR, abs=BLOCK)


def test_switch_with_no_device_left_raises_and_stop_keeps_the_audio(h):
    h.start()
    h.feed(0.5)
    h.module.default_output = -1
    h.module.devices = []
    with pytest.raises(RecorderError):
        h.recorder.switch_device()
    assert h.recorder.state is RecorderState.RECORDING and not h.recorder.keepalive_active
    h.clock.advance(0.5)
    result = h.recorder.stop()
    assert result.frames == pytest.approx(1.0 * SR, abs=BLOCK)  # zero-filled to the stop
    assert [d.output_name for d in result.devices] == ["Speakers (INZONE H9 II - Game)"]


def test_switch_while_paused(h):
    h.start()
    h.feed(0.5, value=0.25)
    h.recorder.pause()
    h.clock.advance(1.0)
    switch(h, 11)
    h.feed(0.2, value=0.9)  # paused: dropped
    h.clock.advance(0.003)  # (a block stamped at the very resume instant would count)
    h.recorder.resume()
    h.clock.advance(0.05)
    h.feed(0.5, value=0.5)
    result, data = h.stop()
    assert result.frames == pytest.approx(1.05 * SR, abs=BLOCK)
    assert not np.any(data[:, 0] == 0.9)
    assert np.all(data[-int(0.4 * SR) :, 0] == 0.5)


def test_switch_needs_a_recording(h):
    with pytest.raises(RecorderError):
        h.recorder.switch_device()


def test_named_loopback(fake_pyaudio):
    from desktop_audio_recorder.audio.devices import DeviceError

    dm = DeviceManager(fake_pyaudio)
    pa = dm.create_pyaudio()
    assert dm.resolve_named_loopback(pa, "Speakers (Realtek(R) Audio)").index == 14
    with pytest.raises(DeviceError):
        dm.resolve_named_loopback(pa, "Speakers (Gone)")
