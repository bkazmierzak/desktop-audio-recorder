"""Headless WASAPI loopback recorder.

Pipeline::

    PortAudio callback thread          writer thread                     disk
    ─────────────────────────          ─────────────                     ────
    (timestamp, raw bytes) ──queue──►  downmix N ch → stereo  ──►  temp float32 WAV
                                       peak level → on_level()
                                       pause filtering, zero-fill

Silence gaps. A WASAPI loopback stream delivers no packets while nothing is rendering to the
endpoint, which would shorten the recording. Two defences:

1. **Keep-alive:** a silent output stream is opened on the same playback device for the whole
   recording, so the audio engine keeps the endpoint running and loopback keeps delivering
   (silent) packets.
2. **Zero-fill (fallback):** the writer tracks wall-clock time (excluding pauses). If no data has
   arrived for longer than ``gap_threshold`` seconds, it pads the file with zeros so the timeline
   stays aligned to the wall clock. Zero-fill is only triggered by a delivery gap, never by slow
   clock drift between the sound card and the system clock, so it can't insert glitches into
   continuous audio.

Following the default device (``switch_device()``): when Windows switches the default playback
device mid-recording, the UI can move the recording to it. Both streams are closed, PortAudio is
re-initialised (it only re-reads the device list when no instance is left), the new default is
resolved and new capture and keep-alive streams are opened; it takes about 50-100 ms. The file
keeps its format: every capture stream is a *segment* whose blocks carry it through the queue,
so the writer downmixes each block with its own device's channel count, converts a segment at
another rate to the file's rate (``StreamResampler``, the ``resample_poly`` filter), and places
the first block of a new segment at its wall-clock position (the switch gap becomes silence, so
the timeline stays true). If the new device can't be opened the recording goes back to the
previous one (``SwitchResult.error``); if that fails too, ``switch_device`` raises and nothing is
captured any more (the caller stops the recording).

Health, polled by the UI while recording: ``failure`` is set as soon as the writer thread fails
(for example the disk is full), and ``data_gap`` is how long (active time) no audio has arrived.
With the keep-alive running, loopback delivers data continuously, so a long gap means the
capture stream died (device unplugged or disabled). ``stop()`` after a writer failure keeps
what was written before it (``RecordingResult.error`` says why it ended), unless nothing was.

Debug CLI::

    uv run python -m desktop_audio_recorder.audio.recorder --seconds 10 out.wav
"""

from __future__ import annotations

import enum
import logging
import queue
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from desktop_audio_recorder.audio.devices import DeviceManager, LoopbackDevice
from desktop_audio_recorder.audio.downmix import interleaved_to_stereo
from desktop_audio_recorder.audio.resample import StreamResampler

log = logging.getLogger(__name__)

OUTPUT_CHANNELS = 2
DEFAULT_FRAMES_PER_BUFFER = 1024
DEFAULT_GAP_THRESHOLD = 0.25  # seconds without data before zero-fill kicks in
_POLL_INTERVAL = 0.05  # writer-thread queue timeout, seconds
_FLUSH_INTERVAL = 1.0  # seconds of audio between file flushes
_ZERO_CHUNK_SECONDS = 1.0  # write long gaps in chunks of this size
_PA_INPUT_OVERFLOW = 0x2

LevelCallback = Callable[[float, float], None]


class RecorderError(RuntimeError):
    """Invalid recorder state transition or a failure while recording."""


class RecorderState(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"


@dataclass(frozen=True, slots=True)
class RecordingResult:
    path: Path  # temp float32 WAV; the caller owns it from here
    sample_rate: int
    channels: int  # always 2
    frames: int
    zero_filled_frames: int  # frames padded by the wall-clock fallback
    input_overflows: int  # callbacks PortAudio flagged with input overflow
    wall_clock_seconds: float  # active (non-paused) wall-clock time
    keepalive_active: bool
    device: LoopbackDevice  # the device the recording started on (the file's rate is its)
    error: str | None = None  # the recording ended early (the writer failed); audio until then
    devices: tuple[LoopbackDevice, ...] = ()  # every device in order, when it switched

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate

    @property
    def device_names(self) -> str:
        """``"A"``, or ``"A, then B"`` after a switch."""
        names: list[str] = []
        for device in self.devices or (self.device,):
            if not names or names[-1] != device.output_name:
                names.append(device.output_name)
        return ", then ".join(names)


@dataclass(frozen=True, slots=True)
class SwitchResult:
    """What ``switch_device()`` did."""

    device: LoopbackDevice  # recording from this one now
    previous: LoopbackDevice
    at: float  # active seconds into the recording
    error: str | None = None  # the new default couldn't be opened: back on ``previous``

    @property
    def switched(self) -> bool:
        return self.error is None


class _Segment:
    """One capture stream's stretch of the recording (its blocks carry it in the queue)."""

    def __init__(self, device: LoopbackDevice, number: int) -> None:
        self.device = device
        self.number = number
        self.resampler: StreamResampler | None = None  # made by the writer if rates differ


class ActiveClock:
    """Wall-clock time that excludes paused intervals. Thread-safe."""

    def __init__(self, t0: float) -> None:
        self._lock = threading.Lock()
        self._intervals: list[list[float | None]] = [[t0, None]]

    def pause(self, ts: float) -> None:
        with self._lock:
            if self._intervals[-1][1] is None:
                self._intervals[-1][1] = ts

    def resume(self, ts: float) -> None:
        with self._lock:
            if self._intervals[-1][1] is not None:
                self._intervals.append([ts, None])

    def active(self, ts: float) -> float:
        """Seconds of non-paused time between start and ``ts``."""
        with self._lock:
            total = 0.0
            for start, end in self._intervals:
                stop = ts if end is None else min(end, ts)
                if stop > start:
                    total += stop - start
            return total

    def is_active_at(self, ts: float) -> bool:
        """Whether a block stamped ``ts`` belongs to the recording.

        A block is stamped when its callback runs, i.e. at the *end* of the audio it carries,
        so the pause/stop instant itself still counts.
        """
        with self._lock:
            return any(start <= ts and (end is None or ts <= end) for start, end in self._intervals)


_STOP = object()
TEMP_PREFIX = "recording-"  # temp recordings: <temp dir>/recording-<stamp>-<random>.wav


def default_temp_dir() -> Path:
    """``%TEMP%/DesktopAudioRecorder``: where recordings are written while they are made."""
    return Path(tempfile.gettempdir()) / "DesktopAudioRecorder"


def describe_failure(exc: BaseException) -> str:
    """A short reason for the status bar."""
    if isinstance(exc, MemoryError):
        return "out of memory"
    text = str(exc) or type(exc).__name__
    lowered = text.lower()
    if "disk full" in lowered or "no space" in lowered or getattr(exc, "errno", None) == 28:
        return "the disk is full"
    return text


class LoopbackRecorder:
    """Record the default playback device to a temp float32 stereo WAV.

    Not a Qt object: ``on_level(left_peak, right_peak)`` is called from the writer thread, and
    the UI layer is responsible for marshalling it onto the GUI thread.
    """

    def __init__(
        self,
        device_manager: DeviceManager | None = None,
        *,
        on_level: LevelCallback | None = None,
        temp_dir: str | Path | None = None,
        keepalive: bool = True,
        zero_fill: bool = True,
        gap_threshold: float = DEFAULT_GAP_THRESHOLD,
        frames_per_buffer: int = DEFAULT_FRAMES_PER_BUFFER,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._dm = device_manager or DeviceManager()
        self._on_level = on_level
        self._temp_dir = Path(temp_dir) if temp_dir else default_temp_dir()
        self._keepalive_enabled = keepalive
        self._zero_fill = zero_fill
        self._gap_threshold = gap_threshold
        self._frames_per_buffer = frames_per_buffer
        self._clock_fn = clock

        self._lock = threading.Lock()
        self._state = RecorderState.IDLE
        self._reset_session()

    # ----------------------------------------------------------------- public API

    @property
    def state(self) -> RecorderState:
        return self._state

    @property
    def temp_dir(self) -> Path:
        """Where recordings are written while they are made."""
        return self._temp_dir

    @property
    def device(self) -> LoopbackDevice | None:
        """The device of the current (or last) recording."""
        return self._device

    @property
    def elapsed(self) -> float:
        """Active (non-paused) seconds since ``start()``."""
        clock = self._active_clock
        if clock is None:
            return 0.0
        end = self._stop_ts if self._stop_ts is not None else self._clock_fn()
        return clock.active(end)

    @property
    def levels(self) -> tuple[float, float]:
        """Most recent (left, right) block peaks; nominally 0..1, can exceed 1.0 on overs."""
        return self._levels

    @property
    def keepalive_active(self) -> bool:
        return self._keepalive_stream is not None

    @property
    def failure(self) -> str | None:
        """Why the writer thread stopped (while recording), e.g. ``"disk full"``; else ``None``."""
        error = self._writer_error
        if error is None:
            return None
        return describe_failure(error)

    @property
    def data_gap(self) -> float:
        """Active seconds since the last block of audio arrived (0 when not recording)."""
        if self._state is RecorderState.IDLE or self._active_clock is None:
            return 0.0
        return max(0.0, self.elapsed - self._last_data_active)

    def start(self) -> LoopbackDevice:
        """Resolve the current default device and start recording. Returns the device."""
        with self._lock:
            if self._state is not RecorderState.IDLE:
                raise RecorderError(f"cannot start while {self._state.value}")
            self._reset_session()
            try:
                self._open_session()
            except BaseException:
                self._close_streams()
                if self._writer is not None:
                    self._queue.put(_STOP)
                    self._writer.join()
                self._close_file()
                self._terminate_pa()
                raise
            self._state = RecorderState.RECORDING
            log.info(
                "Recording from %s (%d Hz, %d ch, keep-alive %s) to %s",
                self._device.output_name,
                self._device.sample_rate,
                self._device.channels,
                "on" if self.keepalive_active else "off",
                self._path,
            )
            return self._device

    def pause(self) -> None:
        with self._lock:
            if self._state is not RecorderState.RECORDING:
                raise RecorderError(f"cannot pause while {self._state.value}")
            self._active_clock.pause(self._clock_fn())
            self._state = RecorderState.PAUSED

    def resume(self) -> None:
        with self._lock:
            if self._state is not RecorderState.PAUSED:
                raise RecorderError(f"cannot resume while {self._state.value}")
            self._active_clock.resume(self._clock_fn())
            self._state = RecorderState.RECORDING

    def switch_device(self) -> SwitchResult:
        """Move the recording to the current default playback device (see the module
        docstring). Call it from the thread that runs ``start()``/``stop()``, never at the same
        time as them. Raises ``RecorderError`` when not recording, or when neither the new nor
        the previous device can be opened (then nothing is captured until ``stop()``)."""
        with self._lock:
            if self._state is RecorderState.IDLE:
                raise RecorderError("cannot switch devices: not recording")
            previous = self._device
            at = self.elapsed
        # pause()/resume() stay usable meanwhile: they only touch the clock.
        self._close_streams()
        self._terminate_pa()
        try:
            self._pa = self._dm.create_pyaudio()
            device = self._dm.resolve_default_loopback(self._pa)
            self._open_streams(device)
            self._capture_stream.start_stream()
        except Exception as exc:
            reason = str(exc) or type(exc).__name__
            log.warning("Could not switch the recording to the default device: %s", reason)
            self._close_streams()
            try:
                if self._pa is None:
                    self._pa = self._dm.create_pyaudio()
                back = self._dm.resolve_named_loopback(self._pa, previous.output_name)
                self._open_streams(back)
                self._capture_stream.start_stream()
            except Exception as again:
                self._close_streams()
                log.error("Could not reopen %s either", previous.output_name, exc_info=True)
                raise RecorderError("no device to record from") from again
            log.info("Still recording from %s", back.output_name)
            return SwitchResult(back, previous, at, reason)
        log.info(
            "Switched the recording at %.3f s from %s to %s (%d Hz, %d ch, keep-alive %s)",
            at,
            previous.output_name,
            device.output_name,
            device.sample_rate,
            device.channels,
            "on" if self.keepalive_active else "off",
        )
        return SwitchResult(device, previous, at)

    def stop(self) -> RecordingResult:
        """Stop recording, finish the temp WAV and return its description."""
        with self._lock:
            if self._state is RecorderState.IDLE:
                raise RecorderError("cannot stop: not recording")
            self._stop_ts = self._clock_fn()
            self._active_clock.pause(self._stop_ts)  # drops blocks stamped after this
            keepalive_was_active = self.keepalive_active
            self._close_streams()
            self._queue.put(_STOP)
            self._writer.join()
            try:
                self._close_file()
            except Exception as exc:  # e.g. the header can't be written (disk full)
                log.exception("Could not finish the recording file")
                if self._writer_error is None:
                    self._writer_error = exc
                self._file = None
                self._frames_written = 0  # the file isn't usable
            self._terminate_pa()
            self._state = RecorderState.IDLE
            error = self._writer_error
            if error is not None and self._frames_written == 0:
                raise RecorderError("recording failed") from error
            result = RecordingResult(
                path=self._path,
                sample_rate=self._rate,
                channels=OUTPUT_CHANNELS,
                frames=self._frames_written,
                zero_filled_frames=self._zero_filled,
                input_overflows=self._overflows,
                wall_clock_seconds=self._active_clock.active(self._stop_ts),
                keepalive_active=keepalive_was_active,
                device=self._devices[0],
                error=None if error is None else describe_failure(error),
                devices=tuple(self._devices),
            )
            log.info(
                "Recording stopped: %.3f s (%d frames, %d zero-filled, %d overflows)",
                result.duration,
                result.frames,
                result.zero_filled_frames,
                result.input_overflows,
            )
            return result

    # ----------------------------------------------------------- session set-up

    def _reset_session(self) -> None:
        self._pa: Any = None
        self._device: LoopbackDevice | None = getattr(self, "_device", None)
        self._capture_stream: Any = None
        self._keepalive_stream: Any = None
        self._queue: queue.Queue = queue.Queue()
        self._writer: threading.Thread | None = None
        self._writer_error: BaseException | None = None
        self._file: sf.SoundFile | None = None
        self._path: Path | None = None
        self._active_clock: ActiveClock | None = None
        self._stop_ts: float | None = None
        self._frames_written = 0
        self._zero_filled = 0
        self._overflows = 0
        self._last_data_active = 0.0
        self._frames_since_flush = 0
        self._levels = (0.0, 0.0)
        self._silence_cache: dict[tuple[int, int], bytes] = {}
        self._rate = 0  # the file's sample rate (the first device's)
        self._devices: list[LoopbackDevice] = []
        self._segment: _Segment | None = None  # the writer's current segment
        self._align_pending = False  # place the next active block at its wall-clock position

    def _open_session(self) -> None:
        module = self._dm.module
        self._pa_continue = module.paContinue
        self._pa = self._dm.create_pyaudio()
        device = self._dm.resolve_default_loopback(self._pa)
        self._device = device
        self._rate = device.sample_rate

        self._temp_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        with tempfile.NamedTemporaryFile(
            prefix=f"{TEMP_PREFIX}{stamp}-", suffix=".wav", dir=self._temp_dir, delete=False
        ) as handle:
            self._path = Path(handle.name)  # reserve a unique name; soundfile reopens it
        self._file = sf.SoundFile(
            self._path,
            mode="w",
            samplerate=device.sample_rate,
            channels=OUTPUT_CHANNELS,
            subtype="FLOAT",
            format="WAV",
        )

        self._open_streams(device)  # opening takes ~0.1 s: before the clock starts
        self._active_clock = ActiveClock(self._clock_fn())
        self._writer = threading.Thread(
            target=self._writer_loop, name="LoopbackRecorder-writer", daemon=True
        )
        self._writer.start()
        self._capture_stream.start_stream()

    def _open_streams(self, device: LoopbackDevice) -> None:
        """Open the keep-alive (started) and capture (not started) streams on ``device``, as a
        new segment of the recording."""
        module = self._dm.module
        segment = _Segment(device, len(self._devices))
        if self._keepalive_enabled:
            self._keepalive_stream = self._open_keepalive(module, device)
        self._capture_stream = self._pa.open(
            format=module.paFloat32,
            channels=device.channels,
            rate=device.sample_rate,
            input=True,
            input_device_index=device.index,
            frames_per_buffer=self._frames_per_buffer,
            stream_callback=self._input_callback(segment),
            start=False,
        )
        self._device = device
        self._devices.append(device)

    def _open_keepalive(self, module: Any, device: LoopbackDevice) -> Any:
        try:
            stream = self._pa.open(
                format=module.paFloat32,
                channels=device.output_channels,
                rate=device.sample_rate,
                output=True,
                output_device_index=device.output_index,
                frames_per_buffer=self._frames_per_buffer,
                stream_callback=self._output_callback(device.output_channels),
            )
        except Exception:
            log.warning(
                "Could not open keep-alive stream on %s; relying on zero-fill",
                device.output_name,
                exc_info=True,
            )
            return None
        return stream

    # ------------------------------------------------- PortAudio callbacks (RT)

    def _input_callback(self, segment: _Segment):
        put, clock, result = self._queue.put_nowait, self._clock_fn, (None, self._pa_continue)

        def on_input(in_data, frame_count, time_info, status):
            put((clock(), in_data, status, segment))
            return result

        return on_input

    def _output_callback(self, channels: int):
        def on_output(in_data, frame_count, time_info, status):
            key = (frame_count, channels)
            silence = self._silence_cache.get(key)
            if silence is None:
                silence = bytes(frame_count * channels * 4)
                self._silence_cache[key] = silence
            return (silence, self._pa_continue)

        return on_output

    # ------------------------------------------------------------ writer thread

    def _writer_loop(self) -> None:
        try:
            while True:
                try:
                    item = self._queue.get(timeout=_POLL_INTERVAL)
                except queue.Empty:
                    self._fill_idle_gap(self._clock_fn())
                    continue
                if item is _STOP:
                    break
                self._handle_block(*item)
            self._finish_segment()
            self._fill_final_gap()
        except BaseException as exc:  # surfaced from stop()
            log.exception("Recorder writer thread failed")
            self._writer_error = exc

    def _handle_block(self, ts: float, data: bytes, status: int, segment: _Segment) -> None:
        if segment is not self._segment:
            self._finish_segment()
            self._align_pending = self._segment is not None  # a switch: mind the gap
            self._segment = segment
            if segment.device.sample_rate != self._rate:
                segment.resampler = StreamResampler(segment.device.sample_rate, self._rate)
                log.info(
                    "Converting %s from %d Hz to the recording's %d Hz",
                    segment.device.output_name,
                    segment.device.sample_rate,
                    self._rate,
                )
        if status & _PA_INPUT_OVERFLOW:
            self._overflows += 1
        block = interleaved_to_stereo(data, segment.device.channels)
        if len(block):
            peaks = np.abs(block).max(axis=0)
            self._levels = (float(peaks[0]), float(peaks[1]))
            if self._on_level is not None:
                self._on_level(*self._levels)
        clock = self._active_clock
        if not clock.is_active_at(ts):
            return  # paused (or after stop)
        if segment.resampler is not None:
            block = segment.resampler.process(block)
        now_active = clock.active(ts)
        gap = now_active - self._last_data_active > self._gap_threshold
        if self._zero_fill and (gap or self._align_pending):
            # Data resumed after a gap (or from a new device): pad so this block starts where
            # the wall clock says.
            self._write_zeros_until(self._frames_at(now_active) - len(block))
        self._align_pending = False
        self._last_data_active = now_active
        self._write(block)

    def _finish_segment(self) -> None:
        """The current segment is over: write what its resampler still holds."""
        segment = self._segment
        if segment is not None and segment.resampler is not None:
            tail = segment.resampler.flush()
            if len(tail) and self._last_data_active > 0:
                self._write(tail)

    def _fill_idle_gap(self, now: float) -> None:
        """While no data arrives, keep the file roughly in step with the wall clock.

        Stays ``gap_threshold`` behind, so that when data resumes ``_handle_block`` can place it
        precisely.
        """
        if not self._zero_fill:
            return
        now_active = self._active_clock.active(now)
        if now_active - self._last_data_active > self._gap_threshold:
            self._write_zeros_until(self._frames_at(now_active - self._gap_threshold))

    def _fill_final_gap(self) -> None:
        if not self._zero_fill or self._stop_ts is None:
            return
        end_active = self._active_clock.active(self._stop_ts)
        if end_active - self._last_data_active > self._gap_threshold:
            self._write_zeros_until(self._frames_at(end_active))

    def _frames_at(self, active_seconds: float) -> int:
        return int(round(active_seconds * self._rate))

    def _write_zeros_until(self, target_frames: int) -> None:
        missing = target_frames - self._frames_written
        chunk = int(self._rate * _ZERO_CHUNK_SECONDS)
        while missing > 0:
            n = min(missing, chunk)
            self._write(np.zeros((n, OUTPUT_CHANNELS), dtype=np.float32))
            self._zero_filled += n
            missing -= n

    def _write(self, block: np.ndarray) -> None:
        self._write_block(block)
        self._frames_written += len(block)
        self._frames_since_flush += len(block)
        if self._frames_since_flush >= self._rate * _FLUSH_INTERVAL:
            self._file.flush()
            self._frames_since_flush = 0

    def _write_block(self, block: np.ndarray) -> None:
        """The one place that writes audio (tests replace it to simulate a full disk)."""
        self._file.write(block)

    # ------------------------------------------------------------------ cleanup

    def _close_streams(self) -> None:
        for attr in ("_capture_stream", "_keepalive_stream"):
            stream = getattr(self, attr)
            if stream is None:
                continue
            try:
                stream.stop_stream()
            except Exception:
                log.debug("stop_stream failed", exc_info=True)
            try:
                stream.close()
            except Exception:
                log.debug("close failed", exc_info=True)
            setattr(self, attr, None)

    def _close_file(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _terminate_pa(self) -> None:
        if self._pa is not None:
            try:
                self._pa.terminate()
            finally:
                self._pa = None


# ---------------------------------------------------------------------- debug CLI


def _meter(peak: float, width: int = 20) -> str:
    filled = int(round(min(peak, 1.0) * width))
    return "#" * filled + "-" * (width - filled)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import shutil
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m desktop_audio_recorder.audio.recorder",
        description="Record the default playback device (WASAPI loopback) to a WAV file.",
    )
    parser.add_argument("output", nargs="?", type=Path, help="output .wav path")
    parser.add_argument("--seconds", type=float, default=10.0, help="duration (default 10)")
    parser.add_argument("--no-keepalive", action="store_true", help="disable keep-alive stream")
    parser.add_argument("--no-zero-fill", action="store_true", help="disable zero-fill fallback")
    parser.add_argument("--list-devices", action="store_true", help="list output devices")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    dm = DeviceManager()
    if args.list_devices:
        for dev in dm.list_output_devices():
            mark = "*" if dev.is_default else " "
            print(f"{mark} [{dev.index:>2}] {dev.name}  {dev.sample_rate} Hz, {dev.channels} ch")
        if args.output is None:
            return 0
    if args.output is None:
        parser.error("output path is required")

    recorder = LoopbackRecorder(
        dm, keepalive=not args.no_keepalive, zero_fill=not args.no_zero_fill
    )
    device = recorder.start()
    print(f"Recording from: {device.output_name} ({device.sample_rate} Hz, {device.channels} ch)")
    try:
        while recorder.elapsed < args.seconds:
            left, right = recorder.levels
            sys.stdout.write(f"\r{recorder.elapsed:6.1f} s  L {_meter(left)}  R {_meter(right)}")
            sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        result = recorder.stop()
    print()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(result.path, args.output)
    print(
        f"Saved {args.output}: {result.duration:.3f} s "
        f"(wall clock {result.wall_clock_seconds:.3f} s, "
        f"{result.zero_filled_frames} zero-filled frames, "
        f"{result.input_overflows} overflows, "
        f"keep-alive {'on' if result.keepalive_active else 'off'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
