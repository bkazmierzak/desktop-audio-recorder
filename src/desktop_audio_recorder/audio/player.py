"""``ClipPlayer``: plays an ``AudioClip`` through a sounddevice ``OutputStream`` callback.

No Qt here. The GUI calls ``play()``/``pause()``/``stop()``/``seek()`` and polls ``position``
and ``update()`` from a ~30 Hz timer.

Threading. The PortAudio callback only copies (and clips to ±1.0) from the clip's array into
the output buffer. The GUI and the callback share a few attributes, each replaced by a single
assignment of an immutable value (atomic in CPython), so no locks are needed:

* ``_request`` (GUI → callback): ``(seq, frame, region_start, region_end)``. A new ``seq`` tells
  the callback to jump to ``frame`` and play ``[region_start, region_end)``.
* ``_loop`` (GUI → callback): wrap at the region end instead of stopping.
* ``_anchor`` (callback → GUI): ``(seq, frame, dac_time)``, the first frame of the latest
  buffer and the stream time at which it will be heard.
* ``_seek_dac`` (callback → GUI): ``(seq, dac_time)`` of the first buffer after a jump.
* ``_end_mark`` (callback → GUI): ``(seq, dac_time)`` at which the last frame of the range
  finishes playing, once the callback has written it (without loop).

End of playback. The callback keeps the stream running with silence after the last frame, and
``update()`` closes the stream once the stream clock passes ``_end_mark``. (PortAudio's WASAPI
path reports a stream that raised ``CallbackStop`` as finished when the last buffer is handed
over, about one output latency before it's heard; closing then would cut off the tail.)

Heard position. The callback runs ahead of the speaker by the output latency. ``position``
interpolates from the latest anchor with the stream clock, so the playhead tracks what is
heard rather than what was last written; just after a jump it holds the target until the
first new buffer reaches the speaker. The reported position never steps backwards within one
segment (PortAudio's prefill buffers report slightly inconsistent DAC times, which would make
the playhead jitter back a few ms once), except for a loop wrap or a jump.

Playback range. With a selection, ``play()`` plays the selection (from the cursor if it's
inside, else from the selection start). Without one it plays the whole clip from the cursor
(from the start if the cursor is at the end). ``seek()`` outside the playing range while
playing switches to the whole clip.

Default output. ``SoundDeviceBackend`` re-initialises sounddevice's PortAudio before each
stream (PortAudio snapshots the device list at initialisation) and opens the WASAPI default
output with ``auto_convert`` so Windows converts rate and channel count.

Stream thread. Re-initialising PortAudio and opening a WASAPI stream takes ~90 ms on some
drivers, and closing one ~50 ms, too long for the GUI thread. With ``background=True`` (the
default with the real backend) ``play()`` only publishes the request and returns; the stream is
opened and started on one shared ``portaudio-out`` thread, and closed there too. All players
share that thread, so PortAudio's (non-thread-safe) terminate/initialise never overlaps a
stream operation of another player. Until the stream is running the position holds the start
frame, as it does after a seek. Each stream's callback carries a token: a stream that is still
being closed in the background writes silence and leaves the shared state alone. An open
failure is reported by the next ``update()`` (``last_error``) instead of raising.
"""

from __future__ import annotations

import concurrent.futures
import enum
import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import sounddevice as sd

from desktop_audio_recorder.audio.clip import AudioClip

log = logging.getLogger(__name__)

DEFAULT_LATENCY = 0.05  # seconds; low enough for snappy seeks, safe against GIL hiccups


class PlayerError(RuntimeError):
    """The output stream could not be opened or started."""


class PlaybackState(enum.Enum):
    STOPPED = "stopped"  # no stream; ``position`` is the cursor
    PLAYING = "playing"


class OutputStream(Protocol):
    @property
    def time(self) -> float: ...

    @property
    def latency(self) -> float: ...

    def start(self) -> None: ...

    def abort(self) -> None: ...

    def close(self) -> None: ...


class StreamBackend(Protocol):
    def open_stream(
        self,
        *,
        samplerate: int,
        channels: int,
        callback: Callable[[np.ndarray, int, Any, Any], None],
        finished_callback: Callable[[], None],
    ) -> OutputStream: ...


# --------------------------------------------------------------------------- sounddevice backend


def reinitialize_portaudio(module: Any = sd) -> None:
    """Terminate and re-initialise sounddevice's PortAudio so it re-enumerates devices.

    Closes every sounddevice stream in the process; only call it with none open.
    """
    levels = max(1, module._initialized)
    for _ in range(module._initialized):
        module._terminate()
    for _ in range(levels):
        module._initialize()


def wasapi_default_output(module: Any = sd) -> tuple[int | None, str]:
    """``(device index, name)`` of the WASAPI default output, or ``(None, name)`` for PortAudio's
    default output when WASAPI isn't available."""
    for api in module.query_hostapis():
        if "WASAPI" in api["name"] and api["default_output_device"] >= 0:
            index = api["default_output_device"]
            return index, module.query_devices(index)["name"]
    info = module.query_devices(kind="output")
    return None, info["name"]


class SoundDeviceBackend:
    """Opens ``sounddevice.OutputStream``s on the *current* Windows default output."""

    def __init__(self, latency: float = DEFAULT_LATENCY, follow_default: bool = True) -> None:
        self.latency = latency
        self.follow_default = follow_default
        self.device_name: str | None = None

    def open_stream(self, *, samplerate, channels, callback, finished_callback):
        if self.follow_default:
            reinitialize_portaudio()
        device, name = wasapi_default_output()
        extra = sd.WasapiSettings(auto_convert=True) if device is not None else None
        stream = sd.OutputStream(
            samplerate=samplerate,
            channels=channels,
            dtype="float32",
            device=device,
            latency=self.latency,
            callback=callback,
            finished_callback=finished_callback,
            extra_settings=extra,
        )
        self.device_name = name
        return stream


# --------------------------------------------------------------------------- stream thread

_stream_executor: concurrent.futures.ThreadPoolExecutor | None = None
_stream_executor_lock = threading.Lock()


def stream_executor() -> concurrent.futures.ThreadPoolExecutor:
    """The one thread that opens and closes every player's output streams."""
    global _stream_executor
    with _stream_executor_lock:
        if _stream_executor is None:
            _stream_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="portaudio-out"
            )
        return _stream_executor


def wait_for_stream_thread(timeout: float = 5.0) -> bool:
    """Wait until every queued open/close has run (``False`` on timeout)."""
    try:
        stream_executor().submit(lambda: None).result(timeout)
    except concurrent.futures.TimeoutError:
        return False
    return True


# --------------------------------------------------------------------------- player


class ClipPlayer:
    def __init__(self, backend: StreamBackend | None = None, *, background: bool | None = None):
        """``background``: open and close streams on the shared stream thread (see the module
        docstring); by default only with the real sounddevice backend."""
        self._background = backend is None if background is None else background
        self._backend = backend if backend is not None else SoundDeviceBackend()
        self._clip: AudioClip | None = None
        self._samples = np.zeros((0, 2), dtype=np.float32)
        self._rate = 48000
        self._state = PlaybackState.STOPPED
        self._cursor = 0  # frame; the position while stopped
        self._selection: tuple[int, int] | None = None
        self._loop = False
        self._stream: OutputStream | None = None
        self._stream_token = 0
        self._stream_lock = threading.Lock()  # guards _stream/_stream_token handover
        self._open_error: tuple[int, str] | None = None  # (token, message), background opens
        self._latency = 0.0
        self.last_error: str | None = None
        self.underflows = 0
        # Shared with the callback (see module docstring).
        self._seq = 0
        self._request: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._anchor: tuple[int, int, float] | None = None
        self._seek_dac: tuple[int, float] | None = None
        self._finished_token = 0
        self._end_mark: tuple[int, float] | None = None
        self._last_heard: tuple[int, int] | None = None  # (seq, frame), GUI thread only
        # Callback-private.
        self._cb_seq = 0
        self._cb_pos = 0
        self._cb_region = (0, 0)
        self._cb_tail = 0  # silent frames written past the end of the range

    # ------------------------------------------------------------------ properties

    @property
    def clip(self) -> AudioClip | None:
        return self._clip

    @property
    def state(self) -> PlaybackState:
        return self._state

    @property
    def is_playing(self) -> bool:
        return self._state is PlaybackState.PLAYING

    @property
    def loop(self) -> bool:
        return self._loop

    @loop.setter
    def loop(self, value: bool) -> None:
        self._loop = bool(value)

    @property
    def device_name(self) -> str | None:
        return getattr(self._backend, "device_name", None)

    @property
    def background(self) -> bool:
        return self._background

    @property
    def streaming(self) -> bool:
        """Whether the output stream of the current playback is open and running."""
        return self._stream is not None

    @property
    def selection(self) -> tuple[float, float] | None:
        if self._selection is None:
            return None
        a, b = self._selection
        return a / self._rate, b / self._rate

    @property
    def play_region(self) -> tuple[float, float]:
        """The range being played (while playing) or that ``play()`` would play."""
        if self.is_playing:
            _, _, a, b = self._request
        else:
            a, b = self._region_for(self._cursor)[:2]
        return a / self._rate, b / self._rate

    @property
    def position_frame(self) -> int:
        if self._state is PlaybackState.STOPPED:
            return self._cursor
        return self._heard_frame()

    @property
    def position(self) -> float:
        """Seconds. While playing, the frame being heard now (see module docstring)."""
        return self.position_frame / self._rate

    # ------------------------------------------------------------------ clip & selection

    def set_clip(self, clip: AudioClip | None) -> None:
        """Replace the clip (stops playback, resets cursor and selection)."""
        self._close_stream()
        self._state = PlaybackState.STOPPED
        self._clip = clip
        self._samples = clip.samples if clip is not None else np.zeros((0, 2), dtype=np.float32)
        self._rate = clip.sample_rate if clip is not None else 48000
        self._cursor = 0
        self._selection = None

    def replace_clip(self, clip: AudioClip, cursor: int | None = None) -> None:
        """Swap in an edited version of the clip, keeping the cursor (or moving it to frame
        ``cursor``), clamped to the new length. Pauses playback first; the selection is clamped
        (the caller usually sets a new one)."""
        if self.is_playing:
            self.pause()
        self._close_stream()
        self._state = PlaybackState.STOPPED
        self._clip = clip
        self._samples = clip.samples
        self._rate = clip.sample_rate
        self._cursor = min(max(0, self._cursor if cursor is None else int(cursor)), clip.frames)
        if self._selection is not None:
            a, b = (min(v, clip.frames) for v in self._selection)
            self._selection = (a, b) if b > a else None

    def swap_clip(self, clip: AudioClip) -> bool:
        """Swap in another version of the same audio (same length and rate, e.g. the processed
        preview for an A/B comparison) without interrupting playback: the callback's next
        buffer comes from it, at the same position. Cursor and selection are kept. Returns
        ``False`` (changing nothing) when the lengths or rates differ."""
        current = self._clip
        if current is None or clip.frames != current.frames or clip.sample_rate != self._rate:
            return False
        self._clip = clip
        self._samples = clip.samples  # one assignment: the callback reads it once per buffer
        return True

    def set_selection(self, start: float | None, end: float | None = None) -> None:
        """Select ``[start, end)`` seconds; ``None`` (or an empty range) clears it.

        Takes effect at the next ``play()``.
        """
        if self._clip is None or start is None or end is None:
            self._selection = None
            return
        a, b = self._clip.frame_range(start, end)
        self._selection = (a, b) if b > a else None

    # ------------------------------------------------------------------ transport

    def play(self) -> None:
        """Start playing (no-op while playing). Raises ``PlayerError`` if the device fails
        (in the background mode the failure shows up in ``update()`` instead)."""
        if self.is_playing or self._clip is None or self._clip.frames == 0:
            return
        start_region, end_region, start = self._region_for(self._cursor)
        self._end_mark = None
        self.last_error = None
        self._publish(start, start_region, end_region)
        self._cb_seq = -1  # make the first callback apply the request
        with self._stream_lock:
            self._stream_token += 1
            token = self._stream_token
            self._open_error = None
        self._state = PlaybackState.PLAYING
        log.info(
            "Playback started at %.3f s (range %.3f-%.3f s, loop %s)",
            start / self._rate,
            start_region / self._rate,
            end_region / self._rate,
            self._loop,
        )
        if self._background:
            stream_executor().submit(self._open_stream, token, self._rate)
            return
        try:
            self._open_stream(token, self._rate, raise_errors=True)
        except PlayerError:
            self._state = PlaybackState.STOPPED
            raise

    def _open_stream(self, token: int, rate: int, raise_errors: bool = False) -> None:
        """Open and start the output stream (on the stream thread in the background mode)."""

        def callback(outdata, frames, time_info, status, _token=token):
            self._callback(outdata, frames, time_info, status, _token)

        started = time.perf_counter()
        stream = None
        try:
            try:
                stream = self._backend.open_stream(
                    samplerate=rate,
                    channels=2,
                    callback=callback,
                    finished_callback=lambda: self._on_finished(token),
                )
            except Exception as exc:
                raise PlayerError(f"Could not open the playback device: {exc}") from exc
            self._latency = float(getattr(stream, "latency", 0.0) or 0.0)
            try:
                stream.start()
            except Exception as exc:
                raise PlayerError(f"Could not start playback: {exc}") from exc
        except PlayerError as exc:
            if stream is not None:
                self._dispose(stream)
            log.error("%s", exc)
            if raise_errors:
                raise
            with self._stream_lock:
                if token == self._stream_token:
                    self._open_error = (token, str(exc))
            return
        with self._stream_lock:
            current = token == self._stream_token
            if current:
                self._stream = stream
        if not current:  # paused/stopped while it was opening
            self._dispose(stream)
            return
        log.info(
            "Output stream on %s open in %.0f ms",
            self.device_name,
            (time.perf_counter() - started) * 1e3,
        )

    def pause(self) -> None:
        """Stop the stream and keep the heard position as the cursor."""
        if not self.is_playing:
            return
        self._cursor = self._heard_frame()
        self._close_stream()
        self._state = PlaybackState.STOPPED

    def toggle(self) -> None:
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def stop(self) -> None:
        """Stop and return the cursor to the start of the selection (or of the clip)."""
        self._close_stream()
        self._state = PlaybackState.STOPPED
        self._cursor = self._selection[0] if self._selection else 0

    def seek(self, seconds: float) -> None:
        if self._clip is None:
            return
        frame = self._clip.seconds_to_frame(seconds)
        if not self.is_playing:
            self._cursor = frame
            return
        _, _, a, b = self._request
        if not a <= frame < b:
            a, b = 0, self._clip.frames
        self._publish(frame, a, b)

    def update(self) -> bool:
        """Finish a playback that reached its end, or whose stream failed to open (call from
        the GUI timer). Returns ``True`` when the state changed to ``STOPPED``."""
        stream = self._stream
        if stream is None:
            error = self._open_error
            if self.is_playing and error is not None and error[0] == self._stream_token:
                self._open_error = None
                self.last_error = error[1]
                self._close_stream()
                self._state = PlaybackState.STOPPED
                return True
            return False
        mark = self._end_mark
        reached_end = mark is not None and mark[0] == self._request[0]
        if self._finished_token == self._stream_token and not reached_end:
            # The stream stopped by itself: device lost or failed.
            self._cursor = self._heard_frame()
            self.last_error = "Playback stopped unexpectedly (output device lost?)"
            log.warning(self.last_error)
        elif reached_end and (self._finished_token == self._stream_token or self._now() >= mark[1]):
            self._cursor = self._request[3]
        else:
            return False
        self._close_stream()
        self._state = PlaybackState.STOPPED
        return True

    def close(self, wait: bool = True) -> None:
        """Stop and release the stream; with ``wait``, until the stream thread has closed it."""
        self._close_stream()
        self._state = PlaybackState.STOPPED
        if wait and self._background and not wait_for_stream_thread():
            log.warning("The output stream did not close in time")

    # ------------------------------------------------------------------ internals

    def _region_for(self, cursor: int) -> tuple[int, int, int]:
        """``(region_start, region_end, start_frame)`` for a ``play()`` from ``cursor``."""
        if self._selection is not None:
            a, b = self._selection
        else:
            a, b = 0, self._samples.shape[0]
        start = cursor if a <= cursor < b else a
        return a, b, start

    def _publish(self, frame: int, region_start: int, region_end: int) -> None:
        self._seq += 1
        self._request = (self._seq, frame, region_start, region_end)

    def _close_stream(self) -> None:
        with self._stream_lock:
            stream, self._stream = self._stream, None
            self._stream_token += 1  # ignore its callbacks; an open in flight is now stale
            self._open_error = None
        self._anchor = None
        self._seek_dac = None
        self._end_mark = None
        self._last_heard = None
        if stream is None:
            return
        if self._background:
            stream_executor().submit(self._dispose, stream)
        else:
            self._dispose(stream)

    @staticmethod
    def _dispose(stream: OutputStream) -> None:
        try:
            stream.abort()
        except Exception:
            log.debug("abort() failed", exc_info=True)
        try:
            stream.close()
        except Exception:
            log.debug("close() failed", exc_info=True)

    def _on_finished(self, token: int) -> None:  # PortAudio thread
        self._finished_token = token

    def _now(self) -> float:
        try:
            return float(self._stream.time)
        except Exception:
            return math.inf

    def _heard_frame(self) -> int:
        frame = self._estimate_heard_frame()
        seq, _, a, b = self._request
        last = self._last_heard
        if last is not None and last[0] == seq and frame < last[1]:
            wrapped = self._loop and last[1] - frame > (b - a) / 2
            if not wrapped:
                return last[1]
        self._last_heard = (seq, frame)
        return frame

    def _estimate_heard_frame(self) -> int:
        seq, target, a, b = self._request
        anchor, mark, stream = self._anchor, self._seek_dac, self._stream
        if anchor is None or mark is None or stream is None or anchor[0] != seq or mark[0] != seq:
            return target  # the callback hasn't picked up the latest jump yet
        try:
            now = float(stream.time)
        except Exception:
            return anchor[1]
        if now < mark[1]:
            return target  # the first buffer after the jump isn't audible yet
        pos = anchor[1] + (now - anchor[2]) * self._rate
        length = b - a
        if length > 0 and (pos < a or pos >= b):
            if self._loop:
                pos = a + math.fmod(pos - a, length)
                if pos < a:
                    pos += length
            else:
                pos = min(max(pos, a), b)
        return min(max(round(pos), a), b)

    def _callback(
        self, outdata: np.ndarray, frames: int, time_info: Any, status: Any, token: int
    ) -> None:
        # PortAudio thread: copy only.
        if token != self._stream_token:  # a stream being closed: silence, touch nothing
            outdata.fill(0)
            return
        if status:
            self.underflows += 1
        request = self._request
        dac = time_info.outputBufferDacTime
        if dac <= 0:
            dac = time_info.currentTime + self._latency
        if request[0] != self._cb_seq:
            self._cb_seq = request[0]
            self._cb_pos = request[1]
            self._cb_region = (request[2], request[3])
            self._cb_tail = 0
            self._seek_dac = (request[0], dac)
        pos = self._cb_pos
        start, end = self._cb_region
        # Past the end the anchor keeps counting (virtual frames), so the heard position keeps
        # advancing through the tail and reaches the end when the last frame is heard.
        self._anchor = (self._cb_seq, pos + self._cb_tail, dac)
        samples = self._samples
        written = 0
        while written < frames:
            if pos >= end:
                if self._loop and end > start:
                    pos = start
                    self._cb_tail = 0
                    self._end_mark = None  # loop switched back on during the tail
                    continue
                # Past the end: silence until the GUI closes the stream (see update()).
                outdata[written:] = 0
                if self._end_mark is None or self._end_mark[0] != self._cb_seq:
                    self._end_mark = (self._cb_seq, dac + written / self._rate)
                self._cb_tail += frames - written
                break
            n = min(frames - written, end - pos)
            np.clip(samples[pos : pos + n], -1.0, 1.0, out=outdata[written : written + n])
            written += n
            pos += n
        self._cb_pos = pos
