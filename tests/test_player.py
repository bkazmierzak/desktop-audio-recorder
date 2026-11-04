"""ClipPlayer with a fake output stream: copying, ranges, loop, seek and heard position.

The clips here run at 1000 Hz, so one frame is one millisecond, and each sample holds its own
frame index / 1000, so the output buffer shows exactly which frames were played.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import FakeBackend

from desktop_audio_recorder.audio.clip import AudioClip
from desktop_audio_recorder.audio.player import (
    ClipPlayer,
    PlaybackState,
    PlayerError,
    reinitialize_portaudio,
    wasapi_default_output,
)

RATE = 1000


def ramp_clip(frames=1000):
    x = np.arange(frames, dtype=np.float32) / 1000
    return AudioClip(np.column_stack([x, -x]), RATE)


def frames_of(out):
    """Frame indices in an output buffer (NaN-free; silence reads as 0)."""
    assert not np.isnan(out).any()
    np.testing.assert_array_equal(out[:, 0], -out[:, 1])
    return np.rint(out[:, 0] * 1000).astype(int).tolist()


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def player(backend):
    p = ClipPlayer(backend)
    p.set_clip(ramp_clip())
    return p


def run(stream, frames, step=0.1):
    """One callback, then advance the stream clock by ``step`` seconds."""
    out = stream.pump(frames)
    stream.time += step
    return out


# --------------------------------------------------------------------------- basics


def test_play_opens_stream_at_clip_rate(player, backend):
    assert player.state is PlaybackState.STOPPED
    player.play()
    assert player.state is PlaybackState.PLAYING
    assert player.is_playing
    stream = backend.stream
    assert stream.started
    assert stream.samplerate == RATE
    assert stream.channels == 2
    assert player.device_name == "Fake Speakers"
    player.play()  # no-op while playing
    assert len(backend.streams) == 1


def test_play_without_clip_is_a_noop(backend):
    player = ClipPlayer(backend)
    player.play()
    assert backend.streams == []
    player.set_clip(AudioClip.silent(0, RATE))
    player.play()
    assert backend.streams == []
    assert player.position == 0.0


def test_callback_copies_consecutive_frames(player, backend):
    player.play()
    s = backend.stream
    assert frames_of(run(s, 100)) == list(range(0, 100))
    assert frames_of(run(s, 50)) == list(range(100, 150))


def test_output_is_clipped_to_full_scale(backend):
    data = np.array([[1.5, -2.0], [0.5, -0.5], [-1.2, 1.0]], dtype=np.float32)
    player = ClipPlayer(backend)
    player.set_clip(AudioClip(data, RATE))
    original = player.clip.samples.copy()
    player.play()
    out = backend.stream.pump(3)
    np.testing.assert_array_equal(out, [[1.0, -1.0], [0.5, -0.5], [-1.0, 1.0]])
    np.testing.assert_array_equal(player.clip.samples, original)  # the clip keeps its overs


def test_end_of_clip_pads_with_silence_and_stops_after_the_tail_is_heard(player, backend):
    player.seek(0.95)
    player.play()
    s = backend.stream
    s.time = 100.0
    out = s.pump(100)  # frames 950-999 heard 100.02-100.07, then silence
    assert frames_of(out)[:50] == list(range(950, 1000))
    assert (out[50:] == 0).all()
    assert not s.completed  # the stream keeps running (silence) until the tail is heard
    s.time = 100.06
    assert player.update() is False
    assert player.is_playing
    assert player.position_frame == 990
    s.time = 100.1
    assert (s.pump(100) == 0).all()  # more silence, no new end mark
    s.time = 100.07
    assert player.update() is True
    assert player.state is PlaybackState.STOPPED
    assert player.position_frame == 1000
    assert player.last_error is None
    assert s.aborted and s.closed


def test_finished_callback_after_end_is_a_normal_end(player, backend):
    player.seek(0.95)
    player.play()
    s = backend.stream
    s.pump(100)
    s.finish()  # e.g. the host stopped the stream right after the last buffer
    assert player.update() is True
    assert player.last_error is None
    assert player.position_frame == 1000


def test_exact_end_stops_in_the_same_callback(player, backend):
    player.seek(0.9)
    player.play()
    s = backend.stream
    s.time = 100.0
    assert frames_of(s.pump(100)) == list(range(900, 1000))
    s.time = 100.11
    assert player.update() is False  # the end isn't known until the next callback
    assert (s.pump(100) == 0).all()  # end mark: 100.11 + 0.02
    assert player.update() is False
    s.time = 100.13
    assert player.update() is True


def test_play_after_end_restarts_from_the_beginning(player, backend):
    player.seek(1.0)
    assert player.position_frame == 1000
    player.play()
    assert frames_of(run(backend.stream, 10)) == list(range(10))


def test_pause_keeps_heard_position_and_resumes_there(player, backend):
    player.play()
    s = backend.stream
    run(s, 100, step=0.0)  # written up to frame 100; heard from t=100.02
    s.time = 100.05  # 30 ms into the first buffer
    player.pause()
    assert player.state is PlaybackState.STOPPED
    assert player.position_frame == 30  # heard, not the 100 written
    assert s.aborted and s.closed
    player.play()
    assert frames_of(backend.stream.pump(5)) == [30, 31, 32, 33, 34]


def test_stop_returns_to_start(player, backend):
    player.seek(0.5)
    player.play()
    run(backend.stream, 100)
    player.stop()
    assert player.state is PlaybackState.STOPPED
    assert player.position_frame == 0
    assert backend.stream.closed


def test_toggle(player, backend):
    player.toggle()
    assert player.is_playing
    player.toggle()
    assert not player.is_playing


def test_set_clip_stops_playback_and_resets(player, backend):
    player.set_selection(0.1, 0.2)
    player.seek(0.15)
    player.play()
    player.set_clip(ramp_clip(500))
    assert not player.is_playing
    assert backend.stream.closed
    assert player.position == 0.0
    assert player.selection is None


# --------------------------------------------------------------------------- errors


def test_open_error_raises_player_error(player, backend):
    backend.open_error = OSError("device busy")
    with pytest.raises(PlayerError, match="device busy"):
        player.play()
    assert player.state is PlaybackState.STOPPED


def test_start_error_closes_stream(player, backend):
    backend.start_error = OSError("cannot start")
    with pytest.raises(PlayerError, match="cannot start"):
        player.play()
    assert backend.stream.closed
    assert player.state is PlaybackState.STOPPED


def test_unexpected_finish_reports_error(player, backend):
    player.play()
    s = backend.stream
    run(s, 100, step=0.0)
    s.time = 100.07
    s.finish()  # the stream died without the callback reaching the end
    assert player.update() is True
    assert player.state is PlaybackState.STOPPED
    assert player.last_error and "unexpectedly" in player.last_error
    assert player.position_frame == 50


def test_finished_callback_of_a_closed_stream_is_ignored(player, backend):
    player.play()
    old = backend.stream
    player.pause()
    player.play()
    old.finish()
    assert player.update() is False
    assert player.is_playing


def test_underflow_status_is_counted(player, backend):
    player.play()
    backend.stream.pump(10, status=4)
    assert player.underflows == 1


# --------------------------------------------------------------------------- selection & loop


def test_selection_plays_only_the_selection(player, backend):
    player.set_selection(0.2, 0.25)
    assert player.selection == pytest.approx((0.2, 0.25))
    player.play()
    s = backend.stream
    out = run(s, 100)
    assert frames_of(out)[:50] == list(range(200, 250))
    assert (out[50:] == 0).all()
    assert player.update() is True
    assert player.position_frame == 250
    player.play()  # at the selection end: restarts at the selection start
    assert frames_of(backend.stream.pump(3)) == [200, 201, 202]


def test_selection_starts_from_cursor_inside_it(player, backend):
    player.set_selection(0.2, 0.4)
    player.seek(0.3)
    player.play()
    assert frames_of(backend.stream.pump(3)) == [300, 301, 302]


def test_selection_starts_at_its_start_when_cursor_outside(player, backend):
    player.set_selection(0.4, 0.2)  # reversed
    player.seek(0.7)
    assert player.play_region == pytest.approx((0.2, 0.4))
    player.play()
    assert frames_of(backend.stream.pump(3)) == [200, 201, 202]


def test_empty_selection_is_cleared(player):
    player.set_selection(0.3, 0.3)
    assert player.selection is None
    player.set_selection(0.1, 0.2)
    player.set_selection(None)
    assert player.selection is None
    assert player.play_region == (0.0, 1.0)


def test_stop_returns_to_selection_start(player, backend):
    player.set_selection(0.2, 0.4)
    player.play()
    run(backend.stream, 50)
    player.stop()
    assert player.position_frame == 200


def test_loop_wraps_the_selection(player, backend):
    player.loop = True
    player.set_selection(0.1, 0.13)
    player.play()
    s = backend.stream
    assert frames_of(run(s, 70)) == [*range(100, 130), *range(100, 130), *range(100, 110)]
    assert player.update() is False
    assert frames_of(run(s, 25)) == [*range(110, 130), *range(100, 105)]


def test_loop_wraps_the_whole_clip(player, backend):
    player.loop = True
    player.seek(0.99)
    player.play()
    assert frames_of(backend.stream.pump(15)) == [*range(990, 1000), *range(0, 5)]


def test_loop_off_while_playing_stops_at_the_end(player, backend):
    player.loop = True
    player.set_selection(0.1, 0.12)
    player.play()
    s = backend.stream
    run(s, 30)  # wrapped once, now at 110
    player.loop = False
    out = run(s, 30)
    assert frames_of(out)[:10] == list(range(110, 120))
    assert (out[10:] == 0).all()
    assert player.update() is True


def test_loop_on_during_the_tail_keeps_playing(player, backend):
    player.set_selection(0.1, 0.12)
    player.play()
    s = backend.stream
    assert (run(s, 30)[20:] == 0).all()  # reached the end; tail playing
    player.loop = True
    assert frames_of(run(s, 5)) == [100, 101, 102, 103, 104]
    s.time += 1.0
    assert player.update() is False
    assert player.is_playing


# --------------------------------------------------------------------------- seeking


def test_seek_while_stopped_moves_cursor(player):
    player.seek(0.4)
    assert player.position == pytest.approx(0.4)
    player.seek(5.0)
    assert player.position == pytest.approx(1.0)
    player.seek(-1.0)
    assert player.position == 0.0


def test_seek_while_playing_jumps_at_next_callback(player, backend):
    player.play()
    s = backend.stream
    run(s, 100)
    player.seek(0.6)
    assert player.position_frame == 600  # target shown right away
    assert frames_of(run(s, 5)) == [600, 601, 602, 603, 604]


def test_seek_outside_selection_while_playing_switches_to_whole_clip(player, backend):
    player.set_selection(0.1, 0.2)
    player.play()
    s = backend.stream
    run(s, 50)
    player.seek(0.8)
    out = run(s, 250)
    assert frames_of(out)[:200] == list(range(800, 1000))
    assert player.play_region == (0.0, 1.0)


def test_seek_inside_selection_keeps_selection_range(player, backend):
    player.set_selection(0.1, 0.2)
    player.play()
    s = backend.stream
    run(s, 10)
    player.seek(0.19)
    out = run(s, 20)
    assert frames_of(out)[:10] == list(range(190, 200))
    assert (out[10:] == 0).all()


def test_seek_back_during_the_tail_keeps_playing(player, backend):
    player.seek(0.95)
    player.play()
    s = backend.stream
    run(s, 100)  # reached the end
    player.seek(0.5)
    assert frames_of(run(s, 5)) == [500, 501, 502, 503, 504]
    s.time += 1.0
    assert player.update() is False
    assert player.is_playing


# --------------------------------------------------------------------------- heard position


def test_position_interpolates_from_dac_time(player, backend):
    """Callbacks run ``latency`` ahead of the speaker; position follows the speaker."""
    player.play()
    s = backend.stream  # latency 20 ms: buffer written at t is heard from t + 0.02
    assert player.position_frame == 0  # no callback yet
    s.time = 100.0
    s.pump(100)  # frames 0-99, heard 100.02-100.12
    s.time = 100.01
    assert player.position_frame == 0  # nothing audible yet
    s.time = 100.05
    assert player.position_frame == 30
    s.time = 100.10
    s.pump(100)  # frames 100-199, heard from 100.12
    s.time = 100.11
    assert player.position_frame == 90  # still in the first buffer
    s.time = 100.15
    assert player.position_frame == 130


def test_position_after_seek_holds_target_until_audible(player, backend):
    player.play()
    s = backend.stream
    s.pump(100)
    s.time = 100.05
    player.seek(0.5)
    assert player.position_frame == 500
    s.time = 100.06
    s.pump(100)  # first buffer after the seek, heard from 100.08
    s.time = 100.07
    assert player.position_frame == 500
    s.time = 100.09
    assert player.position_frame == 510


def test_position_wraps_in_loop(player, backend):
    player.loop = True
    player.set_selection(0.1, 0.2)
    player.seek(0.19)
    player.play()
    s = backend.stream
    s.time = 100.0
    s.pump(30)  # 190-199 then 100-119, heard from 100.02
    s.time = 100.02 + 0.015
    assert player.position_frame == 105  # 15 ms after 190 -> wrapped to 105
    s.pump(30)  # anchor: frame 120 heard at 100.055
    s.time = 100.045  # 10 ms before the new anchor: 110
    assert player.position_frame == 110


def test_position_clamps_at_end_without_loop(player, backend):
    player.seek(0.95)
    player.play()
    s = backend.stream
    s.pump(100)
    s.time = 100.5
    assert player.position_frame == 1000


def test_zero_dac_time_falls_back_to_current_time_plus_latency(player, backend):
    player.play()
    s = backend.stream
    s.dac_offset = None  # host API reports 0
    s.pump(100)
    s.time = 100.05
    assert player.position_frame == 30  # latency 0.02 from the stream


# --------------------------------------------------------------------------- sounddevice helpers


def test_reinitialize_portaudio_terminates_then_initialises():
    calls = []
    module = SimpleNamespace(_initialized=1)
    module._terminate = lambda: (calls.append("t"), setattr(module, "_initialized", 0))
    module._initialize = lambda: (calls.append("i"), setattr(module, "_initialized", 1))
    reinitialize_portaudio(module)
    assert calls == ["t", "i"]
    assert module._initialized == 1


def test_wasapi_default_output_prefers_wasapi():
    module = SimpleNamespace(
        query_hostapis=lambda: [
            {"name": "MME", "default_output_device": 3},
            {"name": "Windows WASAPI", "default_output_device": 10},
        ],
        query_devices=lambda index=None, kind=None: {"name": f"dev {index}"},
    )
    assert wasapi_default_output(module) == (10, "dev 10")


def test_wasapi_default_output_falls_back_to_portaudio_default():
    module = SimpleNamespace(
        query_hostapis=lambda: [{"name": "MME", "default_output_device": 3}],
        query_devices=lambda index=None, kind=None: {"name": f"default {kind}"},
    )
    assert wasapi_default_output(module) == (None, "default output")


def test_position_keeps_advancing_through_the_tail(player, backend):
    """Silent buffers after the end must not pin the playhead one latency short of the end."""
    player.seek(0.9)
    player.play()
    s = backend.stream  # latency 20 ms; 10 ms buffers written 20 ms ahead
    positions = []
    for _ in range(20):
        s.pump(10)
        s.time += 0.005
        positions.append(player.position_frame)
        s.time += 0.005
        positions.append(player.position_frame)
    assert positions == sorted(positions)
    assert positions[-1] == 1000
    assert max(p for p in positions if p < 1000) >= 990


def test_position_never_steps_back_on_dac_jitter(player, backend):
    player.play()
    s = backend.stream
    s.pump(100)  # frame 0 heard at 100.02
    s.time = 100.05
    assert player.position_frame == 30
    # The next buffer claims frame 100 is heard at 100.14 (consistent would be 100.12).
    s.dac_offset = 0.09
    s.pump(100)
    s.time = 100.051
    assert player.position_frame == 30  # held, not 11
    s.time = 100.17
    assert player.position_frame == 130


def test_position_may_step_back_on_loop_wrap(player, backend):
    player.loop = True
    player.set_selection(0.1, 0.2)
    player.seek(0.18)
    player.play()
    s = backend.stream
    s.time = 100.0
    s.pump(40)  # 180-199, 100-119 heard from 100.02
    s.time = 100.035
    assert player.position_frame == 195
    s.time = 100.045
    assert player.position_frame == 105


# --------------------------------------------------------------------------- edits


def test_replace_clip_keeps_cursor_and_clamps(player, backend):
    player.seek(0.4)
    edited = ramp_clip(600)
    player.replace_clip(edited)
    assert player.clip is edited
    assert player.position_frame == 400
    player.replace_clip(ramp_clip(300))
    assert player.position_frame == 300  # clamped to the new end
    player.replace_clip(ramp_clip(1000), cursor=123)
    assert player.position_frame == 123


def test_replace_clip_pauses_playback_and_plays_the_new_samples(player, backend):
    player.play()
    run(backend.stream, 100)
    edited = AudioClip(np.zeros((1000, 2), np.float32), RATE)
    player.replace_clip(edited, cursor=0)
    assert player.state is PlaybackState.STOPPED
    assert backend.stream.closed
    player.play()
    assert not backend.stream.pump(50).any()  # the edited (silent) samples


def test_replace_clip_clamps_the_selection(player):
    player.set_selection(0.5, 0.9)
    player.replace_clip(ramp_clip(700))
    assert player.selection == pytest.approx((0.5, 0.7))
    player.replace_clip(ramp_clip(400))
    assert player.selection is None


def test_swap_clip_switches_versions_without_interrupting(fake_backend):
    a = AudioClip(np.full((48000, 2), 0.25, np.float32), 48000)
    b = AudioClip(np.full((48000, 2), -0.5, np.float32), 48000)
    player = ClipPlayer(fake_backend)
    player.set_clip(a)
    player.play()
    stream = fake_backend.stream
    assert np.all(stream.pump(100) == 0.25)
    assert player.swap_clip(b) and player.clip is b and player.is_playing
    assert np.all(stream.pump(100) == -0.5)
    assert len(fake_backend.streams) == 1 and not stream.closed
    assert not player.swap_clip(AudioClip(np.zeros((100, 2), np.float32), 48000))
    assert not player.swap_clip(AudioClip(np.zeros((48000, 2), np.float32), 44100))
    assert player.clip is b
    player.stop()
    assert player.swap_clip(a) and player.clip is a  # also while stopped
    assert not ClipPlayer(fake_backend).swap_clip(a)  # nothing to swap


# --------------------------------------------------------------------------- background streams


class GatedBackend(FakeBackend):
    """Records which thread opens/closes streams; ``gate`` holds ``open_stream`` back."""

    def __init__(self):
        import threading

        super().__init__()
        self.gate = threading.Event()
        self.gate.set()
        self.threads = []

    def open_stream(self, **kwargs):
        import threading

        self.threads.append(("open", threading.current_thread().name))
        self.gate.wait(5)
        stream = super().open_stream(**kwargs)
        backend = self

        def abort(stream=stream):
            backend.threads.append(("abort", threading.current_thread().name))
            stream.aborted = True

        stream.abort = abort
        return stream


@pytest.fixture
def bg():
    from desktop_audio_recorder.audio.player import wait_for_stream_thread

    backend = GatedBackend()
    player = ClipPlayer(backend, background=True)
    player.set_clip(ramp_clip())
    yield player, backend
    backend.gate.set()
    player.close()
    assert wait_for_stream_thread()


def settle():
    from desktop_audio_recorder.audio.player import wait_for_stream_thread

    assert wait_for_stream_thread()


def test_background_is_the_default_only_for_the_real_backend(backend):
    assert not ClipPlayer(backend).background
    assert ClipPlayer(backend, background=True).background
    assert ClipPlayer().background


def test_background_play_returns_before_the_stream_opens(bg):
    player, backend = bg
    backend.gate.clear()
    player.seek(0.3)
    player.play()
    assert player.is_playing
    assert not player.streaming
    assert player.position_frame == 300  # holds the start until the stream runs
    assert player.update() is False
    backend.gate.set()
    settle()
    assert player.streaming
    assert backend.stream.started
    assert backend.threads[0][1].startswith("portaudio-out")
    assert frames_of(run(backend.stream, 10)) == list(range(300, 310))


def test_background_close_happens_on_the_stream_thread(bg):
    player, backend = bg
    player.play()
    settle()
    stream = backend.stream
    player.pause()
    assert not player.is_playing and not player.streaming
    settle()
    assert stream.aborted and stream.closed
    assert backend.threads[-1][0] == "abort"
    assert backend.threads[-1][1].startswith("portaudio-out")


def test_pause_while_opening_discards_the_late_stream(bg):
    player, backend = bg
    backend.gate.clear()
    player.play()
    player.pause()
    assert player.state is PlaybackState.STOPPED
    backend.gate.set()
    settle()
    assert not player.streaming
    assert backend.stream.aborted and backend.stream.closed


def test_quick_play_pause_play_ends_with_one_running_stream(bg):
    player, backend = bg
    backend.gate.clear()
    player.play()
    player.pause()
    player.play()
    backend.gate.set()
    settle()
    assert player.streaming
    assert len(backend.streams) == 2
    first, second = backend.streams
    assert first.closed and not second.closed
    assert player._stream is second


def test_a_closing_stream_writes_silence_and_leaves_the_state_alone(bg):
    player, backend = bg
    player.play()
    settle()
    old = backend.stream
    run(old, 100)
    player.pause()
    cursor = player.position_frame
    out = old.pump(50)  # a last callback before the abort lands
    assert (out == 0).all()
    assert player._anchor is None
    assert player.position_frame == cursor


def test_background_open_error_is_reported_by_update(bg):
    player, backend = bg
    backend.open_error = OSError("device busy")
    player.play()  # no exception here
    settle()
    assert player.is_playing
    assert player.update() is True
    assert player.state is PlaybackState.STOPPED
    assert "device busy" in player.last_error
    player.update()
    assert player.last_error is not None  # kept until the next play()


def test_background_start_error_closes_the_stream(bg):
    player, backend = bg
    backend.start_error = OSError("cannot start")
    player.play()
    settle()
    assert player.update() is True
    assert "cannot start" in player.last_error
    assert backend.stream.closed


def test_a_stale_open_error_is_ignored(bg):
    player, backend = bg
    backend.gate.clear()
    backend.open_error = OSError("device busy")
    player.play()
    player.pause()
    backend.gate.set()
    settle()
    backend.open_error = None
    player.play()
    settle()
    assert player.update() is False
    assert player.streaming


def test_close_waits_for_the_stream_thread(bg):
    player, backend = bg
    player.play()
    settle()
    stream = backend.stream
    player.close()
    assert stream.closed  # already closed when close() returns
