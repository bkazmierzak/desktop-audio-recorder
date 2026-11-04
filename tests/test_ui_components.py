"""State machine, transport bar and level meter (offscreen widgets, no audio)."""

import math

import pytest
from PySide6.QtCore import Qt

from desktop_audio_recorder.ui.level_meter import (
    FLOOR_DB,
    LevelMeter,
    MeterChannel,
    db_to_fraction,
    level_to_fraction,
    peak_to_db,
)
from desktop_audio_recorder.ui.state import (
    TRANSITIONS,
    AppState,
    EditControls,
    InvalidTransition,
    StateMachine,
    controls_for,
    edit_controls,
    has_progress,
    is_capturing,
)
from desktop_audio_recorder.ui.transport_bar import TransportBar, format_elapsed

S = AppState

# ------------------------------------------------------------------------- state machine


@pytest.mark.parametrize(
    "path",
    [
        [S.STARTING, S.RECORDING, S.PAUSED, S.RECORDING, S.STOPPING, S.LOADING, S.STOPPED],
        [S.STARTING, S.RECORDING, S.STOPPING, S.LOADING, S.STOPPED, S.STARTING, S.RECORDING],
        [S.STARTING, S.IDLE],  # start failed
        [S.STARTING, S.RECORDING, S.PAUSED, S.STOPPING, S.IDLE],  # stop failed
        [S.STARTING, S.RECORDING, S.STOPPING, S.LOADING, S.IDLE],  # load failed
        [S.LOADING, S.STOPPED, S.PLAYING],  # a recovered recording
        [S.LOADING, S.IDLE, S.STARTING],  # recovering it failed
        [S.STARTING, S.RECORDING, S.STOPPING, S.LOADING, S.STOPPED, S.STARTING, S.STOPPED],
        # play, pause (or stop / reach the end), play again, then record
        [S.STARTING, S.RECORDING, S.STOPPING, S.LOADING, S.STOPPED, S.PLAYING, S.STOPPED]
        + [S.PLAYING, S.STOPPED, S.STARTING],
        # a long edit, then play and edit again, then record
        [S.STARTING, S.RECORDING, S.STOPPING, S.LOADING, S.STOPPED, S.EDITING, S.STOPPED]
        + [S.PLAYING, S.STOPPED, S.EDITING, S.STOPPED, S.STARTING],
        # noise reduction (done, cancelled or failed), then undo it, play, record
        [S.STARTING, S.RECORDING, S.STOPPING, S.LOADING, S.STOPPED, S.PROCESSING, S.STOPPED]
        + [S.EDITING, S.STOPPED, S.PROCESSING, S.STOPPED, S.PLAYING, S.STOPPED, S.STARTING],
        # export (done, cancelled or failed), edit, export again, then record
        [S.STARTING, S.RECORDING, S.STOPPING, S.LOADING, S.STOPPED, S.EXPORTING, S.STOPPED]
        + [S.EDITING, S.STOPPED, S.EXPORTING, S.STOPPED, S.PLAYING, S.STOPPED, S.STARTING],
    ],
)
def test_valid_paths(path):
    sm = StateMachine()
    for target in path:
        sm.go(target)
    assert sm.state is path[-1]


@pytest.mark.parametrize(
    ("start", "target"),
    [
        (S.IDLE, S.RECORDING),
        (S.IDLE, S.STOPPING),
        (S.IDLE, S.PAUSED),
        (S.IDLE, S.PLAYING),  # nothing to play
        (S.RECORDING, S.STARTING),
        (S.PAUSED, S.PAUSED),
        (S.STOPPED, S.RECORDING),
        (S.STOPPING, S.RECORDING),
        (S.STOPPING, S.STOPPED),  # the clip must load first
        (S.STARTING, S.PAUSED),
        (S.LOADING, S.PLAYING),
        (S.LOADING, S.STARTING),
        (S.PLAYING, S.STARTING),  # recording is disabled while playing
        (S.PLAYING, S.IDLE),
        (S.RECORDING, S.PLAYING),
        (S.PLAYING, S.EDITING),  # playback is paused (-> STOPPED) before an edit
        (S.IDLE, S.EDITING),  # nothing to edit
        (S.EDITING, S.PLAYING),
        (S.EDITING, S.STARTING),
        (S.EDITING, S.IDLE),  # a failed edit keeps the clip
        (S.LOADING, S.EDITING),
        (S.PLAYING, S.PROCESSING),  # playback is paused before noise reduction
        (S.IDLE, S.PROCESSING),
        (S.EDITING, S.PROCESSING),
        (S.PROCESSING, S.EDITING),
        (S.PROCESSING, S.PLAYING),
        (S.PROCESSING, S.STARTING),
        (S.PROCESSING, S.IDLE),  # cancelled or failed: the clip is unchanged
        (S.PLAYING, S.EXPORTING),  # playback is paused before an export
        (S.IDLE, S.EXPORTING),  # nothing to export
        (S.LOADING, S.EXPORTING),
        (S.EDITING, S.EXPORTING),
        (S.PROCESSING, S.EXPORTING),
        (S.EXPORTING, S.PROCESSING),
        (S.EXPORTING, S.EDITING),
        (S.EXPORTING, S.PLAYING),
        (S.EXPORTING, S.STARTING),  # Save, then record: only after the export has finished
        (S.EXPORTING, S.IDLE),  # cancelled or failed: the clip is still there
    ],
)
def test_invalid_transitions_raise(start, target):
    sm = StateMachine(start)
    assert not sm.can(target)
    with pytest.raises(InvalidTransition):
        sm.go(target)
    assert sm.state is start


def test_every_state_has_transitions():
    assert set(TRANSITIONS) == set(AppState)
    reachable = {S.IDLE}.union(*TRANSITIONS.values())
    assert reachable == set(AppState)


@pytest.mark.parametrize(
    ("state", "record", "pause", "stop", "refresh", "label"),
    [
        (S.IDLE, True, False, False, True, "Pause"),
        (S.STARTING, False, False, False, False, "Pause"),
        (S.RECORDING, False, True, True, False, "Pause"),
        (S.PAUSED, False, True, True, False, "Resume"),
        (S.STOPPING, False, False, False, False, "Pause"),
        (S.LOADING, False, False, False, False, "Pause"),
        (S.STOPPED, True, False, False, True, "Pause"),
        (S.PLAYING, False, False, False, True, "Pause"),
        (S.EDITING, False, False, False, False, "Pause"),
        (S.PROCESSING, False, False, False, False, "Pause"),
        (S.EXPORTING, False, False, False, False, "Pause"),
    ],
)
def test_controls_per_state(state, record, pause, stop, refresh, label):
    c = controls_for(state)
    assert (c.record, c.pause, c.stop, c.refresh, c.pause_label) == (
        record,
        pause,
        stop,
        refresh,
        label,
    )


@pytest.mark.parametrize(
    ("state", "play", "play_label", "stop_playback", "clip"),
    [
        (S.IDLE, False, "Play", False, False),
        (S.STARTING, False, "Play", False, False),
        (S.RECORDING, False, "Play", False, False),
        (S.PAUSED, False, "Play", False, False),
        (S.STOPPING, False, "Play", False, False),
        (S.LOADING, False, "Play", False, False),
        (S.STOPPED, True, "Play", False, True),
        (S.PLAYING, True, "Pause", True, True),
        (S.EDITING, False, "Play", False, False),
        (S.PROCESSING, False, "Play", False, False),
        (S.EXPORTING, False, "Play", False, False),
    ],
)
def test_playback_controls_per_state(state, play, play_label, stop_playback, clip):
    c = controls_for(state)
    assert (c.play, c.play_label, c.stop_playback, c.clip) == (
        play,
        play_label,
        stop_playback,
        clip,
    )


@pytest.mark.parametrize(
    ("state", "edit"),
    [
        (S.IDLE, False),
        (S.STARTING, False),
        (S.RECORDING, False),
        (S.PAUSED, False),
        (S.STOPPING, False),
        (S.LOADING, False),
        (S.STOPPED, True),
        (S.PLAYING, True),  # an edit pauses playback first
        (S.EDITING, False),  # one edit at a time
        (S.PROCESSING, False),
        (S.EXPORTING, False),  # the clip is being read
    ],
)
def test_edit_allowed_per_state(state, edit):
    assert controls_for(state).edit is edit


def test_only_processing_and_exporting_can_be_cancelled():
    assert {s for s in AppState if controls_for(s).cancel} == {S.PROCESSING, S.EXPORTING}
    assert {s for s in AppState if has_progress(s)} == {S.PROCESSING, S.EXPORTING}


def test_export_needs_an_idle_clip():
    # PLAYING is allowed: the export pauses playback first.
    assert {s for s in AppState if controls_for(s).export} == {S.STOPPED, S.PLAYING}


_NONE = EditControls(*([False] * 11))


@pytest.mark.parametrize(
    ("state", "selection", "undo", "redo", "expected"),
    [
        (
            S.STOPPED,
            False,
            False,
            False,
            EditControls(*[False] * 5, True, True, True, False, False, True),
        ),
        (S.STOPPED, True, False, False, EditControls(*[True] * 8, False, False, True)),
        (S.STOPPED, False, True, False, EditControls(*[False] * 5, *[True] * 4, False, True)),
        (
            S.STOPPED,
            False,
            False,
            True,
            EditControls(*[False] * 5, *[True] * 3, False, True, True),
        ),
        (S.PLAYING, True, True, True, EditControls(*[True] * 11)),
        (S.EDITING, True, True, True, _NONE),
        (S.PROCESSING, True, True, True, _NONE),
        (S.EXPORTING, True, True, True, _NONE),
        (S.LOADING, True, True, True, _NONE),
        (S.RECORDING, True, True, True, _NONE),
        (S.IDLE, False, False, False, _NONE),
    ],
)
def test_edit_controls_table(state, selection, undo, redo, expected):
    got = edit_controls(state, has_selection=selection, can_undo=undo, can_redo=redo)
    assert got == expected


def test_is_capturing():
    assert {s for s in AppState if is_capturing(s)} == {
        S.STARTING,
        S.RECORDING,
        S.PAUSED,
        S.STOPPING,
    }


# ------------------------------------------------------------------------- transport bar


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0.0, "00:00.0"),
        (0.3, "00:00.3"),
        (0.96, "00:00.9"),  # truncated, never ahead of the recording
        (3.04, "00:03.0"),
        (59.99, "00:59.9"),
        (61.25, "01:01.2"),
        (599.9, "09:59.9"),
        (3599.95, "59:59.9"),
        (3600.0, "1:00:00.0"),
        (3725.5, "1:02:05.5"),
        (-1.0, "00:00.0"),
        (math.nan, "00:00.0"),
        (math.inf, "00:00.0"),
    ],
)
def test_format_elapsed(seconds, text):
    assert format_elapsed(seconds) == text


@pytest.fixture
def transport(qtbot):
    bar = TransportBar()
    qtbot.addWidget(bar)
    return bar


@pytest.mark.parametrize("state", list(AppState))
def test_transport_buttons_follow_state(transport, state):
    transport.set_state(state)
    c = controls_for(state)
    assert transport.record_button.isEnabled() is c.record
    assert transport.pause_button.isEnabled() is c.pause
    assert transport.stop_button.isEnabled() is c.stop
    assert transport.pause_button.text() == c.pause_label


def test_pause_button_emits_pause_or_resume(transport, qtbot):
    seen = []
    transport.pause_clicked.connect(lambda: seen.append("pause"))
    transport.resume_clicked.connect(lambda: seen.append("resume"))
    transport.set_state(S.RECORDING)
    qtbot.mouseClick(transport.pause_button, Qt.MouseButton.LeftButton)
    transport.set_state(S.PAUSED)
    qtbot.mouseClick(transport.pause_button, Qt.MouseButton.LeftButton)
    assert seen == ["pause", "resume"]


def test_record_and_stop_buttons_emit(transport, qtbot):
    with qtbot.waitSignal(transport.record_clicked, timeout=500):
        qtbot.mouseClick(transport.record_button, Qt.MouseButton.LeftButton)
    transport.set_state(S.RECORDING)
    with qtbot.waitSignal(transport.stop_clicked, timeout=500):
        qtbot.mouseClick(transport.stop_button, Qt.MouseButton.LeftButton)


def test_disabled_button_does_not_emit(transport, qtbot):
    seen = []
    transport.stop_clicked.connect(lambda: seen.append("stop"))
    transport.set_state(S.IDLE)
    qtbot.mouseClick(transport.stop_button, Qt.MouseButton.LeftButton)
    assert seen == []


def test_set_elapsed(transport):
    transport.set_elapsed(75.43)
    assert transport.elapsed_label.text() == "01:15.4"


# ------------------------------------------------------------------------- meter mapping


@pytest.mark.parametrize(
    ("peak", "db"),
    [
        (1.0, 0.0),
        (0.1, -20.0),
        (0.001, -60.0),
        (0.0, FLOOR_DB),
        (-0.5, FLOOR_DB),
        (1e-9, FLOOR_DB),
        (math.nan, FLOOR_DB),
        (2.0, 20 * math.log10(2.0)),
    ],
)
def test_peak_to_db(peak, db):
    assert peak_to_db(peak) == pytest.approx(db)


@pytest.mark.parametrize(
    ("peak", "fraction"),
    [
        (0.0, 0.0),
        (0.001, 0.0),  # -60 dBFS: the floor
        (0.01, 1 / 3),  # -40 dBFS
        (0.1, 2 / 3),  # -20 dBFS
        (0.5, (20 * math.log10(0.5) + 60) / 60),
        (1.0, 1.0),
        (1.5, 1.0),  # over full scale: clamped (and shown as clipped)
    ],
)
def test_level_to_fraction(peak, fraction):
    assert level_to_fraction(peak) == pytest.approx(fraction)


def test_db_to_fraction_clamps():
    assert db_to_fraction(-100) == 0.0
    assert db_to_fraction(6) == 1.0
    assert db_to_fraction(-30) == pytest.approx(0.5)


# ------------------------------------------------------------------------- ballistics


def test_meter_attack_is_instant():
    ch = MeterChannel()
    ch.update(0.1, 0.03)
    assert ch.db == pytest.approx(-20.0)
    assert ch.fraction == pytest.approx(2 / 3)


def test_meter_decays_linearly_in_db():
    ch = MeterChannel(decay_db_per_s=24.0)
    ch.update(1.0, 0.03)
    ch.update(0.0, 0.5)
    assert ch.db == pytest.approx(-12.0)
    ch.update(0.0, 10.0)
    assert ch.db == FLOOR_DB


def test_meter_louder_input_overrides_decay():
    ch = MeterChannel()
    ch.update(0.1, 0.03)
    ch.update(1.0, 0.03)
    assert ch.db == pytest.approx(0.0)


def test_peak_hold_holds_then_falls():
    ch = MeterChannel(peak_hold_s=1.5, decay_db_per_s=24.0)
    ch.update(0.1, 0.03)  # -20 dB
    for _ in range(10):  # 1.0 s of silence: still held
        ch.update(0.0, 0.1)
    assert ch.hold_db == pytest.approx(-20.0)
    assert ch.db < ch.hold_db
    ch.update(0.0, 0.5)  # 1.5 s: hold just expired (0 s over)
    assert ch.hold_db == pytest.approx(-20.0)
    ch.update(0.0, 0.25)  # 0.25 s past the hold time
    assert ch.hold_db == pytest.approx(-26.0)
    ch.update(0.0, 10.0)
    assert ch.hold_db == FLOOR_DB


def test_peak_hold_resets_on_new_peak():
    ch = MeterChannel(peak_hold_s=1.0)
    ch.update(0.1, 0.03)
    ch.update(0.0, 0.9)
    ch.update(0.5, 0.03)  # louder: new hold, timer restarts
    ch.update(0.0, 0.9)
    assert ch.hold_db == pytest.approx(peak_to_db(0.5))


def test_clip_latches_over_full_scale_only():
    ch = MeterChannel(clip_hold_s=2.0)
    ch.update(1.0, 0.03)
    assert not ch.clipped  # exactly full scale is not an over
    ch.update(1.2, 0.03)
    assert ch.clipped
    ch.update(0.1, 1.5)
    assert ch.clipped  # still latched
    ch.update(0.1, 0.6)
    assert not ch.clipped  # > 2 s since the last over


def test_clear_clip_and_at_rest():
    ch = MeterChannel()
    ch.update(3.0, 0.03)
    assert not ch.at_rest
    ch.clear_clip()
    assert not ch.clipped
    for _ in range(100):
        ch.update(0.0, 0.1)
    assert ch.at_rest


# ------------------------------------------------------------------------- meter widget


@pytest.fixture
def meter(qtbot):
    widget = LevelMeter()
    widget.resize(500, widget.sizeHint().height())
    qtbot.addWidget(widget)
    return widget


def test_meter_widget_maps_levels(meter):
    meter.set_levels(0.1, 1.5, 0.03)
    left, right = meter.fractions
    assert left == pytest.approx(2 / 3)
    assert right == 1.0
    assert meter.clipped == (False, True)


def test_meter_click_clears_clip(meter, qtbot):
    meter.set_levels(1.5, 1.5, 0.03)
    with qtbot.waitSignal(meter.clip_cleared, timeout=500):
        qtbot.mouseClick(meter, Qt.MouseButton.LeftButton)
    assert meter.clipped == (False, False)


def test_meter_reset(meter):
    meter.set_levels(1.5, 0.5, 0.03)
    meter.reset()
    assert meter.fractions == (0.0, 0.0)
    assert meter.at_rest


def test_meter_paints(meter):
    meter.set_levels(0.3, 1.2, 0.03)
    image = meter.grab().toImage()
    assert not image.isNull()
    assert image.width() == 500
