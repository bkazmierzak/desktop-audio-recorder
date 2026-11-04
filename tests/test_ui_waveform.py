"""WaveformView, PlaybackBar and the time-axis labels (offscreen widgets, no audio)."""

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from desktop_audio_recorder.audio.clip import AudioClip
from desktop_audio_recorder.ui.state import AppState, controls_for
from desktop_audio_recorder.ui.transport_bar import (
    PlaybackBar,
    format_position,
    format_selection,
)
from desktop_audio_recorder.ui.waveform_view import WaveformView, format_time_tick

RATE = 8000

# ------------------------------------------------------------------------- axis labels


@pytest.mark.parametrize(
    ("seconds", "spacing", "text"),
    [
        (0.0, 10.0, "00:00"),
        (65.0, 5.0, "01:05"),
        (3599.0, 60.0, "59:59"),
        (3600.0, 600.0, "1:00:00"),
        (3725.0, 60.0, "1:02:05"),
        (1.5, 0.5, "00:01.5"),
        (61.25, 0.25, "01:01.25"),  # as many decimals as the spacing needs
        (0.125, 0.025, "00:00.125"),
        (0.2, 0.2, "00:00.2"),
        (0.0015, 0.0005, "00:00.002"),  # at most milliseconds
        (59.96, 0.1, "01:00.0"),  # rounding carries into the minutes
        (-0.5, 0.5, "-00:00.5"),
        (-0.0, 1.0, "00:00"),
        (float("nan"), 1.0, ""),
    ],
)
def test_format_time_tick(seconds, spacing, text):
    assert format_time_tick(seconds, spacing) == text


def test_format_position_and_selection():
    assert format_position(1.25, 62.0) == "00:01.2 / 01:02.0"
    assert format_selection(None) == "No selection"
    assert format_selection((1.0, 2.5)) == "Selection 00:01.0 – 00:02.5 (1.5 s)"


# ------------------------------------------------------------------------- playback bar


@pytest.fixture
def bar(qtbot):
    b = PlaybackBar()
    qtbot.addWidget(b)
    return b


@pytest.mark.parametrize("state", list(AppState))
def test_playback_bar_follows_state(bar, state):
    bar.set_state(state)
    c = controls_for(state)
    assert bar.play_button.isEnabled() is c.play
    assert bar.play_button.text() == c.play_label
    assert bar.stop_button.isEnabled() is c.stop_playback
    assert bar.loop_button.isEnabled() is c.clip


def test_playback_bar_signals(bar, qtbot):
    bar.set_state(AppState.STOPPED)
    with qtbot.waitSignal(bar.play_clicked, timeout=500):
        qtbot.mouseClick(bar.play_button, Qt.MouseButton.LeftButton)
    bar.set_state(AppState.PLAYING)
    with qtbot.waitSignal(bar.pause_clicked, timeout=500):
        qtbot.mouseClick(bar.play_button, Qt.MouseButton.LeftButton)
    with qtbot.waitSignal(bar.stop_clicked, timeout=500):
        qtbot.mouseClick(bar.stop_button, Qt.MouseButton.LeftButton)
    with qtbot.waitSignal(bar.loop_toggled, timeout=500) as blocker:
        qtbot.mouseClick(bar.loop_button, Qt.MouseButton.LeftButton)
    assert blocker.args == [True]
    assert bar.loop


def test_playback_bar_labels(bar):
    bar.set_position(3.0, 10.0)
    assert bar.position_label.text() == "00:03.0 / 00:10.0"
    bar.set_selection((1.0, 2.0))
    assert bar.selection_label.text() == "Selection 00:01.0 – 00:02.0 (1.0 s)"


# ------------------------------------------------------------------------- waveform view


def make_clip(seconds=10.0):
    t = np.arange(int(seconds * RATE)) / RATE
    x = 0.5 * np.sin(2 * np.pi * 50 * t)
    x[int(4 * RATE) : int(5 * RATE)] *= 2.4  # an over (1.2) between 4 s and 5 s
    return AudioClip(np.column_stack([x, x]).astype(np.float32), RATE)


@pytest.fixture
def view(qtbot):
    v = WaveformView()
    v.resize(800, 240)
    qtbot.addWidget(v)
    v.show()
    qtbot.waitExposed(v)
    v.set_clip(make_clip())
    qtbot.wait(20)
    return v


def viewport_point(view, seconds, y=0.0):
    """Viewport pixel position of (seconds, y) in data coordinates."""
    vb = view._vb
    scene = vb.mapViewToScene(QPointF(seconds, y))
    return view.plot_widget.mapFromScene(scene)


def send_mouse(widget, kind, pos, button, buttons):
    global_pos = widget.mapToGlobal(pos)
    event = QMouseEvent(
        kind, QPointF(pos), QPointF(global_pos), button, buttons, Qt.KeyboardModifier.NoModifier
    )
    QApplication.sendEvent(widget, event)


def drag(view, qtbot, x0, x1, button=Qt.MouseButton.LeftButton):
    vp = view.plot_widget.viewport()
    p0, p1 = viewport_point(view, x0), viewport_point(view, x1)
    send_mouse(vp, QEvent.Type.MouseButtonPress, p0, button, button)
    steps = 8
    for i in range(1, steps + 1):
        p = p0 + (p1 - p0) * (i / steps)
        send_mouse(vp, QEvent.Type.MouseMove, p, Qt.MouseButton.NoButton, button)
        qtbot.wait(5)
    send_mouse(vp, QEvent.Type.MouseButtonRelease, p1, button, Qt.MouseButton.NoButton)
    qtbot.wait(10)


def click(view, seconds):
    vp = view.plot_widget.viewport()
    pos = viewport_point(view, seconds)
    left = Qt.MouseButton.LeftButton
    send_mouse(vp, QEvent.Type.MouseButtonPress, pos, left, left)
    send_mouse(vp, QEvent.Type.MouseButtonRelease, pos, left, Qt.MouseButton.NoButton)


def wheel(view, seconds, delta, modifiers=Qt.KeyboardModifier.NoModifier):
    vp = view.plot_widget.viewport()
    pos = viewport_point(view, seconds)
    event = QWheelEvent(
        QPointF(pos),
        QPointF(vp.mapToGlobal(pos)),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(vp, event)


def test_set_clip_shows_whole_clip(view):
    x0, x1 = view.view_range
    assert x0 == pytest.approx(0.0, abs=1e-6)
    assert x1 == pytest.approx(10.0, abs=1e-6)
    t, y = view.curve.getData()
    width = view._vb.width()
    assert width <= len(t) / 2 <= 2 * width + 1  # one to two bins per pixel column
    assert y.max() == pytest.approx(1.2, abs=1e-3)  # overs are drawn, not clipped
    assert view.playhead.isVisible()
    assert view.playhead_position == 0.0
    assert view.selection is None
    yr = view._vb.viewRange()[1]
    assert yr[1] >= 1.2  # the y range makes room for overs


def test_set_clip_none_clears(view):
    view.set_clip(None)
    t, _ = view.curve.getData()
    assert t is None or len(t) == 0
    assert not view.playhead.isVisible()


def test_click_seeks_and_clears_selection(view, qtbot):
    view.set_selection((1.0, 2.0))
    seeks, selections = [], []
    view.seek_requested.connect(seeks.append)
    view.selection_changed.connect(selections.append)
    click(view, 6.0)
    assert len(seeks) == 1
    assert seeks[0] == pytest.approx(6.0, abs=0.05)
    assert selections == [None]
    assert view.selection is None
    assert not view.region.isVisible()


def test_drag_selects_range(view, qtbot):
    with qtbot.waitSignal(view.selection_changed, timeout=500) as blocker:
        drag(view, qtbot, 2.0, 5.0)
    start, end = blocker.args[0]
    assert start == pytest.approx(2.0, abs=0.05)
    assert end == pytest.approx(5.0, abs=0.05)
    assert view.region.isVisible()
    assert view.selection == (start, end)


def test_drag_backwards_selects_ordered_range(view, qtbot):
    with qtbot.waitSignal(view.selection_changed, timeout=500) as blocker:
        drag(view, qtbot, 7.0, 3.0)
    start, end = blocker.args[0]
    assert start < end
    assert start == pytest.approx(3.0, abs=0.05)


def test_drag_inside_selection_starts_a_new_one(view, qtbot):
    view.set_selection((2.0, 8.0))
    with qtbot.waitSignal(view.selection_changed, timeout=500) as blocker:
        drag(view, qtbot, 4.0, 5.0)
    start, end = blocker.args[0]
    assert start == pytest.approx(4.0, abs=0.05)
    assert end == pytest.approx(5.0, abs=0.05)


def test_selection_is_clamped_to_clip(view):
    view.set_selection((-3.0, 20.0))
    assert view.selection == (0.0, 10.0)
    view.set_selection((4.0, 4.0))
    assert view.selection is None


def test_wheel_zooms_around_pointer(view, qtbot):
    updates = view.envelope_updates
    wheel(view, 2.5, 120 * 3)
    qtbot.wait(20)
    x0, x1 = view.view_range
    assert x1 - x0 < 9.0
    assert x0 < 2.5 < x1
    assert view.envelope_updates > updates  # the envelope follows the zoom
    # Zooming back out stops at the clip bounds.
    for _ in range(10):
        wheel(view, 2.5, -120 * 3)
    qtbot.wait(20)
    x0, x1 = view.view_range
    assert x0 == pytest.approx(0.0, abs=1e-6)
    assert x1 == pytest.approx(10.0, abs=1e-6)


def test_shift_wheel_scrolls(view, qtbot):
    view.set_view_range(2.0, 4.0)
    wheel(view, 3.0, -120, Qt.KeyboardModifier.ShiftModifier)
    qtbot.wait(10)
    x0, x1 = view.view_range
    assert x0 > 2.0
    assert x1 - x0 == pytest.approx(2.0, rel=1e-3)


def test_middle_drag_pans(view, qtbot):
    view.set_view_range(4.0, 6.0)
    drag(view, qtbot, 5.5, 4.5, button=Qt.MouseButton.MiddleButton)
    x0, x1 = view.view_range
    assert x0 == pytest.approx(5.0, abs=0.05)
    assert x1 - x0 == pytest.approx(2.0, rel=1e-3)
    assert view.selection is None


def test_scrollbar_tracks_and_scrolls_view(view, qtbot):
    view.set_view_range(0.0, 2.0)
    qtbot.wait(10)
    sb = view.scrollbar
    assert sb.pageStep() == 2000
    assert sb.maximum() == 8000
    sb.setValue(5000)
    x0, x1 = view.view_range
    assert x0 == pytest.approx(5.0, abs=1e-3)
    assert x1 == pytest.approx(7.0, abs=1e-3)


def test_deep_zoom_shows_samples_and_respects_min_range(view, qtbot):
    view.set_view_range(1.0, 1.0001)  # narrower than the minimum (64 frames)
    qtbot.wait(20)
    x0, x1 = view.view_range
    assert (x1 - x0) * RATE >= 63
    t, _ = view.curve.getData()
    assert len(t) / 2 <= 70  # one bin per frame


def test_set_playhead_follow_pages_view(view, qtbot):
    view.set_view_range(0.0, 2.0)
    view.set_playhead(1.0, follow=True)
    assert view.view_range[0] == pytest.approx(0.0, abs=1e-6)
    view.set_playhead(3.0, follow=True)
    x0, x1 = view.view_range
    assert x0 < 3.0 < x1
    assert x1 - x0 == pytest.approx(2.0, rel=1e-3)
    view.set_playhead(9.9, follow=True)  # near the end: the view stops at the clip end
    assert view.view_range[1] == pytest.approx(10.0, abs=1e-3)
    view.set_playhead(0.5, follow=False)
    assert view.view_range[0] > 1.0
    assert view.playhead_position == pytest.approx(0.5)


def test_set_playhead_does_not_emit_seek(view, qtbot):
    with qtbot.assertNotEmitted(view.seek_requested):
        view.set_playhead(4.0)


def test_dragging_playhead_seeks(view, qtbot):
    view.set_playhead(5.0)
    with qtbot.waitSignal(view.seek_requested, timeout=500) as blocker:
        drag(view, qtbot, 5.0, 3.0)
    assert blocker.args[0] == pytest.approx(3.0, abs=0.05)
    assert view.selection is None  # the drag moved the playhead, not a selection


@pytest.mark.parametrize("span", [60.0, 30.0, 10.0, 3.3, "worst"])
def test_dense_audio_fills_every_pixel_column(qtbot, span):
    """Every bin of loud noise spans at least -0.3..0.3, so that band must be solid at every
    zoom (with too few bins per pixel the zigzag leaves empty columns: moire bands)."""
    rng = np.random.default_rng(5)
    # Long enough to use the pyramid (where bins get grouped); every bin holds enough frames
    # of noise to span the band.
    data = (rng.standard_normal((120 * RATE, 2)) * 0.3).astype(np.float32)
    v = WaveformView()
    v.resize(733, 200)  # an awkward width
    qtbot.addWidget(v)
    v.show()
    qtbot.waitExposed(v)
    v.set_clip(AudioClip(data, RATE))
    if span == "worst":
        # 1% more 512-frame blocks than pixel columns: asking for one bin per pixel would
        # group them in pairs, leaving one bin per two columns.
        span = 1.01 * v._vb.width() * 512 / RATE
    v.set_view_range(41.7, 41.7 + span)
    v.playhead.hide()
    v._guides[2].hide()  # the zero line
    qtbot.wait(30)
    image = v.plot_widget.grab().toImage()
    x0, x1 = v.view_range  # (clamped to the clip)

    def to_pixel(seconds, y):
        return v.plot_widget.mapFromScene(v._vb.mapViewToScene(QPointF(seconds, y)))

    wave = v._wave_color.rgb()
    left, right = to_pixel(x0, 0).x() + 1, to_pixel(x1, 0).x() - 1
    top, bottom = to_pixel(x0, 0.3).y(), to_pixel(x0, -0.3).y()
    painted = sum(
        image.pixelColor(x, y).rgb() == wave
        for x in range(left, right)
        for y in range(top, bottom + 1)
    )
    total = (right - left) * (bottom - top + 1)
    assert painted / total > 0.97


# ------------------------------------------------------------------------- edits


def shorter(clip, seconds):
    return AudioClip(clip.samples[: int(seconds * RATE)].copy(), RATE)


def test_replace_clip_keeps_the_zoomed_view(view):
    view.set_view_range(2.0, 4.0)
    view.set_playhead(3.0)
    view.set_selection((2.5, 3.5))
    edited = shorter(view.clip, 8.0)
    view.replace_clip(edited)
    assert view.clip is edited
    assert view.view_range == pytest.approx((2.0, 4.0), abs=1e-6)
    assert view.playhead_position == pytest.approx(3.0)
    assert view.selection is None  # the window sets the new one
    assert not view.region.isVisible()


def test_replace_clip_shift_and_clamp(view):
    view.set_view_range(6.0, 8.0)
    view.replace_clip(shorter(view.clip, 9.0), shift=-1.0)
    assert view.view_range == pytest.approx((5.0, 7.0), abs=1e-6)
    view.replace_clip(shorter(view.clip, 6.0))  # the view would end past the clip
    assert view.view_range == pytest.approx((4.0, 6.0), abs=1e-6)
    view.set_playhead(5.5)
    view.replace_clip(shorter(view.clip, 1.5))  # shorter than the view
    assert view.view_range == pytest.approx((0.0, 1.5), abs=1e-6)
    assert view.playhead_position == pytest.approx(1.5)


def test_replace_clip_keeps_showing_everything_when_fitted(view):
    longer = AudioClip(np.concatenate([view.clip.samples, view.clip.samples]), RATE)
    view.replace_clip(longer)
    assert view.view_range == pytest.approx((0.0, 20.0), abs=1e-6)
    assert view.scrollbar.maximum() == 0
    even_longer = AudioClip(np.concatenate([longer.samples, view.clip.samples]), RATE)
    view.replace_clip(even_longer, shift=5.0)  # a shift wins: the audio stays put on screen
    assert view.view_range == pytest.approx((5.0, 25.0), abs=1e-6)


def test_replace_clip_updates_curve_and_y_range(view, qtbot):
    quiet = AudioClip(view.clip.samples * 0.1, RATE)
    before = view.envelope_updates
    view.replace_clip(quiet)
    assert view.envelope_updates > before
    _, y = view.curve.getData()
    assert y.max() == pytest.approx(0.12, abs=1e-3)
    assert view._vb.viewRange()[1][1] == pytest.approx(1.05)  # back to the +-1.0 guides
