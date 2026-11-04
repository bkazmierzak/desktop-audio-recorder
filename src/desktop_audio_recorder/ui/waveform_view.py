"""Waveform display (pyqtgraph): envelope, selection, playhead, zoom and scroll.

Mouse:

* wheel: zoom the time axis around the pointer (Ctrl+wheel too)
* Shift+wheel or horizontal wheel: scroll; the scroll bar below also scrolls
* left click: seek there (and clear the selection)
* left drag: select a range; drag the selection edges to adjust it
* middle/right drag: pan
* drag the playhead: seek

The curve is a min/max envelope from ``EnvelopePyramid``, recomputed for the visible range
whenever it changes (coalesced to once per event-loop pass). It asks for two bins per pixel
column: a query returns between half and all of the bins asked for, so every column gets at
least one min-to-max stroke (with fewer, the zigzag leaves empty columns that show as moiré
bands on dense audio).
"""

from __future__ import annotations

import math

import pyqtgraph as pg
from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPalette, QPen
from PySide6.QtWidgets import QScrollBar, QVBoxLayout, QWidget

from desktop_audio_recorder.audio.clip import AudioClip
from desktop_audio_recorder.audio.envelope import EnvelopePyramid
from desktop_audio_recorder.ui.theme import Theme, watch_theme

MIN_VIEW_FRAMES = 64  # deepest zoom: this many frames across the view
BINS_PER_PIXEL = 2
MAX_BINS = 8000
MIN_BINS = 400
SCROLL_STEP = 0.1  # fraction of the view width per wheel notch
FOLLOW_MARGIN = 0.02  # when paging to follow the playhead, keep it this far from the left edge
_SCROLL_UNITS_PER_SECOND = 1000  # the scroll bar works in milliseconds


def format_time_tick(seconds: float, spacing: float) -> str:
    """Axis label: ``MM:SS`` (``H:MM:SS`` from one hour) with decimals for sub-second ticks."""
    if not math.isfinite(seconds):
        return ""
    # Fewest decimals (up to milliseconds) that show the tick spacing exactly: 0.5 -> 1, 0.25 -> 2.
    decimals = 0
    while decimals < 3 and abs(spacing * 10**decimals - round(spacing * 10**decimals)) > 1e-6:
        decimals += 1
    scale = 10**decimals
    units = round(abs(seconds) * scale)
    sign = "-" if seconds < 0 and units else ""
    whole, frac = divmod(units, scale)
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    text = f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"
    if decimals:
        text += f".{frac:0{decimals}d}"
    return sign + text


class TimeAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):  # noqa: N802 (pyqtgraph API)
        return [format_time_tick(v * scale, spacing * scale) for v in values]


class _WaveViewBox(pg.ViewBox):
    """ViewBox with editor-style mouse handling (see module docstring)."""

    clicked = Signal(float)  # x (seconds)
    select_dragged = Signal(float, float, bool)  # anchor x, current x, finished

    def __init__(self) -> None:
        super().__init__(enableMenu=False)
        self.setMouseEnabled(x=True, y=False)

    def mouseClickEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            self.clicked.emit(self.mapToView(ev.pos()).x())
        else:
            ev.ignore()

    def mouseDragEvent(self, ev, axis=None) -> None:  # noqa: N802
        button = ev.button()
        if button == Qt.MouseButton.LeftButton:
            ev.accept()
            anchor = self.mapToView(ev.buttonDownPos()).x()
            current = self.mapToView(ev.pos()).x()
            self.select_dragged.emit(anchor, current, ev.isFinish())
        elif button in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            ev.accept()
            dx = self.mapToView(ev.lastPos()).x() - self.mapToView(ev.pos()).x()
            self.translateBy(x=dx)
        else:
            ev.ignore()

    def wheelEvent(self, ev, axis=None) -> None:  # noqa: N802
        horizontal = ev.orientation() == Qt.Orientation.Horizontal
        if horizontal or ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            ev.accept()
            notches = ev.delta() / 120
            x0, x1 = self.viewRange()[0]
            self.translateBy(x=-notches * SCROLL_STEP * (x1 - x0))
        else:
            super().wheelEvent(ev, axis=0)  # zoom x around the pointer


class _SelectionRegion(pg.LinearRegionItem):
    """Selection whose edges can be dragged; dragging its body starts a new selection instead
    of moving it (the drag falls through to the ViewBox)."""

    def mouseDragEvent(self, ev) -> None:  # noqa: N802
        ev.ignore()

    def mouseClickEvent(self, ev) -> None:  # noqa: N802
        ev.ignore()

    def hoverEvent(self, ev) -> None:  # noqa: N802
        pass  # no body highlight: the body isn't draggable


class WaveformView(QWidget):
    seek_requested = Signal(float)  # seconds
    selection_changed = Signal(object)  # (start, end) seconds, or None

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._clip: AudioClip | None = None
        self._pyramid: EnvelopePyramid | None = None
        self._selection: tuple[float, float] | None = None
        self._dragging_playhead = False
        self._syncing_scrollbar = False
        self.follow_playhead = True
        self.envelope_updates = 0  # for tests / diagnostics

        self._vb = _WaveViewBox()
        self.axis = TimeAxis("bottom")
        self.plot_widget = pg.PlotWidget(self, viewBox=self._vb, axisItems={"bottom": self.axis})
        self.plot_widget.setFrameShape(self.plot_widget.Shape.NoFrame)
        plot = self.plot_widget.getPlotItem()
        plot.hideAxis("left")
        plot.hideButtons()
        plot.setMenuEnabled(False)
        plot.layout.setContentsMargins(0, 4, 0, 0)

        self._guides = [pg.InfiniteLine(y, angle=0, movable=False) for y in (-1.0, 1.0, 0.0)]
        for line in self._guides:
            line.setZValue(-10)
            self._vb.addItem(line, ignoreBounds=True)

        self.curve = pg.PlotDataItem(antialias=False, skipFiniteCheck=True, connect="all")
        self._vb.addItem(self.curve)

        self.region = _SelectionRegion(
            values=(0, 0),
            orientation="vertical",
            movable=True,
        )
        self.region.setZValue(5)
        self.region.hide()
        self._vb.addItem(self.region, ignoreBounds=True)

        playhead_pen = QPen(QColor(255, 120, 40))
        playhead_pen.setWidth(2)
        playhead_pen.setCosmetic(True)
        self.playhead = pg.InfiniteLine(
            0.0, angle=90, pen=playhead_pen, movable=True, hoverPen=playhead_pen
        )
        self.playhead.setZValue(10)
        self._vb.addItem(self.playhead, ignoreBounds=True)

        self.scrollbar = QScrollBar(Qt.Orientation.Horizontal, self)
        self.scrollbar.valueChanged.connect(self._on_scrollbar)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.plot_widget, 1)
        layout.addWidget(self.scrollbar)

        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(0)
        self._update_timer.timeout.connect(self.update_envelope)

        self._vb.sigXRangeChanged.connect(self._on_range_changed)
        self._vb.sigResized.connect(self._schedule_update)
        self._vb.clicked.connect(self._on_clicked)
        self._vb.select_dragged.connect(self._on_select_dragged)
        self.region.sigRegionChangeFinished.connect(self._on_region_edited)
        self.playhead.sigDragged.connect(self._on_playhead_dragged)
        self.playhead.sigPositionChangeFinished.connect(self._on_playhead_released)

        self._wave_color = QColor()
        watch_theme(self, self._set_theme)
        self.set_clip(None)

    def _set_theme(self, theme: Theme) -> None:
        """Colours for the light or dark palette (at start and when Windows switches)."""
        self._wave_color = QColor(theme.wave)
        self.plot_widget.setBackground(self.palette().color(QPalette.ColorRole.Base))
        guide = QColor(theme.ink)
        guide.setAlpha(50)
        for line in self._guides:
            pen = QPen(guide)
            pen.setCosmetic(True)
            if line.value() != 0.0:
                pen.setStyle(Qt.PenStyle.DashLine)  # the full-scale lines
            line.setPen(pen)
        wave_pen = QPen(self._wave_color)
        wave_pen.setCosmetic(True)
        wave_pen.setWidth(1)
        self.curve.setPen(wave_pen)
        # Neutral selection (not the accent colour, which may be as orange as the playhead).
        ink = theme.ink
        fill = QColor(ink)
        fill.setAlpha(45 if theme.dark else 35)
        self.region.setBrush(fill)
        edge = QPen(QColor(ink.red(), ink.green(), ink.blue(), 150))
        edge.setCosmetic(True)
        hover = QPen(ink)
        hover.setCosmetic(True)
        hover.setWidth(2)
        for line in self.region.lines:
            line.setPen(edge)
            line.setHoverPen(hover)
        self.axis.setPen(self.palette().color(QPalette.ColorRole.WindowText))
        self.axis.setTextPen(self.palette().color(QPalette.ColorRole.WindowText))

    # ------------------------------------------------------------------ clip

    @property
    def clip(self) -> AudioClip | None:
        return self._clip

    def set_clip(self, clip: AudioClip | None, pyramid: EnvelopePyramid | None = None) -> None:
        """Show ``clip`` zoomed out to its full length (``pyramid`` is built if not given)."""
        self._clip = clip
        self._selection = None
        self.region.hide()
        self._dragging_playhead = False
        if clip is None or clip.frames == 0:
            self._pyramid = None
            self.curve.setData([], [])
            self.playhead.hide()
            self.scrollbar.setEnabled(False)
            self.setEnabled(clip is not None)
            return
        self._configure(clip, pyramid)
        with QSignalBlocker(self.playhead):
            self.playhead.setValue(0.0)
        self._vb.setXRange(0.0, clip.duration, padding=0)
        self.update_envelope()

    def replace_clip(
        self, clip: AudioClip, pyramid: EnvelopePyramid | None = None, *, shift: float = 0.0
    ) -> None:
        """Show an edited version of the clip without resetting the view.

        The visible range moves by ``shift`` seconds (so the audio that stays doesn't move on
        screen) and is clamped to the new length; the zoom is kept unless the clip got shorter
        than the view. Without a shift, a view that showed the whole clip shows the whole new
        clip. The selection is cleared (the caller sets the new one)
        and the playhead stays where it is, clamped.
        """
        if self._clip is None or clip.frames == 0:
            self.set_clip(clip, pyramid)
            return
        x0, x1 = self.view_range
        tolerance = 0.5 / self._clip.sample_rate
        fitted = x0 <= tolerance and x1 >= self._clip.duration - tolerance
        playhead = self.playhead_position
        self._clip = clip
        self._selection = None
        self.region.hide()
        self._configure(clip, pyramid)
        duration = clip.duration
        if fitted and not shift:  # the whole clip was in view: keep showing all of it
            x0, x1 = 0.0, duration
        width = min(x1 - x0, duration)
        start = min(max(0.0, x0 + shift), duration - width)
        with QSignalBlocker(self.playhead):
            self.playhead.setValue(min(max(playhead, 0.0), duration))
        self._vb.setXRange(start, start + width, padding=0)
        self._sync_scrollbar()
        self.update_envelope()

    def _configure(self, clip: AudioClip, pyramid: EnvelopePyramid | None) -> None:
        """Limits, y range and bounds for a (non-empty) clip."""
        self._pyramid = pyramid if pyramid is not None else EnvelopePyramid(clip.samples)
        duration = clip.duration
        y = max(1.0, self._pyramid.peak) * 1.05  # overs get room; the pyramid avoids a scan
        self._vb.setLimits(
            xMin=0.0,
            xMax=duration,
            minXRange=min(duration, MIN_VIEW_FRAMES / clip.sample_rate),
            maxXRange=duration,
            yMin=-y,
            yMax=y,
        )
        self._vb.setYRange(-y, y, padding=0)
        self.region.setBounds((0.0, duration))
        self.playhead.setBounds((0.0, duration))
        self.playhead.show()
        self.scrollbar.setEnabled(True)
        self.setEnabled(True)

    # ------------------------------------------------------------------ view range

    @property
    def view_range(self) -> tuple[float, float]:
        x0, x1 = self._vb.viewRange()[0]
        return float(x0), float(x1)

    def set_view_range(self, start: float, end: float) -> None:
        self._vb.setXRange(start, end, padding=0)

    def zoom(self, factor: float, center: float | None = None) -> None:
        """Zoom in (``factor`` < 1) or out (> 1) around ``center`` (default: view centre)."""
        x0, x1 = self.view_range
        c = (x0 + x1) / 2 if center is None else center
        self.set_view_range(c - (c - x0) * factor, c + (x1 - c) * factor)

    def zoom_to_fit(self) -> None:
        if self._clip is not None:
            self.set_view_range(0.0, self._clip.duration)

    def _bins(self) -> int:
        width = self._vb.width() * self.devicePixelRatioF()
        return int(min(MAX_BINS, max(MIN_BINS, BINS_PER_PIXEL * width)))

    def _schedule_update(self) -> None:
        self._update_timer.start()

    def _on_range_changed(self) -> None:
        self._schedule_update()
        self._sync_scrollbar()

    def update_envelope(self) -> None:
        """Recompute the curve for the visible range."""
        clip, pyramid = self._clip, self._pyramid
        if clip is None or pyramid is None:
            return
        x0, x1 = self.view_range
        start = max(0, math.floor(x0 * clip.sample_rate))
        stop = min(clip.frames, math.ceil(x1 * clip.sample_rate) + 1)
        env = pyramid.envelope(start, stop, self._bins())
        t, y = env.zigzag(clip.sample_rate)
        self.curve.setData(t, y)
        self.envelope_updates += 1

    def _sync_scrollbar(self) -> None:
        clip = self._clip
        if clip is None or self._syncing_scrollbar:
            return
        x0, x1 = self.view_range
        page = max(1, round((x1 - x0) * _SCROLL_UNITS_PER_SECOND))
        total = round(clip.duration * _SCROLL_UNITS_PER_SECOND)
        with QSignalBlocker(self.scrollbar):
            self.scrollbar.setRange(0, max(0, total - page))
            self.scrollbar.setPageStep(page)
            self.scrollbar.setSingleStep(max(1, page // 20))
            self.scrollbar.setValue(round(x0 * _SCROLL_UNITS_PER_SECOND))

    def _on_scrollbar(self, value: int) -> None:
        if self._clip is None:
            return
        x0, x1 = self.view_range
        width = x1 - x0
        start = value / _SCROLL_UNITS_PER_SECOND
        self._syncing_scrollbar = True
        try:
            self.set_view_range(start, start + width)
        finally:
            self._syncing_scrollbar = False

    # ------------------------------------------------------------------ playhead

    @property
    def playhead_position(self) -> float:
        return float(self.playhead.value())

    def set_playhead(self, seconds: float, *, follow: bool = False) -> None:
        """Move the playhead (no ``seek_requested``). With ``follow``, page the view so it stays
        visible."""
        if self._clip is None or self._dragging_playhead:
            return
        with QSignalBlocker(self.playhead):
            self.playhead.setValue(seconds)
        if follow and self.follow_playhead:
            x0, x1 = self.view_range
            width = x1 - x0
            if width < self._clip.duration and not x0 <= seconds <= x1:
                start = seconds - FOLLOW_MARGIN * width
                start = min(max(0.0, start), self._clip.duration - width)
                self.set_view_range(start, start + width)

    def _on_playhead_dragged(self) -> None:
        self._dragging_playhead = True

    def _on_playhead_released(self) -> None:
        self._dragging_playhead = False
        self.seek_requested.emit(self.playhead_position)

    # ------------------------------------------------------------------ selection

    @property
    def selection(self) -> tuple[float, float] | None:
        return self._selection

    def set_selection(self, selection: tuple[float, float] | None, *, emit: bool = False) -> None:
        clip = self._clip
        if selection is not None and clip is not None:
            a, b = sorted(clip.clamp_seconds(v) for v in selection)
            selection = (a, b) if b > a else None
        else:
            selection = None
        changed = selection != self._selection
        self._selection = selection
        with QSignalBlocker(self.region):
            if selection is None:
                self.region.hide()
            else:
                self.region.setRegion(selection)
                self.region.show()
        if emit and changed:
            self.selection_changed.emit(selection)

    def clear_selection(self) -> None:
        self.set_selection(None, emit=True)

    def _on_clicked(self, x: float) -> None:
        if self._clip is None:
            return
        self.set_selection(None, emit=True)
        self.seek_requested.emit(self._clip.clamp_seconds(x))

    def _on_select_dragged(self, anchor: float, current: float, finished: bool) -> None:
        if self._clip is None:
            return
        self.set_selection((anchor, current))
        if finished:
            self.selection_changed.emit(self._selection)

    def _on_region_edited(self) -> None:
        if self._clip is None or not self.region.isVisible():
            return
        a, b = self.region.getRegion()
        self.set_selection((a, b), emit=True)
