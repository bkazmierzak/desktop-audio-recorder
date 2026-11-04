"""Recording transport (Record / Pause-Resume / Stop + elapsed time) and the playback bar."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

from desktop_audio_recorder.ui.state import AppState, controls_for
from desktop_audio_recorder.ui.theme import Theme, watch_theme

_ICON_PX = 32


def format_elapsed(seconds: float) -> str:
    """``MM:SS.t`` (or ``H:MM:SS.t`` from one hour), truncated to tenths so it never runs ahead."""
    if not math.isfinite(seconds) or seconds < 0:
        seconds = 0.0
    tenths_total = int(math.floor(seconds * 10 + 1e-6))
    tenths = tenths_total % 10
    total_s = tenths_total // 10
    hours, rem = divmod(total_s, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}.{tenths}"
    return f"{minutes:02d}:{secs:02d}.{tenths}"


def _icon(kind: str, color: QColor) -> QIcon:
    pix = QPixmap(_ICON_PX, _ICON_PX)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    s = _ICON_PX
    if kind == "record":
        p.drawEllipse(QPointF(s / 2, s / 2), s * 0.32, s * 0.32)
    elif kind == "pause":
        w = s * 0.18
        p.drawRect(QRectF(s * 0.26, s * 0.2, w, s * 0.6))
        p.drawRect(QRectF(s * 0.56, s * 0.2, w, s * 0.6))
    elif kind in ("resume", "play"):
        triangle = [QPointF(s * 0.3, s * 0.18), QPointF(s * 0.3, s * 0.82), QPointF(s * 0.8, s / 2)]
        p.drawPolygon(triangle)
    elif kind == "stop":
        p.drawRect(QRectF(s * 0.24, s * 0.24, s * 0.52, s * 0.52))
    elif kind == "loop":
        # Three quarters of a circle with an arrowhead at its open end.
        pen = QPen(color, s * 0.1)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(QRectF(s * 0.2, s * 0.2, s * 0.6, s * 0.6), 90 * 16, 270 * 16)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        head = [QPointF(s * 0.5, s * 0.06), QPointF(s * 0.5, s * 0.34), QPointF(s * 0.72, s * 0.2)]
        p.drawPolygon(head)
    p.end()
    icon = QIcon(pix)
    # A greyed-out variant so disabled buttons read as disabled in both themes.
    disabled = QPixmap(pix.size())
    disabled.fill(Qt.GlobalColor.transparent)
    q = QPainter(disabled)
    q.setOpacity(0.3)
    q.drawPixmap(0, 0, pix)
    q.end()
    icon.addPixmap(disabled, QIcon.Mode.Disabled)
    return icon


# Stick to glyphs that Segoe UI has: a character missing from the UI font (e.g. U+275A "❚")
# makes Qt search the system fonts on first paint, which stalled the GUI thread 0.2-1.3 s.
_STATE_TEXT = {
    AppState.IDLE: "",
    AppState.STARTING: "Starting…",
    AppState.RECORDING: "● REC",
    AppState.PAUSED: "PAUSED",
    AppState.STOPPING: "Finishing…",
    AppState.LOADING: "Loading…",
    AppState.STOPPED: "Stopped",
    AppState.PLAYING: "Playing",
    AppState.EDITING: "Editing…",
    AppState.PROCESSING: "Reducing noise…",
    AppState.EXPORTING: "Exporting…",
}


class TransportBar(QWidget):
    record_clicked = Signal()
    pause_clicked = Signal()
    resume_clicked = Signal()
    stop_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = AppState.IDLE
        self._theme: Theme | None = None
        self._icons = self._make_icons()

        self.record_button = self._button("Record", "record", "Start a new recording")
        self.pause_button = self._button("Pause", "pause", "Pause / resume the recording")
        self.stop_button = self._button("Stop", "stop", "Stop recording")
        self.record_button.clicked.connect(self.record_clicked)
        self.pause_button.clicked.connect(self._on_pause_button)
        self.stop_button.clicked.connect(self.stop_clicked)

        self.elapsed_label = QLabel(format_elapsed(0.0), self)
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        mono.setPointSizeF(max(18.0, self.font().pointSizeF() * 2.2))
        mono.setWeight(QFont.Weight.DemiBold)
        self.elapsed_label.setFont(mono)
        self.elapsed_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.elapsed_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.elapsed_label.setAccessibleName("Elapsed time")
        self.elapsed_label.setToolTip("Recorded time (pauses excluded)")

        self.state_label = QLabel("", self)
        self.state_label.setMinimumWidth(90)
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.record_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.stop_button)
        layout.addStretch(1)
        layout.addWidget(self.state_label)
        layout.addWidget(self.elapsed_label)

        self.set_state(AppState.IDLE)
        watch_theme(self, self._set_theme)

    def _make_icons(self) -> dict[str, QIcon]:
        fg = self.palette().color(self.palette().ColorRole.ButtonText)
        return {
            "record": _icon("record", QColor(220, 45, 45)),
            "pause": _icon("pause", fg),
            "resume": _icon("resume", fg),
            "stop": _icon("stop", fg),
        }

    def _set_theme(self, theme: Theme) -> None:
        if self._theme is not None:  # the icons were drawn for another palette
            self._icons = self._make_icons()
            self.record_button.setIcon(self._icons["record"])
            self.stop_button.setIcon(self._icons["stop"])
        self._theme = theme
        self.set_state(self._state)

    def _button(self, text: str, icon: str, tip: str) -> QPushButton:
        button = QPushButton(self._icons[icon], text, self)
        button.setIconSize(QSize(20, 20))
        button.setMinimumHeight(36)
        button.setMinimumWidth(100)
        button.setToolTip(tip)
        return button

    def _on_pause_button(self) -> None:
        if self._state is AppState.PAUSED:
            self.resume_clicked.emit()
        elif self._state is AppState.RECORDING:
            self.pause_clicked.emit()

    @property
    def state(self) -> AppState:
        return self._state

    def set_state(self, state: AppState) -> None:
        self._state = state
        c = controls_for(state)
        self.record_button.setEnabled(c.record)
        self.pause_button.setEnabled(c.pause)
        self.stop_button.setEnabled(c.stop)
        self.pause_button.setText(c.pause_label)
        self.pause_button.setIcon(self._icons["resume" if state is AppState.PAUSED else "pause"])
        self.state_label.setText(_STATE_TEXT[state])
        style = ""
        if self._theme is not None and state in (AppState.RECORDING, AppState.PAUSED):
            color = self._theme.recording if state is AppState.RECORDING else self._theme.warning
            style = f"color: {color.name()}; font-weight: bold;"
        self.state_label.setStyleSheet(style)

    def set_elapsed(self, seconds: float) -> None:
        self.elapsed_label.setText(format_elapsed(seconds))


def format_position(position: float, duration: float) -> str:
    """``"MM:SS.t / MM:SS.t"`` for the playback position display."""
    return f"{format_elapsed(position)} / {format_elapsed(duration)}"


def format_selection(selection: tuple[float, float] | None) -> str:
    if selection is None:
        return "No selection"
    start, end = selection
    return f"Selection {format_elapsed(start)} – {format_elapsed(end)} ({end - start:.1f} s)"


class PlaybackBar(QWidget):
    """Play/Pause, Stop and Loop for the clip, plus the position and selection read-outs."""

    play_clicked = Signal()
    pause_clicked = Signal()
    stop_clicked = Signal()
    loop_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = AppState.IDLE
        self._icons = self._make_icons()

        self.play_button = self._button("Play", "play", "Play / pause the clip (Space)")
        self.stop_button = self._button("Stop", "stop", "Stop playback and go back to the start")
        self.loop_button = self._button("Loop", "loop", "Repeat the selection (or the whole clip)")
        self.loop_button.setCheckable(True)
        self.play_button.clicked.connect(self._on_play_button)
        self.stop_button.clicked.connect(self.stop_clicked)
        self.loop_button.toggled.connect(self.loop_toggled)

        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        mono.setPointSizeF(max(10.0, self.font().pointSizeF() * 1.15))
        self.position_label = QLabel(format_position(0.0, 0.0), self)
        self.position_label.setFont(mono)
        self.position_label.setAccessibleName("Playback position")
        self.position_label.setToolTip("Playback position / clip length (Home: go to start)")
        self.selection_label = QLabel(format_selection(None), self)
        self.selection_label.setEnabled(False)  # secondary text colour

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.play_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.loop_button)
        layout.addSpacing(12)
        layout.addWidget(self.selection_label)
        layout.addStretch(1)
        layout.addWidget(self.position_label)

        self.set_state(AppState.IDLE)
        self._themed = False
        watch_theme(self, self._set_theme)

    def _make_icons(self) -> dict[str, QIcon]:
        fg = self.palette().color(self.palette().ColorRole.ButtonText)
        return {kind: _icon(kind, fg) for kind in ("play", "pause", "stop", "loop")}

    def _set_theme(self, _theme: Theme) -> None:
        if self._themed:  # the icons were drawn for another palette
            self._icons = self._make_icons()
            self.stop_button.setIcon(self._icons["stop"])
            self.loop_button.setIcon(self._icons["loop"])
            self.set_state(self._state)
        self._themed = True

    def _button(self, text: str, icon: str, tip: str) -> QPushButton:
        button = QPushButton(self._icons[icon], text, self)
        button.setIconSize(QSize(16, 16))
        button.setMinimumHeight(30)
        button.setMinimumWidth(84)
        button.setToolTip(tip)
        return button

    def _on_play_button(self) -> None:
        if self._state is AppState.PLAYING:
            self.pause_clicked.emit()
        elif self._state is AppState.STOPPED:
            self.play_clicked.emit()

    @property
    def state(self) -> AppState:
        return self._state

    @property
    def loop(self) -> bool:
        return self.loop_button.isChecked()

    def set_state(self, state: AppState) -> None:
        self._state = state
        c = controls_for(state)
        self.play_button.setEnabled(c.play)
        self.play_button.setText(c.play_label)
        self.play_button.setIcon(self._icons["pause" if state is AppState.PLAYING else "play"])
        self.stop_button.setEnabled(c.stop_playback)
        self.loop_button.setEnabled(c.clip)

    def set_position(self, position: float, duration: float) -> None:
        self.position_label.setText(format_position(position, duration))

    def set_selection(self, selection: tuple[float, float] | None) -> None:
        self.selection_label.setText(format_selection(selection))
