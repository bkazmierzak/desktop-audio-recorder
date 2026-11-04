"""Stereo peak meter with peak-hold, decay and a clip indicator.

The meter is drawn on a dBFS scale from ``FLOOR_DB`` to 0 dBFS. The recorder works in float, so
peaks above 1.0 (over full scale) are possible; those fill the bar and light the CLIP lamp.

Ballistics are kept in ``MeterChannel`` (plain Python, no Qt) so they can be unit-tested with an
explicit time step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from desktop_audio_recorder.ui.theme import Theme, watch_theme

FLOOR_DB = -60.0
DECAY_DB_PER_S = 24.0  # bar fall-back speed
PEAK_HOLD_S = 1.5  # how long the peak-hold tick stays put before falling
CLIP_HOLD_S = 2.0  # how long the CLIP lamp stays lit after the last over
CLIP_THRESHOLD = 1.0  # peaks strictly above this are "over full scale"
SCALE_MARKS_DB = (-48, -36, -24, -18, -12, -6, -3, 0)
_YELLOW_DB = -12.0
_RED_DB = -3.0


def peak_to_db(peak: float, floor_db: float = FLOOR_DB) -> float:
    """Linear peak (1.0 = full scale) to dBFS, clamped below at ``floor_db``."""
    if not math.isfinite(peak) or peak <= 0.0:
        return floor_db
    return max(floor_db, 20.0 * math.log10(peak))


def db_to_fraction(db: float, floor_db: float = FLOOR_DB) -> float:
    """dBFS to the filled fraction of the bar (``floor_db`` -> 0.0, 0 dBFS and above -> 1.0)."""
    return min(1.0, max(0.0, (db - floor_db) / -floor_db))


def level_to_fraction(peak: float, floor_db: float = FLOOR_DB) -> float:
    """Linear peak to the filled fraction of the bar."""
    return db_to_fraction(peak_to_db(peak, floor_db), floor_db)


@dataclass
class MeterChannel:
    """Ballistics for one channel: instant attack, linear-in-dB decay, peak hold, clip latch."""

    floor_db: float = FLOOR_DB
    decay_db_per_s: float = DECAY_DB_PER_S
    peak_hold_s: float = PEAK_HOLD_S
    clip_hold_s: float = CLIP_HOLD_S

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.db = self.floor_db  # displayed bar level
        self.hold_db = self.floor_db  # peak-hold tick
        self._hold_age = 0.0
        self._clip_age: float | None = None  # seconds since the last over; None = not clipped
        self.raw_peak = 0.0

    @property
    def fraction(self) -> float:
        return db_to_fraction(self.db, self.floor_db)

    @property
    def hold_fraction(self) -> float:
        return db_to_fraction(self.hold_db, self.floor_db)

    @property
    def clipped(self) -> bool:
        return self._clip_age is not None

    @property
    def at_rest(self) -> bool:
        return self.db <= self.floor_db and self.hold_db <= self.floor_db and not self.clipped

    def clear_clip(self) -> None:
        self._clip_age = None

    def update(self, peak: float, dt: float) -> None:
        """Feed the newest block peak (linear) after ``dt`` seconds."""
        dt = max(0.0, dt)
        self.raw_peak = peak
        new_db = peak_to_db(peak, self.floor_db)
        fallen = self.db - self.decay_db_per_s * dt
        self.db = max(new_db, fallen, self.floor_db)

        if new_db >= self.hold_db:
            self.hold_db = new_db
            self._hold_age = 0.0
        else:
            self._hold_age += dt
            if self._hold_age > self.peak_hold_s:
                over = min(dt, self._hold_age - self.peak_hold_s)
                self.hold_db = max(self.db, self.hold_db - self.decay_db_per_s * over)

        if math.isfinite(peak) and peak > CLIP_THRESHOLD:
            self._clip_age = 0.0
        elif self._clip_age is not None:
            self._clip_age += dt
            if self._clip_age > self.clip_hold_s:
                self._clip_age = None


class LevelMeter(QWidget):
    """Horizontal L/R peak meter. Feed it with ``set_levels(left, right, dt)``.

    Clicking the meter clears the CLIP lamps.
    """

    clip_cleared = Signal()

    _LABEL_W = 16
    _CLIP_W = 40
    _BAR_H = 12
    _GAP = 4
    _SCALE_H = 14

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.channels = (MeterChannel(), MeterChannel())
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(200)
        self.setToolTip("Peak level (dBFS). CLIP lights when the signal goes over full scale.")
        self.setAccessibleName("Level meter")
        self.theme: Theme
        watch_theme(self, self._set_theme)

    def _set_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.update()

    # ------------------------------------------------------------------ API

    def set_levels(self, left: float, right: float, dt: float) -> None:
        self.channels[0].update(left, dt)
        self.channels[1].update(right, dt)
        self.update()

    def reset(self) -> None:
        for ch in self.channels:
            ch.reset()
        self.update()

    def clear_clip(self) -> None:
        for ch in self.channels:
            ch.clear_clip()
        self.update()
        self.clip_cleared.emit()

    @property
    def fractions(self) -> tuple[float, float]:
        return (self.channels[0].fraction, self.channels[1].fraction)

    @property
    def clipped(self) -> tuple[bool, bool]:
        return (self.channels[0].clipped, self.channels[1].clipped)

    @property
    def at_rest(self) -> bool:
        return all(ch.at_rest for ch in self.channels)

    # ------------------------------------------------------------------ Qt

    def sizeHint(self) -> QSize:
        return QSize(480, 2 * self._BAR_H + self._GAP + self._SCALE_H + 10)

    def minimumSizeHint(self) -> QSize:
        return QSize(200, self.sizeHint().height())

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clear_clip()
        super().mousePressEvent(event)

    def _bar_rect(self, row: int) -> QRectF:
        x = self._LABEL_W
        w = self.width() - self._LABEL_W - self._CLIP_W - self._GAP
        y = 2 + row * (self._BAR_H + self._GAP)
        return QRectF(x, y, max(1, w), self._BAR_H)

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        pal = self.palette()
        text_color = pal.color(pal.ColorRole.WindowText)
        font = QFont(self.font())
        font.setPointSizeF(max(6.5, font.pointSizeF() * 0.8))
        # ClearType gives small labels on the trough coloured fringes.
        font.setStyleStrategy(QFont.StyleStrategy.NoSubpixelAntialias)
        p.setFont(font)

        theme = self.theme
        trough = theme.meter_trough
        for row, (label, ch) in enumerate(zip(("L", "R"), self.channels, strict=True)):
            rect = self._bar_rect(row)
            p.setPen(text_color)
            p.drawText(
                QRectF(0, rect.y(), self._LABEL_W - 4, rect.height()),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            p.fillRect(rect, trough)

            grad = QLinearGradient(rect.left(), 0, rect.right(), 0)
            yellow_at = db_to_fraction(_YELLOW_DB)
            red_at = db_to_fraction(_RED_DB)
            grad.setColorAt(0.0, QColor(30, 170, 80))
            grad.setColorAt(yellow_at, QColor(80, 200, 80))
            grad.setColorAt(min(1.0, yellow_at + 0.001), QColor(225, 200, 40))
            grad.setColorAt(red_at, QColor(235, 170, 40))
            grad.setColorAt(min(1.0, red_at + 0.001), QColor(225, 50, 40))
            grad.setColorAt(1.0, QColor(225, 50, 40))
            filled = QRectF(rect.x(), rect.y(), rect.width() * ch.fraction, rect.height())
            p.fillRect(filled, grad)

            if ch.hold_db > ch.floor_db:
                hx = rect.x() + rect.width() * ch.hold_fraction
                hx = min(hx, rect.right() - 1)
                p.fillRect(QRectF(hx - 1, rect.y(), 2, rect.height()), theme.meter_peak_tick)

            clip_rect = QRectF(rect.right() + self._GAP, rect.y(), self._CLIP_W, rect.height())
            p.fillRect(clip_rect, QColor(230, 30, 30) if ch.clipped else trough)
            p.setPen(QColor(255, 255, 255) if ch.clipped else theme.meter_clip_off_text)
            p.drawText(clip_rect, Qt.AlignmentFlag.AlignCenter, "CLIP")

        # dB scale under the bars.
        top = self._bar_rect(1).bottom() + 2
        p.setPen(QPen(text_color))
        bar = self._bar_rect(0)
        for mark in SCALE_MARKS_DB:
            x = bar.x() + bar.width() * db_to_fraction(mark)
            p.drawLine(int(x), int(top), int(x), int(top) + 3)
            text = "0" if mark == 0 else str(mark)
            p.drawText(
                QRectF(x - 20, top + 2, 40, self._SCALE_H),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                text,
            )
        p.end()
