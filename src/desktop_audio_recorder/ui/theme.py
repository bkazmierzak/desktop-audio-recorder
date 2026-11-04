"""Light and dark colours for the widgets that pick their own colours.

Most of the window uses the style's palette and follows the Windows light/dark setting by
itself. A few things use fixed colours (the level meter, the waveform, the red and amber status
texts, the warning banner); ``theme_for(palette)`` gives the set that suits the palette.

``watch_theme(widget, apply)`` calls ``apply(theme)`` now and again whenever the widget's
palette switches between light and dark. That's the widget's ``PaletteChange`` event, not
``QStyleHints.colorSchemeChanged``: Qt emits the signal before it updates the palette, so a
handler of the signal would still read the old colours. Watch a widget that has no style sheet
of its own (the dialog, not its coloured label): a widget with a style sheet keeps its palette
when its parent's changes, so it gets no ``PaletteChange``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QEvent, QObject
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QWidget


@dataclass(frozen=True)
class Theme:
    dark: bool
    error: QColor  # error text: status bar errors, "no playback device", dialog messages
    warning: QColor  # warning text (export dialog) and the PAUSED label
    recording: QColor  # the "● REC" label
    banner_background: QColor  # the warning banner above the transport
    banner_border: QColor
    banner_text: QColor
    meter_trough: QColor  # the level meter's empty bar and unlit CLIP lamp
    meter_peak_tick: QColor
    meter_clip_off_text: QColor
    wave: QColor  # the waveform
    ink: QColor  # guides and the selection in the waveform (drawn with alpha)

    @property
    def error_style(self) -> str:
        return f"color: {_css(self.error)};"

    @property
    def warning_style(self) -> str:
        return f"color: {_css(self.warning)};"

    @property
    def status_error_style(self) -> str:
        return f"QStatusBar {{ color: {_css(self.error)}; }}"

    @property
    def banner_style(self) -> str:
        return (
            f"QLabel {{ background: {_css(self.banner_background)};"
            f" color: {_css(self.banner_text)};"
            f" border: 1px solid {_css(self.banner_border)};"
            " border-radius: 4px; padding: 6px 10px; }"
        )


DARK = Theme(
    dark=True,
    error=QColor(250, 110, 100),
    warning=QColor(230, 160, 40),
    recording=QColor(250, 110, 100),
    banner_background=QColor(255, 196, 0),
    banner_border=QColor(255, 196, 0),
    banner_text=QColor(30, 30, 30),
    meter_trough=QColor(32, 32, 36),
    meter_peak_tick=QColor(240, 240, 240),
    meter_clip_off_text=QColor(110, 110, 115),
    wave=QColor(90, 170, 255),
    ink=QColor(255, 255, 255),
)
# Each text colour has at least 4.5:1 contrast on its window colour (WCAG AA; tests check it).
LIGHT = Theme(
    dark=False,
    error=QColor(190, 30, 25),
    warning=QColor(150, 90, 0),
    recording=QColor(190, 30, 25),
    banner_background=QColor(255, 232, 150),
    banner_border=QColor(225, 170, 0),
    banner_text=QColor(40, 30, 0),
    meter_trough=QColor(214, 214, 219),
    meter_peak_tick=QColor(40, 40, 44),
    meter_clip_off_text=QColor(120, 120, 126),
    wave=QColor(25, 100, 190),
    ink=QColor(0, 0, 0),
)


def _css(color: QColor) -> str:
    return f"rgb({color.red()}, {color.green()}, {color.blue()})"


def is_dark(palette: QPalette) -> bool:
    """A dark palette: the window colour is closer to black than to white (the active one, so
    a disabled widget reads the same)."""
    return palette.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Window).lightness() < 128


def theme_for(palette: QPalette) -> Theme:
    return DARK if is_dark(palette) else LIGHT


def contrast_ratio(a: QColor, b: QColor) -> float:
    """WCAG contrast ratio of two colours (1 to 21)."""

    def luminance(c: QColor) -> float:
        def channel(v: int) -> float:
            s = v / 255
            return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4

        return 0.2126 * channel(c.red()) + 0.7152 * channel(c.green()) + 0.0722 * channel(c.blue())

    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


class _ThemeWatch(QObject):
    def __init__(self, widget: QWidget, apply: Callable[[Theme], None]) -> None:
        super().__init__(widget)
        self._apply = apply
        self._dark: bool | None = None
        widget.installEventFilter(self)
        self.check(widget)

    def check(self, widget: QWidget) -> None:
        dark = is_dark(widget.palette())
        if dark != self._dark:  # also stops a stylesheet set by apply() from looping
            self._dark = dark
            self._apply(DARK if dark else LIGHT)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.PaletteChange and isinstance(watched, QWidget):
            self.check(watched)
        return False


def watch_theme(widget: QWidget, apply: Callable[[Theme], None]) -> None:
    """Call ``apply(theme)`` now, and again each time ``widget``'s palette turns light or dark."""
    _ThemeWatch(widget, apply)
