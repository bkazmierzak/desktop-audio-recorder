"""Light and dark colours (ui/theme.py) and the widgets that switch with the palette."""

from pathlib import Path

import pytest
from conftest import FakeBackend, FakeDeviceManager, FakeRecorder
from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QWidget

from desktop_audio_recorder.audio.player import ClipPlayer
from desktop_audio_recorder.export.writers import ExportSettings
from desktop_audio_recorder.settings import Settings
from desktop_audio_recorder.ui.export_dialog import ExportDialog
from desktop_audio_recorder.ui.level_meter import LevelMeter
from desktop_audio_recorder.ui.main_window import MainWindow
from desktop_audio_recorder.ui.state import AppState
from desktop_audio_recorder.ui.theme import (
    DARK,
    LIGHT,
    contrast_ratio,
    is_dark,
    theme_for,
    watch_theme,
)
from desktop_audio_recorder.ui.transport_bar import PlaybackBar, TransportBar
from desktop_audio_recorder.ui.waveform_view import WaveformView

# The window colours of Qt's windows11 style (and two neighbours dialogs use).
LIGHT_WINDOWS = (QColor(243, 243, 243), QColor(255, 255, 255), QColor(239, 239, 239))
DARK_WINDOWS = (QColor(30, 30, 30), QColor(32, 32, 32), QColor(43, 43, 43))


def palette(dark: bool) -> QPalette:
    pal = QPalette()
    window, base, text = (
        (QColor(32, 32, 32), QColor(25, 25, 25), QColor(255, 255, 255))
        if dark
        else (QColor(243, 243, 243), QColor(255, 255, 255), QColor(0, 0, 0))
    )
    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
        pal.setColor(group, QPalette.ColorRole.Window, window)
        pal.setColor(group, QPalette.ColorRole.Base, base)
        pal.setColor(group, QPalette.ColorRole.WindowText, text)
        pal.setColor(group, QPalette.ColorRole.ButtonText, text)
    return pal


# ---------------------------------------------------------------------- colour choice


def test_is_dark_follows_the_window_colour():
    assert is_dark(palette(True))
    assert not is_dark(palette(False))
    assert theme_for(palette(True)) is DARK
    assert theme_for(palette(False)) is LIGHT


@pytest.mark.parametrize(
    ("theme", "windows"), [(LIGHT, LIGHT_WINDOWS), (DARK, DARK_WINDOWS)], ids=["light", "dark"]
)
@pytest.mark.parametrize("role", ["error", "warning", "recording"])
def test_text_colours_have_aa_contrast(theme, windows, role):
    for window in windows:
        assert contrast_ratio(getattr(theme, role), window) >= 4.5, window.name()


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=["light", "dark"])
def test_banner_and_meter_contrast(theme):
    assert contrast_ratio(theme.banner_text, theme.banner_background) >= 7
    assert contrast_ratio(theme.meter_peak_tick, theme.meter_trough) >= 4.5
    # The unlit CLIP label is meant to be faint, but still readable.
    assert contrast_ratio(theme.meter_clip_off_text, theme.meter_trough) >= 2


def test_the_light_meter_trough_is_light_and_the_dark_one_dark():
    assert LIGHT.meter_trough.lightness() > 180
    assert DARK.meter_trough.lightness() < 60


def test_contrast_ratio_extremes():
    assert contrast_ratio(QColor(0, 0, 0), QColor(255, 255, 255)) == pytest.approx(21)
    assert contrast_ratio(QColor(90, 90, 90), QColor(90, 90, 90)) == pytest.approx(1)


def test_styles_carry_the_colours():
    assert "rgb(190, 30, 25)" in LIGHT.status_error_style
    assert LIGHT.error_style == "color: rgb(190, 30, 25);"
    assert "rgb(250, 110, 100)" in DARK.error_style
    assert "background: rgb(255, 232, 150)" in LIGHT.banner_style
    assert "background: rgb(255, 196, 0)" in DARK.banner_style


# ---------------------------------------------------------------------- watch_theme


def test_watch_theme_calls_now_and_on_each_switch(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.setPalette(palette(False))
    seen = []
    watch_theme(widget, seen.append)
    assert seen == [LIGHT]
    widget.setPalette(palette(True))
    assert seen == [LIGHT, DARK]
    darker = palette(True)
    darker.setColor(QPalette.ColorRole.Window, QColor(10, 10, 10))
    widget.setPalette(darker)  # still dark: no call
    assert seen == [LIGHT, DARK]
    widget.setPalette(palette(False))
    assert seen == [LIGHT, DARK, LIGHT]


def test_a_stylesheet_set_by_the_callback_does_not_loop(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    calls = []

    def apply(theme):
        calls.append(theme)
        widget.setStyleSheet(theme.error_style)

    watch_theme(widget, apply)
    widget.setPalette(palette(True))
    widget.setPalette(palette(False))
    assert len(calls) <= 3  # start (offscreen: light), dark, light


# ---------------------------------------------------------------------- widgets


def test_level_meter_paints_the_trough_for_the_palette(qtbot):
    meter = LevelMeter()
    qtbot.addWidget(meter)
    meter.resize(300, meter.sizeHint().height())
    for dark, theme in ((True, DARK), (False, LIGHT)):
        meter.setPalette(palette(dark))
        assert meter.theme is theme
        image = meter.grab().toImage()
        rect = meter._bar_rect(0)
        pixel = image.pixelColor(int(rect.center().x()), int(rect.center().y()))
        assert pixel.rgb() == theme.meter_trough.rgb()


def test_level_meter_peak_tick_follows_the_theme(qtbot):
    meter = LevelMeter()
    qtbot.addWidget(meter)
    meter.resize(300, meter.sizeHint().height())
    meter.set_levels(0.1, 0.1, 0.0)  # -20 dB: the bar and the hold tick sit at 2/3
    meter.set_levels(0.0, 0.0, 0.5)  # the bar falls back, the tick holds
    for dark, theme in ((False, LIGHT), (True, DARK)):
        meter.setPalette(palette(dark))
        image = meter.grab().toImage()
        rect = meter._bar_rect(0)
        x = rect.x() + rect.width() * meter.channels[0].hold_fraction
        pixel = image.pixelColor(int(x) - 1, int(rect.center().y()))
        assert pixel.rgb() == theme.meter_peak_tick.rgb()


def test_transport_state_colours_and_icons_follow_the_theme(qtbot):
    bar = TransportBar()
    qtbot.addWidget(bar)
    bar.setPalette(palette(False))
    bar.set_state(AppState.RECORDING)
    assert LIGHT.recording.name() in bar.state_label.styleSheet()
    stop_icon = bar.stop_button.icon().cacheKey()
    bar.setPalette(palette(True))
    assert DARK.recording.name() in bar.state_label.styleSheet()
    assert bar.stop_button.icon().cacheKey() != stop_icon  # redrawn in the new text colour
    bar.set_state(AppState.PAUSED)
    assert DARK.warning.name() in bar.state_label.styleSheet()
    bar.set_state(AppState.IDLE)
    assert bar.state_label.styleSheet() == ""


def test_playback_icons_are_redrawn_on_a_switch(qtbot):
    bar = PlaybackBar()
    qtbot.addWidget(bar)
    bar.setPalette(palette(False))
    before = bar.loop_button.icon().cacheKey()
    bar.setPalette(palette(True))
    assert bar.loop_button.icon().cacheKey() != before


def test_waveform_colours_follow_the_theme(qtbot):
    view = WaveformView()
    qtbot.addWidget(view)
    for dark, theme in ((True, DARK), (False, LIGHT), (True, DARK)):
        view.setPalette(palette(dark))
        assert view._wave_color.rgb() == theme.wave.rgb()
        assert view.curve.opts["pen"].color().rgb() == theme.wave.rgb()
        assert view.plot_widget.backgroundBrush().color() == view.palette().color(
            QPalette.ColorRole.Base
        )


def test_export_dialog_warning_follows_the_theme(qtbot, tmp_path):
    dialog = ExportDialog(
        frames=48000, rate=48000, settings=ExportSettings(), path=Path(tmp_path / "a.wav")
    )
    qtbot.addWidget(dialog)
    dialog.setPalette(palette(True))
    assert DARK.warning_style in dialog.warning_label.styleSheet()
    dialog.setPalette(palette(False))
    assert LIGHT.warning_style in dialog.warning_label.styleSheet()


@pytest.fixture
def window(qtbot, tmp_path):
    w = MainWindow(
        recorder=FakeRecorder(tmp_path),
        device_manager=FakeDeviceManager(),
        player=ClipPlayer(FakeBackend()),
        settings=Settings(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)),
    )
    qtbot.addWidget(w)
    return w


def test_window_styles_follow_the_theme(window):
    window.setPalette(palette(False))
    assert window.theme is LIGHT
    assert window.warning_label.styleSheet() == LIGHT.banner_style
    window.show_error("Something failed")
    assert window.statusBar().styleSheet() == LIGHT.status_error_style
    window._set_device(None)  # "no playback device found"
    assert window.device_label.styleSheet() == LIGHT.error_style

    window.setPalette(palette(True))
    assert window.theme is DARK
    assert window.warning_label.styleSheet() == DARK.banner_style
    assert window.statusBar().styleSheet() == DARK.status_error_style
    assert window.device_label.styleSheet() == DARK.error_style
    assert window.status_is_error

    window.show_info("Ready")
    window.setPalette(palette(False))
    assert window.statusBar().styleSheet() == ""  # info text keeps the style's colour
    assert not window.status_is_error
