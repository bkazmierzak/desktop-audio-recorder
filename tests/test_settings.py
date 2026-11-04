"""settings.py: QSettings wrapper (an INI file per test, never the registry)."""

import pytest
from PySide6.QtCore import QSettings

from desktop_audio_recorder.export.writers import ExportFormat, ExportSettings
from desktop_audio_recorder.settings import Settings, default_export_folder


@pytest.fixture
def ini(tmp_path):
    return str(tmp_path / "settings.ini")


def make(ini):
    return Settings(QSettings(ini, QSettings.Format.IniFormat))


def test_defaults(ini):
    s = make(ini)
    assert s.export_settings == ExportSettings()
    assert s.export_folder == default_export_folder()


def test_round_trip_across_instances(ini, tmp_path):
    s = make(ini)
    s.export_settings = ExportSettings(ExportFormat.MP3, "PCM_24", 320)
    s.export_folder = tmp_path
    s.sync()
    again = make(ini)
    assert again.export_settings == ExportSettings(ExportFormat.MP3, "PCM_24", 320)
    assert again.export_folder == str(tmp_path)


def test_bad_values_fall_back_one_by_one(ini):
    store = QSettings(ini, QSettings.Format.IniFormat)
    store.setValue("export/format", "flac")
    store.setValue("export/wav_subtype", "FLOAT")
    store.setValue("export/mp3_bitrate", "loud")
    s = Settings(store)
    assert s.export_settings == ExportSettings(ExportFormat.WAV, "FLOAT", 192)
    store.setValue("export/format", "mp3")
    store.setValue("export/wav_subtype", "PCM_8")
    store.setValue("export/mp3_bitrate", 160)
    assert s.export_settings == ExportSettings(ExportFormat.MP3, "PCM_16", 192)


def test_a_folder_that_no_longer_exists_falls_back(ini, tmp_path):
    s = make(ini)
    gone = tmp_path / "gone"
    gone.mkdir()
    s.export_folder = gone
    assert s.export_folder == str(gone)
    gone.rmdir()
    assert s.export_folder == default_export_folder()


def test_default_folder_exists():
    from pathlib import Path

    assert Path(default_export_folder()).is_dir()


# --------------------------------------------------------------------------- Phase 7 keys


def test_new_keys_have_defaults(ini):
    from desktop_audio_recorder.processing.denoise import DenoiseSettings

    s = make(ini)
    assert s.window_geometry is None
    assert s.denoise_settings == DenoiseSettings()
    assert s.denoise_profile_from_selection is False
    assert s.loop is False


def test_denoise_settings_round_trip(ini):
    from desktop_audio_recorder.processing.denoise import DenoiseSettings, NoiseMode

    chosen = DenoiseSettings(NoiseMode.NON_STATIONARY, 0.55, 120.0, normalize=True)
    s = make(ini)
    s.denoise_settings = chosen
    s.denoise_profile_from_selection = True
    s.loop = True
    s.sync()
    again = make(ini)
    assert again.denoise_settings == chosen
    assert again.denoise_profile_from_selection is True
    assert again.loop is True


def test_highpass_off_keeps_the_last_cutoff(ini):
    from desktop_audio_recorder.processing.denoise import DenoiseSettings

    s = make(ini)
    s.denoise_settings = DenoiseSettings(highpass_hz=150.0)
    s.denoise_settings = DenoiseSettings(highpass_hz=None)
    assert s.denoise_settings.highpass_hz is None
    store = s.store
    assert float(store.value("denoise/highpass_hz")) == 150.0


@pytest.mark.parametrize(
    "key, value",
    [
        ("denoise/mode", "loud"),
        ("denoise/strength", "lots"),
        ("denoise/strength", 1.5),
        ("denoise/strength", "nan"),
        ("denoise/highpass", "maybe"),
        ("denoise/highpass_hz", 5),
        ("denoise/highpass_hz", 9000),
        ("denoise/normalize", "perhaps"),
    ],
)
def test_bad_denoise_values_fall_back(ini, key, value):
    from desktop_audio_recorder.processing.denoise import DenoiseSettings

    store = QSettings(ini, QSettings.Format.IniFormat)
    store.setValue(key, value)
    assert Settings(store).denoise_settings == DenoiseSettings()


def test_bool_values_from_strings(ini):
    store = QSettings(ini, QSettings.Format.IniFormat)
    s = Settings(store)
    for text, expected in [("true", True), ("0", False), ("on", True), ("garbage", False)]:
        store.setValue("playback/loop", text)
        assert s.loop is expected


def test_window_geometry_round_trip(ini, qtbot):
    from PySide6.QtWidgets import QWidget

    widget = QWidget()
    qtbot.addWidget(widget)
    widget.setGeometry(120, 140, 640, 480)
    s = make(ini)
    s.window_geometry = widget.saveGeometry()
    s.sync()
    restored = make(ini).window_geometry
    assert restored is not None and restored == widget.saveGeometry()
    other = QWidget()
    qtbot.addWidget(other)
    assert other.restoreGeometry(restored)
    assert (other.width(), other.height()) == (640, 480)


def test_garbage_geometry_is_ignored(ini):
    store = QSettings(ini, QSettings.Format.IniFormat)
    store.setValue("window/geometry", 42)
    assert Settings(store).window_geometry is None


def test_dither_round_trip_and_fallback(ini):
    s = make(ini)
    assert s.export_settings.dither  # on by default
    s.export_settings = ExportSettings(ExportFormat.MP3, dither=False)
    assert make(ini).export_settings == ExportSettings(ExportFormat.MP3, dither=False)
    s.store.setValue("export/dither", "garbage")
    assert make(ini).export_settings.dither
