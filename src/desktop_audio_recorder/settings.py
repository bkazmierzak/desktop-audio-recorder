"""Persistent user settings: a thin wrapper around ``QSettings``.

On Windows the values live in the registry under
``HKEY_CURRENT_USER\\Software\\DesktopAudioRecorder\\Desktop Audio Recorder``. Tests pass
their own ``QSettings`` (an INI file in a temp folder). Every getter validates what it reads and
falls back to the default, so a damaged or outdated value never breaks the app.

Stored: the export choices (folder, format, bit depth, bitrate, dither), the File > Open
folder, the main window's geometry, the noise reduction defaults, the Loop toggle and whether a
recording follows the default playback device.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QByteArray, QSettings, QStandardPaths

from desktop_audio_recorder import APP_NAME
from desktop_audio_recorder.export.writers import (
    DEFAULT_DITHER,
    DEFAULT_MP3_BITRATE,
    DEFAULT_WAV_SUBTYPE,
    MP3_BITRATES,
    WAV_SUBTYPES,
    ExportFormat,
    ExportSettings,
)
from desktop_audio_recorder.processing.denoise import (
    HIGHPASS_RANGE_HZ,
    DenoiseSettings,
    NoiseMode,
)

log = logging.getLogger(__name__)

ORGANIZATION = "DesktopAudioRecorder"

_EXPORT_FOLDER = "export/folder"
_OPEN_FOLDER = "open/folder"
_EXPORT_FORMAT = "export/format"
_WAV_SUBTYPE = "export/wav_subtype"
_MP3_BITRATE = "export/mp3_bitrate"
_DITHER = "export/dither"
_WINDOW_GEOMETRY = "window/geometry"
_DENOISE_MODE = "denoise/mode"
_DENOISE_STRENGTH = "denoise/strength"
_DENOISE_HIGHPASS = "denoise/highpass"
_DENOISE_HIGHPASS_HZ = "denoise/highpass_hz"
_DENOISE_NORMALIZE = "denoise/normalize"
_DENOISE_PROFILE = "denoise/profile_from_selection"
_LOOP = "playback/loop"
_FOLLOW_DEFAULT = "recording/follow_default"


def _as_bool(value, default: bool) -> bool:
    """QSettings hands bools back as ``"true"``/``"false"`` strings from INI files and the
    registry; anything unrecognised gives ``default``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1", "yes", "on"):
            return True
        if text in ("false", "0", "no", "off"):
            return False
    return default


def _as_float(value, default: float, low: float, high: float) -> float:
    """A finite float within ``[low, high]``, else ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if low <= number <= high else default  # also rejects NaN


def default_export_folder() -> str:
    """The user's Music folder (or Documents, or home)."""
    for location in (
        QStandardPaths.StandardLocation.MusicLocation,
        QStandardPaths.StandardLocation.DocumentsLocation,
    ):
        folder = QStandardPaths.writableLocation(location)
        if folder and Path(folder).is_dir():
            return str(Path(folder))
    return str(Path.home())


class Settings:
    def __init__(self, store: QSettings | None = None) -> None:
        self._store = (
            store
            if store is not None
            else QSettings(
                QSettings.Format.NativeFormat, QSettings.Scope.UserScope, ORGANIZATION, APP_NAME
            )
        )

    @property
    def store(self) -> QSettings:
        return self._store

    def _get(self, key: str, default=None):
        try:
            return self._store.value(key, default)
        except Exception:  # an unreadable value
            log.warning("Ignoring the unreadable setting %s", key, exc_info=True)
            return default

    def sync(self) -> None:
        self._store.sync()

    # ------------------------------------------------------------------ export

    @property
    def export_folder(self) -> str:
        """The folder of the last export, if it still exists; else the Music folder."""
        value = self._get(_EXPORT_FOLDER)
        if isinstance(value, str) and value and Path(value).is_dir():
            return value
        return default_export_folder()

    @export_folder.setter
    def export_folder(self, folder: str | Path) -> None:
        self._store.setValue(_EXPORT_FOLDER, str(folder))

    @property
    def open_folder(self) -> str:
        """The folder of the last file opened, if it still exists; else the export folder."""
        value = self._get(_OPEN_FOLDER)
        if isinstance(value, str) and value and Path(value).is_dir():
            return value
        return self.export_folder

    @open_folder.setter
    def open_folder(self, folder: str | Path) -> None:
        self._store.setValue(_OPEN_FOLDER, str(folder))

    @property
    def export_settings(self) -> ExportSettings:
        """The last format, WAV bit depth, MP3 bitrate and dither (each falls back on its own)."""
        fmt = self._get(_EXPORT_FORMAT, ExportFormat.WAV.value)
        try:
            export_format = ExportFormat(str(fmt))
        except ValueError:
            export_format = ExportFormat.WAV
        subtype = str(self._get(_WAV_SUBTYPE, DEFAULT_WAV_SUBTYPE))
        if subtype not in WAV_SUBTYPES:
            subtype = DEFAULT_WAV_SUBTYPE
        try:
            bitrate = int(self._get(_MP3_BITRATE, DEFAULT_MP3_BITRATE))
        except (TypeError, ValueError):
            bitrate = DEFAULT_MP3_BITRATE
        if bitrate not in MP3_BITRATES:
            bitrate = DEFAULT_MP3_BITRATE
        dither = _as_bool(self._get(_DITHER), DEFAULT_DITHER)
        return ExportSettings(export_format, subtype, bitrate, dither)

    @export_settings.setter
    def export_settings(self, settings: ExportSettings) -> None:
        self._store.setValue(_EXPORT_FORMAT, settings.format.value)
        self._store.setValue(_WAV_SUBTYPE, settings.wav_subtype)
        self._store.setValue(_MP3_BITRATE, settings.mp3_bitrate)
        self._store.setValue(_DITHER, bool(settings.dither))

    # ------------------------------------------------------------------ window

    @property
    def window_geometry(self) -> QByteArray | None:
        """``QWidget.saveGeometry()`` of the main window when it last closed (``None`` if never)."""
        value = self._get(_WINDOW_GEOMETRY)
        if isinstance(value, QByteArray) and not value.isEmpty():
            return value
        if isinstance(value, (bytes, bytearray)) and value:
            return QByteArray(bytes(value))
        return None

    @window_geometry.setter
    def window_geometry(self, geometry: QByteArray) -> None:
        self._store.setValue(_WINDOW_GEOMETRY, QByteArray(geometry))

    # ------------------------------------------------------------------ noise reduction

    @property
    def denoise_settings(self) -> DenoiseSettings:
        """The noise reduction options used last (each falls back on its own)."""
        default = DenoiseSettings()
        try:
            mode = NoiseMode(str(self._get(_DENOISE_MODE, default.mode.value)))
        except ValueError:
            mode = default.mode
        strength = _as_float(self._get(_DENOISE_STRENGTH), default.strength, 0.0, 1.0)
        highpass_on = _as_bool(self._get(_DENOISE_HIGHPASS), default.highpass_hz is not None)
        cutoff = _as_float(
            self._get(_DENOISE_HIGHPASS_HZ), default.highpass_hz or 80.0, *HIGHPASS_RANGE_HZ
        )
        normalize = _as_bool(self._get(_DENOISE_NORMALIZE), default.normalize)
        return DenoiseSettings(
            mode=mode,
            strength=strength,
            highpass_hz=cutoff if highpass_on else None,
            normalize=normalize,
        )

    @denoise_settings.setter
    def denoise_settings(self, settings: DenoiseSettings) -> None:
        self._store.setValue(_DENOISE_MODE, settings.mode.value)
        self._store.setValue(_DENOISE_STRENGTH, float(settings.strength))
        self._store.setValue(_DENOISE_HIGHPASS, settings.highpass_hz is not None)
        if settings.highpass_hz is not None:  # keep the last cutoff while high-pass is off
            self._store.setValue(_DENOISE_HIGHPASS_HZ, float(settings.highpass_hz))
        self._store.setValue(_DENOISE_NORMALIZE, bool(settings.normalize))

    @property
    def denoise_profile_from_selection(self) -> bool:
        return _as_bool(self._get(_DENOISE_PROFILE), False)

    @denoise_profile_from_selection.setter
    def denoise_profile_from_selection(self, value: bool) -> None:
        self._store.setValue(_DENOISE_PROFILE, bool(value))

    # ------------------------------------------------------------------ playback

    @property
    def loop(self) -> bool:
        """The playback bar's Loop toggle."""
        return _as_bool(self._get(_LOOP), False)

    @loop.setter
    def loop(self, value: bool) -> None:
        self._store.setValue(_LOOP, bool(value))

    # ------------------------------------------------------------------ recording

    @property
    def follow_default_device(self) -> bool:
        """Recording > Follow the default playback device (on by default)."""
        return _as_bool(self._get(_FOLLOW_DEFAULT), True)

    @follow_default_device.setter
    def follow_default_device(self, value: bool) -> None:
        self._store.setValue(_FOLLOW_DEFAULT, bool(value))
