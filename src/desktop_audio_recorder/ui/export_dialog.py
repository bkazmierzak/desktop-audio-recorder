"""Export dialog (format, bit depth, bitrate, dither, file) and the "unsaved changes" prompt.

``ExportDialog`` holds every choice in one place: WAV or MP3, the WAV bit depth or the MP3 bitrate,
and the file, prefilled from the last export folder and a name based on the recording time.
"Browse…" opens the native Windows "Save As" dialog (WAV/MP3 filters; picking a type there
switches the format, and it asks about overwriting). Typing a name with ``.wav``/``.mp3`` picks the
format too, and switching the format changes the extension. An existing file that the Save As
dialog didn't already confirm is confirmed on Export.

``ask_unsaved()`` asks Save / Discard / Cancel before a new recording, opening another file or
closing the window (Discard / Cancel only while the audio is still being recorded or processed).
``ask_open_path()`` is File > Open's native dialog.

Both modal parts are replaceable in tests: ``ExportDialog.get_save_path`` and
``ExportDialog.confirm_overwrite``, and the main window's ``prompt_export``/``prompt_unsaved``.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from desktop_audio_recorder.export.writers import (
    MP3_BITRATES,
    WAV_SUBTYPES,
    ExportFormat,
    ExportSettings,
    estimate_bytes,
    mp3_rate,
)
from desktop_audio_recorder.ui.theme import watch_theme
from desktop_audio_recorder.ui.transport_bar import format_elapsed

FILTERS = {
    ExportFormat.WAV: "WAV audio (*.wav)",
    ExportFormat.MP3: "MP3 audio (*.mp3)",
}
# What File > Open offers: everything libsndfile 1.2 reads that people have (MP3 via mpg123).
OPEN_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".oga", ".aif", ".aiff", ".w64", ".rf64")
OPEN_FILTERS = (
    "Audio files (" + " ".join(f"*{e}" for e in OPEN_EXTENSIONS) + ")",
    "WAV audio (*.wav)",
    "MP3 audio (*.mp3)",
    "All files (*)",
)
WAV_HINTS = {
    "PCM_16": "16-bit PCM (CD quality, plays everywhere)",
    "PCM_24": "24-bit PCM (more headroom for further editing)",
    "FLOAT": "32-bit float (lossless; keeps peaks above 0 dBFS)",
}
MP3_HINTS = {
    128: "128 kbps (small)",
    192: "192 kbps (good)",
    256: "256 kbps",
    320: "320 kbps (best)",
}


@dataclass(frozen=True, slots=True)
class ExportRequest:
    path: Path
    settings: ExportSettings


def format_size(nbytes: int) -> str:
    """``"850 KB"``, ``"2.3 MB"``, ``"1.25 GB"``."""
    if nbytes < 1000 * 1000:
        return f"{max(1, round(nbytes / 1000))} KB"
    if nbytes < 1000**3:
        return f"{nbytes / 1000**2:.1f} MB"
    return f"{nbytes / 1000**3:.2f} GB"


def with_extension(path: str | Path, export_format: ExportFormat) -> Path:
    """``path`` with the format's extension (replacing a .wav/.mp3 one, else appended)."""
    path = Path(path)
    if ExportFormat.from_path(path) is not None:
        return path.with_suffix(export_format.extension)
    return path.with_name(path.name + export_format.extension)


def native_save_path(
    parent: QWidget | None, path: Path, export_format: ExportFormat
) -> tuple[str, ExportFormat] | None:
    """The native "Save As" dialog; ``(path, format of the chosen filter)`` or ``None``."""
    chosen, selected = QFileDialog.getSaveFileName(
        parent,
        "Export as",
        str(path),
        ";;".join(FILTERS.values()),
        FILTERS[export_format],
    )
    if not chosen:
        return None
    by_filter = next((f for f, text in FILTERS.items() if text == selected), export_format)
    return chosen, by_filter


def ask_open_path(parent: QWidget | None, folder: str | Path) -> Path | None:
    """The native "Open" dialog for an audio file; ``None`` if cancelled."""
    chosen, _ = QFileDialog.getOpenFileName(
        parent, "Open audio file", str(folder), ";;".join(OPEN_FILTERS), OPEN_FILTERS[0]
    )
    return Path(chosen) if chosen else None


def confirm_overwrite(parent: QWidget | None, path: Path) -> bool:
    answer = QMessageBox.question(
        parent,
        "Export",
        f"{path.name} already exists.\nDo you want to replace it?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return answer == QMessageBox.StandardButton.Yes


class ExportDialog(QDialog):
    def __init__(
        self,
        *,
        frames: int,
        rate: int,
        settings: ExportSettings,
        path: Path,
        has_overs: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.setModal(True)
        self._frames = frames
        self._rate = rate
        self._has_overs = has_overs
        self._confirmed: Path | None = None  # a path the Save As dialog already confirmed
        # Replaceable in tests.
        self.get_save_path: Callable[[Path, ExportFormat], tuple[str, ExportFormat] | None] = (
            lambda p, f: native_save_path(self, p, f)
        )
        self.confirm_overwrite: Callable[[Path], bool] = lambda p: confirm_overwrite(self, p)
        self._build_ui()
        self._set_settings(settings)
        self.path_edit.setText(str(with_extension(path, settings.format)))
        self._refresh()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)

        grid.addWidget(QLabel("Format:", self), 0, 0)
        formats = QHBoxLayout()
        self.wav_radio = QRadioButton("WAV", self)
        self.mp3_radio = QRadioButton("MP3", self)
        self._format_group = QButtonGroup(self)
        self._format_group.addButton(self.wav_radio)
        self._format_group.addButton(self.mp3_radio)
        formats.addWidget(self.wav_radio)
        formats.addWidget(self.mp3_radio)
        formats.addStretch(1)
        grid.addLayout(formats, 0, 1, 1, 2)

        self.wav_label = QLabel("Bit depth:", self)
        self.wav_combo = QComboBox(self)
        for subtype in WAV_SUBTYPES:
            self.wav_combo.addItem(WAV_HINTS[subtype], subtype)
        grid.addWidget(self.wav_label, 1, 0)
        grid.addWidget(self.wav_combo, 1, 1, 1, 2)

        self.mp3_label = QLabel("Bitrate:", self)
        self.mp3_combo = QComboBox(self)
        for bitrate in MP3_BITRATES:
            self.mp3_combo.addItem(MP3_HINTS[bitrate], bitrate)
        grid.addWidget(self.mp3_label, 2, 0)
        grid.addWidget(self.mp3_combo, 2, 1, 1, 2)

        self.dither_check = QCheckBox("Dither (recommended for 16-bit)", self)
        self.dither_check.setToolTip(
            "Adds very quiet noise (about -96 dBFS) before rounding to 16 bits, so quiet passages"
            " and fades keep their detail instead of picking up distortion"
        )
        grid.addWidget(self.dither_check, 3, 1, 1, 2)
        # Keep the label column as wide as its widest label, so the fields don't move when
        # switching between WAV and MP3 hides one of the rows.
        metrics = self.fontMetrics()
        grid.setColumnMinimumWidth(
            0, max(metrics.horizontalAdvance(t) for t in ("Format:", "Bit depth:", "Bitrate:"))
        )

        grid.addWidget(QLabel("File:", self), 4, 0)
        self.path_edit = QLineEdit(self)
        self.path_edit.setMinimumWidth(380)
        self.browse_button = QPushButton("Browse…", self)
        self.browse_button.setToolTip("Choose the folder and file name")
        grid.addWidget(self.path_edit, 4, 1)
        grid.addWidget(self.browse_button, 4, 2)
        root.addLayout(grid)

        self.summary_label = QLabel(self)
        self.summary_label.setEnabled(False)  # secondary text colour
        root.addWidget(self.summary_label)
        self.warning_label = QLabel(self)
        self.warning_label.setWordWrap(True)
        watch_theme(self, lambda theme: self.warning_label.setStyleSheet(theme.warning_style))
        root.addWidget(self.warning_label)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        self.export_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.export_button.setText("Export")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        self.wav_radio.toggled.connect(self._on_format_toggled)
        self.wav_combo.currentIndexChanged.connect(self._refresh)
        self.mp3_combo.currentIndexChanged.connect(self._refresh)
        self.dither_check.toggled.connect(self._refresh)
        self.path_edit.textChanged.connect(self._on_path_edited)
        self.browse_button.clicked.connect(self.browse)

    def _set_settings(self, settings: ExportSettings) -> None:
        (self.wav_radio if settings.format is ExportFormat.WAV else self.mp3_radio).setChecked(True)
        self.wav_combo.setCurrentIndex(WAV_SUBTYPES.index(settings.wav_subtype))
        self.mp3_combo.setCurrentIndex(MP3_BITRATES.index(settings.mp3_bitrate))
        self.dither_check.setChecked(settings.dither)

    # ------------------------------------------------------------------ state

    @property
    def export_format(self) -> ExportFormat:
        return ExportFormat.WAV if self.wav_radio.isChecked() else ExportFormat.MP3

    def settings(self) -> ExportSettings:
        return ExportSettings(
            self.export_format,
            self.wav_combo.currentData(),
            self.mp3_combo.currentData(),
            self.dither_check.isChecked(),
        )

    def path(self) -> Path | None:
        """The file to write (with the format's extension), or ``None`` if not usable."""
        text = self.path_edit.text().strip().strip('"')
        if not text:
            return None
        path = Path(text)
        if ExportFormat.from_path(path) is not self.export_format:
            path = with_extension(path, self.export_format)
        return path

    def request(self) -> ExportRequest | None:
        path = self.path()
        return None if path is None else ExportRequest(path, self.settings())

    def set_format(self, export_format: ExportFormat) -> None:
        (self.wav_radio if export_format is ExportFormat.WAV else self.mp3_radio).setChecked(True)

    def _on_format_toggled(self) -> None:
        text = self.path_edit.text().strip()
        if text:
            new = str(with_extension(text, self.export_format))
            if new != text:
                self.path_edit.blockSignals(True)
                self.path_edit.setText(new)
                self.path_edit.blockSignals(False)
        self._refresh()

    def _on_path_edited(self, text: str) -> None:
        typed = ExportFormat.from_path(text.strip())
        if typed is not None and typed is not self.export_format:
            self.wav_radio.blockSignals(True)
            self.mp3_radio.blockSignals(True)
            self.set_format(typed)
            self.wav_radio.blockSignals(False)
            self.mp3_radio.blockSignals(False)
        self._refresh()

    def _refresh(self) -> None:
        settings = self.settings()
        wav = settings.format is ExportFormat.WAV
        for widget in (self.wav_label, self.wav_combo):
            widget.setVisible(wav)
        for widget in (self.mp3_label, self.mp3_combo):
            widget.setVisible(not wav)
        # Only 16-bit output is dithered; the choice is kept (and saved) either way.
        self.dither_check.setVisible(settings.is_16_bit)
        rate = self._rate if wav else mp3_rate(self._rate)
        size = format_size(estimate_bytes(self._frames, self._rate, settings))
        summary = f"{format_elapsed(self._frames / self._rate)} · {rate / 1000:g} kHz stereo"
        if rate != self._rate:
            summary += f" (resampled from {self._rate / 1000:g} kHz)"
        self.summary_label.setText(f"{summary} · about {size}")
        warnings = []
        if self._has_overs and settings.clips_overs:
            warnings.append(
                "The clip has peaks above 0 dBFS; they will be clipped. Normalize first, or"
                " export a 32-bit float WAV, to keep them."
            )
        path = self.path()
        if path is not None and not path.parent.is_dir():
            warnings.append(f"The folder {path.parent} doesn't exist.")
        self.warning_label.setText("\n".join(warnings))
        self.warning_label.setVisible(bool(warnings))
        self.export_button.setEnabled(path is not None and path.parent.is_dir())

    # ------------------------------------------------------------------ actions

    def browse(self) -> None:
        current = self.path() or Path.home() / f"Recording{self.export_format.extension}"
        chosen = self.get_save_path(current, self.export_format)
        if chosen is None:
            return
        text, by_filter = chosen
        path = Path(text)
        export_format = ExportFormat.from_path(path) or by_filter
        path = with_extension(path, export_format)
        self.set_format(export_format)
        self.path_edit.setText(str(path))
        self._confirmed = path  # the Save As dialog asked about replacing it

    def accept(self) -> None:
        path = self.path()
        if path is None or not path.parent.is_dir():
            return
        if path.exists() and path != self._confirmed and not self.confirm_overwrite(path):
            return
        self.path_edit.setText(str(path))
        super().accept()


def ask_export(
    parent: QWidget | None,
    *,
    frames: int,
    rate: int,
    settings: ExportSettings,
    path: Path,
    has_overs: bool,
) -> ExportRequest | None:
    """Modal export dialog; ``None`` if cancelled."""
    dialog = ExportDialog(
        frames=frames, rate=rate, settings=settings, path=path, has_overs=has_overs, parent=parent
    )
    try:
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.request()
    finally:
        dialog.deleteLater()


# --------------------------------------------------------------------------- unsaved changes


class UnsavedChoice(enum.Enum):
    SAVE = "save"
    DISCARD = "discard"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class UnsavedPrompt:
    """What to ask: ``closing`` (else a new recording) and whether Save is possible now."""

    closing: bool
    can_save: bool
    text: str
    detail: str


def ask_unsaved(parent: QWidget | None, prompt: UnsavedPrompt) -> UnsavedChoice:
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Unsaved audio")
    box.setText(prompt.text)
    box.setInformativeText(prompt.detail)
    buttons = QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel
    if prompt.can_save:
        buttons |= QMessageBox.StandardButton.Save
    box.setStandardButtons(buttons)
    box.setDefaultButton(
        QMessageBox.StandardButton.Save if prompt.can_save else QMessageBox.StandardButton.Cancel
    )
    box.setEscapeButton(QMessageBox.StandardButton.Cancel)
    if prompt.can_save:
        box.button(QMessageBox.StandardButton.Save).setText("Save…")
    answer = box.exec()
    if answer == QMessageBox.StandardButton.Save:
        return UnsavedChoice.SAVE
    if answer == QMessageBox.StandardButton.Discard:
        return UnsavedChoice.DISCARD
    return UnsavedChoice.CANCEL


class RecoveryChoice(enum.Enum):
    RECOVER = "recover"
    DELETE = "delete"
    LATER = "later"


@dataclass(frozen=True, slots=True)
class RecoveryPrompt:
    """A recording left behind by a crash: ``text`` names it, ``count`` is how many there are
    (Delete deletes all of them)."""

    text: str
    detail: str
    count: int


def ask_recovery(parent: QWidget | None, prompt: RecoveryPrompt) -> RecoveryChoice:
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("Recover recording")
    box.setText(prompt.text)
    box.setInformativeText(prompt.detail)
    recover = box.addButton("Recover", QMessageBox.ButtonRole.AcceptRole)
    delete = box.addButton(
        "Delete" if prompt.count == 1 else f"Delete all {prompt.count}",
        QMessageBox.ButtonRole.DestructiveRole,
    )
    later = box.addButton("Not now", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(recover)
    box.setEscapeButton(later)
    box.exec()
    clicked = box.clickedButton()
    if clicked is recover:
        return RecoveryChoice.RECOVER
    if clicked is delete:
        return RecoveryChoice.DELETE
    return RecoveryChoice.LATER
