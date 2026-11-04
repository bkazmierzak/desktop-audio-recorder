"""The Noise reduction dialog: options, noise profile, Preview with A/B playback.

What gets processed (``plan_denoise``, no Qt):

* With "Use the selection as the noise profile" (stationary mode only), the selection is the
  noise sample and the **whole clip** is processed.
* Otherwise the **selection** is processed, or the whole clip without one. The stationary mode
  then takes its noise profile from the quietest parts of that audio.

Preview processes part of it on the given runner (with a progress bar; a new Preview or closing
the dialog cancels the previous one): the selection, up to ``PREVIEW_MAX_SELECTION_SECONDS``,
or the first ``PREVIEW_SECONDS`` of the clip. The noise profile is taken exactly as Apply will
take it, so the preview sounds like the result. Original and Processed play that part on a
separate ``ClipPlayer`` in a loop; switching between them while playing keeps the position, for
an A/B comparison. Changing an option marks the preview out of date (it still plays) until
Preview is pressed again.

The dialog only collects a ``DenoiseRequest``; the main window runs the noise reduction itself
(``MainWindow.reduce_noise``), with progress and Cancel, as one undoable edit.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from desktop_audio_recorder.audio.clip import AudioClip
from desktop_audio_recorder.audio.player import ClipPlayer, PlayerError
from desktop_audio_recorder.processing.denoise import (
    HIGHPASS_RANGE_HZ,
    MIN_FRAMES,
    Cancelled,
    DenoiseSettings,
    NoiseMode,
    denoise,
    estimate_noise_profile,
)
from desktop_audio_recorder.ui.theme import watch_theme
from desktop_audio_recorder.ui.transport_bar import format_elapsed
from desktop_audio_recorder.ui.workers import ProgressJob, SerialTaskRunner

log = logging.getLogger(__name__)

PREVIEW_SECONDS = 10.0
PREVIEW_MAX_SELECTION_SECONDS = 30.0
PREVIEW_POLL_MS = 30
ORIGINAL, PROCESSED = "original", "processed"


@dataclass(frozen=True, slots=True)
class DenoiseRequest:
    settings: DenoiseSettings
    profile_from_selection: bool = False


@dataclass(frozen=True, slots=True)
class DenoisePlan:
    """Frame ranges in the clip: what to process, the noise profile (``None``: estimated from
    the target) and the preview."""

    start: int
    stop: int
    selected: bool  # the target is the user's selection (it stays selected afterwards)
    profile: tuple[int, int] | None
    preview: tuple[int, int]

    @property
    def frames(self) -> int:
        return self.stop - self.start


def plan_denoise(
    frames: int,
    rate: int,
    selection: tuple[int, int] | None,
    settings: DenoiseSettings,
    profile_from_selection: bool = False,
) -> DenoisePlan:
    """Where to process (see module docstring). Raises ``ValueError`` with a message for the
    user when the audio or the noise selection is too short."""
    shortest = f"{MIN_FRAMES / rate * 1000:.0f} ms"
    profile = None
    if profile_from_selection and settings.uses_profile:
        if selection is None:
            raise ValueError("Select a stretch of noise (no voice) to use as the noise profile")
        if selection[1] - selection[0] < MIN_FRAMES:
            raise ValueError(f"The noise selection is too short (at least {shortest})")
        start, stop, selected = 0, frames, False
        profile = selection
    elif selection is not None:
        start, stop, selected = selection[0], selection[1], True
    else:
        start, stop, selected = 0, frames, False
    if stop - start < MIN_FRAMES:
        what = "selection" if selected else "clip"
        raise ValueError(f"The {what} is too short for noise reduction (at least {shortest})")
    limit = PREVIEW_MAX_SELECTION_SECONDS if selected else PREVIEW_SECONDS
    preview = (start, min(stop, start + round(limit * rate)))
    return DenoisePlan(start, stop, selected, profile, preview)


def describe_range(plan: DenoisePlan, total_frames: int, rate: int) -> str:
    if plan.selected:
        return (
            f"the selection ({format_elapsed(plan.start / rate)} - "
            f"{format_elapsed(plan.stop / rate)}, {format_elapsed(plan.frames / rate)})"
        )
    return f"the whole clip ({format_elapsed(total_frames / rate)})"


def preview_task(
    samples: np.ndarray,
    rate: int,
    plan: DenoisePlan,
    settings: DenoiseSettings,
    job: ProgressJob,
) -> np.ndarray:
    """Process the preview part the way Apply will process it (runs on a worker thread)."""
    profile = None
    if plan.profile is not None:
        profile = samples[plan.profile[0] : plan.profile[1]]
    elif settings.uses_profile:
        profile = estimate_noise_profile(samples[plan.start : plan.stop], rate)
    if job.is_cancelled():
        raise Cancelled("Preview cancelled")
    a, b = plan.preview
    return denoise(samples[a:b], rate, settings, profile, job.report, job.is_cancelled)


class DenoiseDialog(QDialog):
    def __init__(
        self,
        clip: AudioClip,
        selection: tuple[int, int] | None,
        settings: DenoiseSettings | None = None,
        profile_from_selection: bool = False,
        *,
        runner: SerialTaskRunner,
        player: ClipPlayer | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Noise reduction")
        self.setModal(True)
        self._clip = clip
        self._selection = selection
        self._runner = runner
        self._player = player if player is not None else ClipPlayer()
        self._job: ProgressJob | None = None
        self._closed = False
        self._original: AudioClip | None = None  # the preview part, unprocessed
        self._original_range: tuple[int, int] | None = None
        self._processed: AudioClip | None = None
        self._processed_key: tuple | None = None  # (settings, plan) the preview was made with
        self._playing: str | None = None  # ORIGINAL / PROCESSED while playing
        self._error: str | None = None  # the last preview/playback problem, shown until fixed
        self._timer = QTimer(self)
        self._timer.setInterval(PREVIEW_POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._build_ui()
        self._set_settings(settings or DenoiseSettings(), profile_from_selection)
        self._refresh()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self.target_label = QLabel(self)
        self.target_label.setWordWrap(True)
        root.addWidget(self.target_label)

        mode_box = QGroupBox("Noise type", self)
        mode_layout = QVBoxLayout(mode_box)
        self.stationary_radio = QRadioButton(
            "Steady (stationary): hiss, hum, a fan, air conditioning", mode_box
        )
        self.nonstationary_radio = QRadioButton(
            "Changing (non-stationary): traffic, chatter, noise that comes and goes", mode_box
        )
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.stationary_radio)
        self._mode_group.addButton(self.nonstationary_radio)
        mode_layout.addWidget(self.stationary_radio)
        mode_layout.addWidget(self.nonstationary_radio)
        self.profile_check = QCheckBox("Use the selection as the noise profile", mode_box)
        self.profile_check.setToolTip(
            "Select a stretch with only noise (no voice) before opening this dialog; the whole"
            " clip is then cleaned using that noise as the reference"
        )
        mode_layout.addWidget(self.profile_check)
        self.profile_hint = QLabel(mode_box)
        self.profile_hint.setWordWrap(True)
        self.profile_hint.setEnabled(False)  # secondary text colour
        mode_layout.addWidget(self.profile_hint)
        root.addWidget(mode_box)

        options = QGridLayout()
        options.addWidget(QLabel("Strength:", self), 0, 0)
        self.strength_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setSingleStep(5)
        self.strength_slider.setPageStep(10)
        self.strength_slider.setToolTip(
            "How much of the noise to remove. Lower values sound more natural."
        )
        options.addWidget(self.strength_slider, 0, 1)
        self.strength_label = QLabel(self)
        self.strength_label.setMinimumWidth(48)
        options.addWidget(self.strength_label, 0, 2)
        self.highpass_check = QCheckBox("High-pass filter at", self)
        self.highpass_check.setToolTip("Remove rumble below the cutoff (handling noise, hum)")
        options.addWidget(self.highpass_check, 1, 0)
        self.highpass_spin = QSpinBox(self)
        self.highpass_spin.setRange(int(HIGHPASS_RANGE_HZ[0]), int(HIGHPASS_RANGE_HZ[1]))
        self.highpass_spin.setSingleStep(10)
        self.highpass_spin.setSuffix(" Hz")
        options.addWidget(self.highpass_spin, 1, 1)
        self.normalize_check = QCheckBox("Normalize to -1 dBFS afterwards", self)
        options.addWidget(self.normalize_check, 2, 0, 1, 3)
        root.addLayout(options)

        preview_box = QGroupBox("Preview", self)
        preview_layout = QVBoxLayout(preview_box)
        row = QHBoxLayout()
        self.preview_button = QPushButton("Preview", preview_box)
        self.preview_button.setToolTip("Process a short part with these settings")
        self.preview_progress = QProgressBar(preview_box)
        self.preview_progress.setRange(0, 100)
        self.preview_progress.setTextVisible(False)
        self.preview_progress.setMaximumHeight(10)
        row.addWidget(self.preview_button)
        row.addWidget(self.preview_progress, 1)
        preview_layout.addLayout(row)
        row = QHBoxLayout()
        self.original_button = QPushButton("Play original", preview_box)
        self.processed_button = QPushButton("Play processed", preview_box)
        self.stop_button = QPushButton("Stop", preview_box)
        for button in (self.original_button, self.processed_button, self.stop_button):
            row.addWidget(button)
        preview_layout.addLayout(row)
        self.preview_label = QLabel(preview_box)
        self.preview_label.setWordWrap(True)
        preview_layout.addWidget(self.preview_label)
        root.addWidget(preview_box)

        self.message_label = QLabel(self)
        self.message_label.setWordWrap(True)
        watch_theme(self, lambda theme: self.message_label.setStyleSheet(theme.error_style))
        root.addWidget(self.message_label)

        self.buttons = QDialogButtonBox(self)
        self.apply_button = self.buttons.addButton("Apply", QDialogButtonBox.ButtonRole.AcceptRole)
        self.buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        self.stationary_radio.toggled.connect(self._refresh)
        self.profile_check.toggled.connect(self._refresh)
        self.strength_slider.valueChanged.connect(self._refresh)
        self.highpass_check.toggled.connect(self._refresh)
        self.highpass_spin.valueChanged.connect(self._refresh)
        self.normalize_check.toggled.connect(self._refresh)
        self.preview_button.clicked.connect(self.start_preview)
        self.original_button.clicked.connect(lambda: self.play(ORIGINAL))
        self.processed_button.clicked.connect(lambda: self.play(PROCESSED))
        self.stop_button.clicked.connect(self.stop_playback)

    def _set_settings(self, settings: DenoiseSettings, profile_from_selection: bool) -> None:
        self.stationary_radio.setChecked(settings.stationary)
        self.nonstationary_radio.setChecked(not settings.stationary)
        self.strength_slider.setValue(round(settings.strength * 100))
        self.highpass_check.setChecked(settings.highpass_hz is not None)
        self.highpass_spin.setValue(round(settings.highpass_hz or 80))
        self.normalize_check.setChecked(settings.normalize)
        self.profile_check.setChecked(profile_from_selection and self._selection is not None)

    # ------------------------------------------------------------------ values

    def settings(self) -> DenoiseSettings:
        return DenoiseSettings(
            mode=NoiseMode.STATIONARY
            if self.stationary_radio.isChecked()
            else NoiseMode.NON_STATIONARY,
            strength=self.strength_slider.value() / 100,
            highpass_hz=float(self.highpass_spin.value())
            if self.highpass_check.isChecked()
            else None,
            normalize=self.normalize_check.isChecked(),
        )

    @property
    def profile_from_selection(self) -> bool:
        return self.profile_check.isEnabled() and self.profile_check.isChecked()

    def request(self) -> DenoiseRequest:
        return DenoiseRequest(self.settings(), self.profile_from_selection)

    def plan(self) -> DenoisePlan:
        """Raises ``ValueError`` (with a message) when there is nothing usable to process."""
        clip = self._clip
        return plan_denoise(
            clip.frames,
            clip.sample_rate,
            self._selection,
            self.settings(),
            self.profile_from_selection,
        )

    @property
    def job(self) -> ProgressJob | None:
        """The running preview's job, if any."""
        return self._job

    @property
    def player(self) -> ClipPlayer:
        return self._player

    @property
    def playing(self) -> str | None:
        return self._playing

    @property
    def preview_up_to_date(self) -> bool:
        return self._processed is not None and self._processed_key == self._key()

    def _key(self) -> tuple | None:
        try:
            return (self.settings(), self.plan())
        except ValueError:
            return None

    # ------------------------------------------------------------------ state

    def _refresh(self) -> None:
        settings = self.settings()
        self.strength_label.setText(f"{self.strength_slider.value()} %")
        self.highpass_spin.setEnabled(self.highpass_check.isChecked())
        has_selection = self._selection is not None
        self.profile_check.setEnabled(settings.uses_profile and has_selection)
        if not settings.gates:
            hint = "Strength 0: only the filter and normalize options apply."
        elif not settings.stationary:
            hint = "The changing-noise mode follows the noise by itself; it uses no profile."
        elif not has_selection:
            hint = (
                "Tip: select a stretch with only noise first to use it as the noise profile."
                " Without one, the quietest parts of the audio are used."
            )
        elif self.profile_from_selection:
            hint = "The selection is the noise sample; the whole clip is cleaned."
        else:
            hint = "The quietest parts of the selection are used as the noise sample."
        self.profile_hint.setText(hint)

        rate = self._clip.sample_rate
        try:
            plan = self.plan()
        except ValueError as exc:
            plan = None
            self.message_label.setText(str(exc))
        else:
            self.message_label.setText(
                "These settings change nothing." if settings.is_noop else self._error or ""
            )
        if plan is None:
            self.target_label.setText("Noise reduction can't be applied here.")
        else:
            self.target_label.setText(
                f"Applies to {describe_range(plan, self._clip.frames, rate)}."
            )
        self.apply_button.setEnabled(plan is not None and not settings.is_noop)

        running = self._job is not None
        self.preview_button.setEnabled(plan is not None and not settings.is_noop)
        self.preview_button.setText("Previewing…" if running else "Preview")
        self.preview_progress.setVisible(running)
        if plan is not None and self._original_range != plan.preview:
            self._set_original(plan)
        self.original_button.setEnabled(self._original is not None)
        self.processed_button.setEnabled(self._processed is not None)
        self.stop_button.setEnabled(self._playing is not None)
        self._update_preview_label()

    def _set_original(self, plan: DenoisePlan) -> None:
        """The preview part changed (e.g. the profile option moved the target): drop the
        preview made for the previous part."""
        a, b = plan.preview
        if self._playing is not None:
            self._player.stop()
            self._playing = None
        self._original = AudioClip(self._clip.samples[a:b].copy(), self._clip.sample_rate)
        self._original_range = plan.preview
        self._processed = None
        self._processed_key = None

    def _update_preview_label(self) -> None:
        rate = self._clip.sample_rate
        if self._original is None:
            self.preview_label.setText("")
            return
        a, b = self._original_range
        part = f"{format_elapsed(a / rate)} - {format_elapsed(b / rate)}"
        if self._job is not None:
            text = f"Processing {part}…"
        elif self._playing is not None:
            position = format_elapsed(self._player.position)
            text = f"Playing the {self._playing} {position} ({part}, looped)"
        elif self._processed is None:
            text = f"Preview processes {part}."
        else:
            text = f"Preview ready ({part})."
        if self._processed is not None and not self.preview_up_to_date:
            text += " The settings changed: press Preview to hear them."
        self.preview_label.setText(text)

    # ------------------------------------------------------------------ preview processing

    def start_preview(self) -> None:
        if self._closed:
            return
        try:
            plan, settings = self.plan(), self.settings()
        except ValueError:
            return
        if settings.is_noop:
            return
        if self._job is not None:
            self._job.cancel()
        job = self._job = ProgressJob()
        self._error = None
        samples, rate = self._clip.samples, self._clip.sample_rate
        self.preview_progress.setValue(0)
        self._timer.start()
        self._runner.submit(
            lambda: preview_task(samples, rate, plan, settings, job),
            lambda result: self._on_preview_done(job, (settings, plan), result),
            lambda exc: self._on_preview_failed(job, exc),
        )
        self._refresh()

    def _on_preview_done(self, job: ProgressJob, key: tuple, result: np.ndarray) -> None:
        if self._closed or job is not self._job:
            return
        self._job = None
        if key[1].preview != self._original_range:
            self._refresh()  # made for a part that is no longer the preview
            return
        self._processed = AudioClip(result, self._clip.sample_rate)
        self._processed_key = key
        self._error = None
        if self._playing == PROCESSED or self._playing is None:
            self.play(PROCESSED)  # swap in the new version at the same position, or start
        self._refresh()

    def _on_preview_failed(self, job: ProgressJob, exc: BaseException) -> None:
        if self._closed or job is not self._job:
            return
        self._job = None
        if not isinstance(exc, Cancelled):
            log.error("Preview failed", exc_info=exc)
            self._error = f"Preview failed: {exc}"
        self._refresh()

    # ------------------------------------------------------------------ preview playback

    def play(self, which: str) -> None:
        """Play the original or the processed preview, looped. Switching while playing swaps
        the audio under the running stream (``ClipPlayer.swap_clip``): seamless, same position,
        and no stream reopen (closing and reopening one costs ~100+ ms on the GUI thread)."""
        clip = self._original if which == ORIGINAL else self._processed
        if clip is None or self._closed:
            return
        player = self._player
        if self._playing is not None and player.is_playing and player.swap_clip(clip):
            self._playing = which
            self._error = None
            self._refresh()
            return
        position = player.position if self._playing is not None else 0.0
        player.set_clip(clip)
        player.loop = True
        player.seek(position)
        try:
            player.play()
        except PlayerError as exc:
            self._playing = None
            self._error = str(exc)
            self._refresh()
            return
        self._playing = which
        self._error = None
        self._timer.start()
        self._refresh()

    def stop_playback(self) -> None:
        self._player.stop()
        self._playing = None
        self._refresh()

    def _poll(self) -> None:
        if self._job is not None:
            self.preview_progress.setValue(round(self._job.fraction * 100))
        if self._playing is not None and self._player.update():
            self._playing = None  # ended (device lost: the loop never ends by itself)
            if self._player.last_error:
                self._error = self._player.last_error
            self._refresh()
        if self._playing is not None:
            self._update_preview_label()
        if self._job is None and self._playing is None:
            self._timer.stop()

    # ------------------------------------------------------------------ closing

    def done(self, result: int) -> None:
        self._close()
        super().done(result)

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        if self._job is not None:
            self._job.cancel()
            self._job = None
        self._player.close()
        self._playing = None


def ask_denoise(
    parent: QWidget | None,
    clip: AudioClip,
    selection: tuple[int, int] | None,
    settings: DenoiseSettings,
    profile_from_selection: bool,
    *,
    runner: SerialTaskRunner,
    player_factory: Callable[[], ClipPlayer] = ClipPlayer,
) -> DenoiseRequest | None:
    """Modal dialog; ``None`` if cancelled."""
    dialog = DenoiseDialog(
        clip,
        selection,
        settings,
        profile_from_selection,
        runner=runner,
        player=player_factory(),
        parent=parent,
    )
    try:
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.request()
    finally:
        dialog.deleteLater()
