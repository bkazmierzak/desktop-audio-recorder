"""Noise reduction in the UI: plan_denoise, DenoiseDialog (options, preview A/B) and the main
window's PROCESSING state (progress, Cancel, undo).

Offscreen, with the fake recorder writing a noisy voice-like clip and fake playback backends.
"""

import threading

import numpy as np
import pytest
from conftest import FakeBackend, FakeDeviceManager, FakeRecorder, speech_like
from PySide6.QtCore import Qt

from desktop_audio_recorder.audio.clip import AudioClip
from desktop_audio_recorder.audio.player import ClipPlayer
from desktop_audio_recorder.processing.denoise import (
    MIN_FRAMES,
    DenoiseSettings,
    NoiseMode,
    denoise,
    snr_db,
)
from desktop_audio_recorder.ui import main_window as mw
from desktop_audio_recorder.ui.denoise_dialog import (
    ORIGINAL,
    PROCESSED,
    DenoiseDialog,
    DenoisePlan,
    DenoiseRequest,
    plan_denoise,
)
from desktop_audio_recorder.ui.state import AppState
from desktop_audio_recorder.ui.workers import ProgressJob, SerialTaskRunner

TIMEOUT = 5000
RATE = 48000
SECONDS = 3.0
FRAMES = round(SECONDS * RATE)
STATIONARY = DenoiseSettings()
NON_STATIONARY = DenoiseSettings(mode=NoiseMode.NON_STATIONARY)


def noisy_clip_samples(seconds=SECONDS, level=0.05):
    clean, _ = speech_like(seconds, RATE)
    clean = np.repeat(clean[:, None], 2, axis=1)
    noise = level * np.random.default_rng(1).standard_normal(clean.shape)
    return clean, (clean + noise).astype(np.float32)


CLEAN, NOISY = noisy_clip_samples()


def same(a, b):
    assert a.shape == b.shape and a.tobytes() == b.tobytes()


# ------------------------------------------------------------------------- plan_denoise


def test_plan_whole_clip_without_selection():
    plan = plan_denoise(20 * RATE, RATE, None, STATIONARY)
    assert plan == DenoisePlan(0, 20 * RATE, False, None, (0, 10 * RATE))


def test_plan_selection():
    plan = plan_denoise(100 * RATE, RATE, (RATE, 3 * RATE), STATIONARY)
    assert plan == DenoisePlan(RATE, 3 * RATE, True, None, (RATE, 3 * RATE))
    long = plan_denoise(100 * RATE, RATE, (RATE, 80 * RATE), STATIONARY)
    assert long.preview == (RATE, 31 * RATE)  # a long selection previews its first 30 s


def test_plan_selection_as_noise_profile_processes_the_whole_clip():
    plan = plan_denoise(5 * RATE, RATE, (0, RATE // 2), STATIONARY, True)
    assert plan == DenoisePlan(0, 5 * RATE, False, (0, RATE // 2), (0, 5 * RATE))
    # Only the stationary gate uses a profile.
    for settings in (NON_STATIONARY, DenoiseSettings(strength=0.0)):
        plan = plan_denoise(5 * RATE, RATE, (0, RATE // 2), settings, True)
        assert plan.selected and plan.profile is None


def test_plan_errors():
    with pytest.raises(ValueError, match="Select a stretch of noise"):
        plan_denoise(5 * RATE, RATE, None, STATIONARY, True)
    with pytest.raises(ValueError, match="noise selection is too short"):
        plan_denoise(5 * RATE, RATE, (0, MIN_FRAMES - 1), STATIONARY, True)
    with pytest.raises(ValueError, match="selection is too short"):
        plan_denoise(5 * RATE, RATE, (0, MIN_FRAMES - 1), STATIONARY)
    with pytest.raises(ValueError, match="clip is too short"):
        plan_denoise(MIN_FRAMES - 1, RATE, None, STATIONARY)


# ------------------------------------------------------------------------- dialog


@pytest.fixture
def runner(qtbot):
    r = SerialTaskRunner(name="test-loader")
    yield r
    r.drain()


@pytest.fixture
def preview_backend():
    return FakeBackend()


@pytest.fixture
def make_dialog(qtbot, runner, preview_backend):
    def make(selection=None, settings=None, profile=False, samples=NOISY):
        dialog = DenoiseDialog(
            AudioClip(samples, RATE),
            selection,
            settings,
            profile,
            runner=runner,
            player=ClipPlayer(preview_backend),
        )
        qtbot.addWidget(dialog)
        return dialog

    return make


def test_dialog_shows_the_settings_and_returns_them(make_dialog):
    settings = DenoiseSettings(
        mode=NoiseMode.NON_STATIONARY, strength=0.6, highpass_hz=120.0, normalize=True
    )
    dialog = make_dialog(settings=settings)
    assert dialog.nonstationary_radio.isChecked()
    assert dialog.strength_slider.value() == 60 and dialog.strength_label.text() == "60 %"
    assert dialog.highpass_check.isChecked() and dialog.highpass_spin.value() == 120
    assert dialog.normalize_check.isChecked()
    assert dialog.request() == DenoiseRequest(settings, False)
    dialog.stationary_radio.setChecked(True)
    dialog.highpass_check.setChecked(False)
    assert not dialog.highpass_spin.isEnabled()
    assert dialog.settings() == DenoiseSettings(strength=0.6, highpass_hz=None, normalize=True)


def test_dialog_target_and_profile_option(make_dialog):
    dialog = make_dialog()
    assert not dialog.profile_check.isEnabled()  # no selection
    assert dialog.target_label.text() == "Applies to the whole clip (00:03.0)."
    dialog = make_dialog(selection=(RATE // 2, RATE))
    assert dialog.profile_check.isEnabled()
    assert dialog.target_label.text().startswith("Applies to the selection (00:00.5 - 00:01.0")
    dialog.profile_check.setChecked(True)
    assert dialog.request().profile_from_selection
    assert dialog.target_label.text() == "Applies to the whole clip (00:03.0)."
    assert "the whole clip is cleaned" in dialog.profile_hint.text()
    dialog.nonstationary_radio.setChecked(True)  # no profile in this mode
    assert not dialog.profile_check.isEnabled()
    assert not dialog.request().profile_from_selection
    assert dialog.target_label.text().startswith("Applies to the selection")
    dialog.stationary_radio.setChecked(True)
    dialog.strength_slider.setValue(0)
    assert not dialog.profile_check.isEnabled()


def test_dialog_profile_option_restored_only_with_a_selection(make_dialog):
    assert not make_dialog(profile=True).request().profile_from_selection
    assert make_dialog(selection=(0, RATE), profile=True).request().profile_from_selection


def test_dialog_apply_disabled_for_noop_or_too_short(make_dialog):
    dialog = make_dialog()
    assert dialog.apply_button.isEnabled() and dialog.preview_button.isEnabled()
    dialog.strength_slider.setValue(0)
    dialog.highpass_check.setChecked(False)
    assert not dialog.apply_button.isEnabled() and not dialog.preview_button.isEnabled()
    assert dialog.message_label.text() == "These settings change nothing."
    dialog.normalize_check.setChecked(True)
    assert dialog.apply_button.isEnabled()
    short = make_dialog(selection=(0, 100))
    assert not short.apply_button.isEnabled()
    assert "too short" in short.message_label.text()


def test_dialog_labels_use_safe_glyphs(make_dialog):
    dialog = make_dialog(selection=(0, RATE))
    texts = [dialog.target_label.text(), dialog.profile_hint.text(), dialog.preview_label.text()]
    texts += [b.text() for b in (dialog.preview_button, dialog.original_button)]
    assert all(t.isascii() or t.endswith("…") or "…" in t for t in texts)


def test_preview_processes_and_plays_like_apply(make_dialog, qtbot, preview_backend):
    dialog = make_dialog()
    assert dialog.original_button.isEnabled() and not dialog.processed_button.isEnabled()
    qtbot.mouseClick(dialog.preview_button, Qt.MouseButton.LeftButton)
    assert dialog.job is not None and dialog.preview_button.text() == "Previewing…"
    qtbot.waitUntil(lambda: dialog.playing == PROCESSED, timeout=TIMEOUT)
    assert dialog.job is None and dialog.preview_up_to_date
    assert dialog.player.loop and preview_backend.stream.started
    # The preview (the first 10 s: here the whole clip) is exactly what Apply produces.
    applied = denoise(NOISY, RATE, STATIONARY)
    same(dialog.player.clip.samples, applied)
    assert "Playing the processed" in dialog.preview_label.text()


def test_preview_ab_switch_keeps_the_position(make_dialog, qtbot, preview_backend):
    dialog = make_dialog()
    dialog.play(ORIGINAL)
    assert dialog.playing == ORIGINAL
    same(dialog.player.clip.samples, NOISY)
    stream = preview_backend.stream
    stream.pump(4800)
    stream.time += 0.5
    stream.pump(4800)
    position = dialog.player.position
    assert position > 0.05
    dialog.start_preview()
    qtbot.waitUntil(lambda: dialog.processed_button.isEnabled(), timeout=TIMEOUT)
    assert dialog.playing == ORIGINAL  # a finished preview doesn't interrupt the original
    qtbot.mouseClick(dialog.processed_button, Qt.MouseButton.LeftButton)
    assert dialog.playing == PROCESSED
    # Swapped under the running stream: no reopen, and the next buffer continues where the
    # original left off, from the processed audio.
    assert len(preview_backend.streams) == 1 and not stream.closed
    out = stream.pump(480)
    same(out, np.clip(dialog.player.clip.samples[9600:10080], -1, 1))
    assert dialog.player.position >= position
    qtbot.mouseClick(dialog.stop_button, Qt.MouseButton.LeftButton)
    assert dialog.playing is None and not dialog.stop_button.isEnabled()


def test_changed_settings_mark_the_preview_out_of_date(make_dialog, qtbot):
    dialog = make_dialog()
    dialog.start_preview()
    qtbot.waitUntil(lambda: dialog.preview_up_to_date, timeout=TIMEOUT)
    dialog.strength_slider.setValue(50)
    assert not dialog.preview_up_to_date
    assert dialog.processed_button.isEnabled()  # the old preview still plays
    assert "press Preview" in dialog.preview_label.text()
    dialog.strength_slider.setValue(90)
    assert dialog.preview_up_to_date


def test_changing_the_preview_part_drops_the_old_preview(make_dialog, qtbot):
    dialog = make_dialog(selection=(RATE // 2, 2 * RATE))
    dialog.start_preview()
    qtbot.waitUntil(lambda: dialog.playing == PROCESSED, timeout=TIMEOUT)
    assert dialog.player.clip.frames == round(1.5 * RATE)
    dialog.profile_check.setChecked(True)  # the whole clip: a different preview part
    assert dialog.playing is None and not dialog.processed_button.isEnabled()


def test_a_new_preview_cancels_the_running_one(make_dialog, runner, qtbot):
    dialog = make_dialog()
    gate = threading.Event()
    runner.submit(lambda: gate.wait(5))
    dialog.start_preview()
    first = dialog.job
    dialog.start_preview()
    assert first.is_cancelled() and dialog.job is not first
    gate.set()
    qtbot.waitUntil(lambda: dialog.playing == PROCESSED, timeout=TIMEOUT)


def test_closing_cancels_the_preview_and_closes_the_player(make_dialog, runner, qtbot):
    dialog = make_dialog()
    dialog.play(ORIGINAL)
    gate = threading.Event()
    runner.submit(lambda: gate.wait(5))
    dialog.start_preview()
    job = dialog.job
    dialog.reject()
    assert job.is_cancelled() and dialog.playing is None
    assert dialog.player.clip is not None and not dialog.player.is_playing
    gate.set()
    assert runner.drain()  # the cancelled preview finishes quickly; its result is ignored
    dialog.play(ORIGINAL)
    assert dialog.playing is None


def test_preview_failure_is_shown(make_dialog, qtbot, monkeypatch):
    from desktop_audio_recorder.ui import denoise_dialog

    def boom(*args, **kwargs):
        raise RuntimeError("kaputt")

    monkeypatch.setattr(denoise_dialog, "denoise", boom)
    dialog = make_dialog()
    dialog.start_preview()
    qtbot.waitUntil(lambda: dialog.job is None, timeout=TIMEOUT)
    assert dialog.message_label.text() == "Preview failed: kaputt"
    assert dialog.preview_button.isEnabled()


# ------------------------------------------------------------------------- main window


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def window(qtbot, tmp_path, backend):
    recorder = FakeRecorder(tmp_path)
    recorder.samples = NOISY
    w = mw.MainWindow(
        recorder=recorder,
        device_manager=FakeDeviceManager(),
        clock=Clock(),
        player=ClipPlayer(backend),
        preview_player=lambda: ClipPlayer(FakeBackend()),
    )
    qtbot.addWidget(w)
    qtbot.waitUntil(lambda: w.device is not None, timeout=TIMEOUT)
    qtbot.mouseClick(w.transport.record_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: w.state is AppState.RECORDING, timeout=TIMEOUT)
    qtbot.mouseClick(w.transport.stop_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: w.state is AppState.STOPPED, timeout=TIMEOUT)
    same(w.clip.samples, NOISY)
    return w


def hold_loader(window):
    gate = threading.Event()
    window._loader.submit(lambda: gate.wait(5))
    return gate


def select(window, start, end):
    window.waveform.set_selection((start, end), emit=True)


def run_denoise(window, qtbot, request):
    window.reduce_noise(request)
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)


def test_toolbar_action_opens_the_prompt(window):
    asked = []
    window.prompt_denoise = lambda: asked.append(1)  # cancelled
    window.edit_toolbar.denoise_action.trigger()
    assert asked == [1] and window.state is AppState.STOPPED
    same(window.clip.samples, NOISY)


def test_denoise_whole_clip_is_cleaner_and_undoable(window, qtbot):
    window.prompt_denoise = lambda: DenoiseRequest(STATIONARY)
    gate = hold_loader(window)
    window.edit_toolbar.denoise_action.trigger()
    assert window.state is AppState.PROCESSING
    assert window.transport.state_label.text() == "Reducing noise…"
    assert window.progress_bar.isVisibleTo(window) and window.cancel_button.isVisibleTo(window)
    assert not any(a.isEnabled() for a in window.edit_toolbar.edit_actions.values())
    assert not window.playback.play_button.isEnabled()
    assert not window.transport.record_button.isEnabled()
    assert window.status_text == "Reducing noise (stationary, 90 %, high-pass 80 Hz)…"
    gate.set()
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)
    assert not window.progress_bar.isVisibleTo(window)
    assert not window.cancel_button.isVisibleTo(window)
    assert window.job is None
    assert window.status_text.startswith("Noise reduction (stationary, 90 %, high-pass 80 Hz):")
    assert snr_db(CLEAN, window.clip.samples) > snr_db(CLEAN, NOISY) + 3
    assert window.player.clip is window.clip and window.waveform.clip is window.clip
    assert window.edit_toolbar.undo_action.toolTip() == "Undo: Noise reduction (Ctrl+Z)"
    assert window.waveform.selection is None
    processed = window.clip.samples.copy()
    window.undo()
    same(window.clip.samples, NOISY)
    assert window.status_text == "Undid: Noise reduction"
    window.redo()
    same(window.clip.samples, processed)


def test_denoise_selection_only(window, qtbot):
    select(window, 1.0, 2.0)
    run_denoise(window, qtbot, DenoiseRequest(STATIONARY))
    samples = window.clip.samples
    same(samples[:RATE], NOISY[:RATE])
    same(samples[2 * RATE :], NOISY[2 * RATE :])
    assert not np.array_equal(samples[RATE : 2 * RATE], NOISY[RATE : 2 * RATE])
    assert window.waveform.selection == pytest.approx((1.0, 2.0))  # still selected


def test_selection_as_noise_profile_processes_the_whole_clip(window, qtbot):
    select(window, 0.0, 0.25)  # before the first "word": noise only
    run_denoise(window, qtbot, DenoiseRequest(STATIONARY, profile_from_selection=True))
    expected = denoise(NOISY, RATE, STATIONARY, noise_profile=NOISY[: RATE // 4])
    same(window.clip.samples, expected)
    assert window.denoise_profile_from_selection is False  # given directly, not remembered


def test_dialog_choices_are_remembered(window, qtbot):
    settings = DenoiseSettings(strength=0.5)
    select(window, 0.0, 0.25)
    window.prompt_denoise = lambda: DenoiseRequest(settings, True)
    window.reduce_noise()
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)
    assert window.denoise_settings == settings and window.denoise_profile_from_selection


def test_cancel_leaves_clip_and_history_unchanged(window, qtbot):
    window.apply_gain(-3.0)
    window.undo()  # something to redo, which a cancelled run must not drop
    gate = hold_loader(window)
    window.reduce_noise(DenoiseRequest(STATIONARY))
    assert window.state is AppState.PROCESSING
    qtbot.mouseClick(window.cancel_button, Qt.MouseButton.LeftButton)
    assert window.job.is_cancelled()
    assert window.status_text == "Cancelling…" and not window.cancel_button.isEnabled()
    gate.set()
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)
    same(window.clip.samples, NOISY)
    assert window.status_text == "Noise reduction cancelled; the clip is unchanged"
    assert not window.status_is_error
    assert window.session.history.redo_label == "Gain -3.0 dB"
    assert not window.session.history.can_undo


def test_cancel_midway(window, qtbot, monkeypatch):
    """Cancel while denoise is running: it stops at its next chunk."""
    started, release = threading.Event(), threading.Event()
    real = mw.denoise

    def slow(samples, rate, settings, profile, progress, cancelled):
        def report(fraction):
            progress(fraction)
            if fraction > 0:
                started.set()
                release.wait(5)

        return real(samples, rate, settings, profile, report, cancelled, chunk_seconds=0.5)

    monkeypatch.setattr(mw, "denoise", slow)
    window.reduce_noise(DenoiseRequest(STATIONARY))
    assert started.wait(5)
    qtbot.waitUntil(lambda: window.progress_bar.value() > 0, timeout=TIMEOUT)
    window.cancel_processing()
    release.set()
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)
    same(window.clip.samples, NOISY)
    assert not window.session.history.can_undo


def test_progress_bar_follows_the_job(window, qtbot, monkeypatch):
    halfway, release = threading.Event(), threading.Event()

    def fake(samples, rate, settings, profile, progress, cancelled):
        progress(0.5)
        halfway.set()
        release.wait(5)
        return samples.copy()

    monkeypatch.setattr(mw, "denoise", fake)
    window.reduce_noise(DenoiseRequest(STATIONARY))
    assert halfway.wait(5)
    qtbot.waitUntil(lambda: window.progress_bar.value() == 50, timeout=TIMEOUT)
    release.set()
    qtbot.waitUntil(lambda: window.state is AppState.STOPPED, timeout=TIMEOUT)


def test_failure_shows_an_error_and_keeps_the_clip(window, qtbot, monkeypatch):
    def boom(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr(mw, "denoise", boom)
    run_denoise(window, qtbot, DenoiseRequest(STATIONARY))
    assert window.status_is_error
    assert window.status_text == "Could not reduce noise: not enough memory"
    same(window.clip.samples, NOISY)


def test_noop_and_invalid_requests_do_nothing(window, qtbot):
    window.reduce_noise(DenoiseRequest(DenoiseSettings(strength=0.0, highpass_hz=None)))
    assert window.state is AppState.STOPPED
    assert window.status_text == "Noise reduction: these settings change nothing"
    window.reduce_noise(DenoiseRequest(STATIONARY, profile_from_selection=True))  # no selection
    assert window.state is AppState.STOPPED
    assert window.status_text.startswith("Select a stretch of noise")
    same(window.clip.samples, NOISY)


def test_denoise_while_playing_pauses_first(window, qtbot, backend):
    qtbot.mouseClick(window.playback.play_button, Qt.MouseButton.LeftButton)
    assert window.state is AppState.PLAYING
    backend.stream.pump(4800)
    run_denoise(window, qtbot, DenoiseRequest(STATIONARY))
    assert backend.stream.closed and not window.player.is_playing
    assert window.session.history.undo_label == "Noise reduction"


def test_close_during_processing_cancels_it(window, qtbot):
    gate = hold_loader(window)
    window.reduce_noise(DenoiseRequest(STATIONARY))
    job = window.job
    gate.set()
    window.close()
    assert job.is_cancelled()
    assert not window._loader.busy


def test_progress_job():
    job = ProgressJob()
    assert job.fraction == 0.0 and not job.is_cancelled()
    job.report(1.5)
    assert job.fraction == 1.0
    job.report(-1)
    assert job.fraction == 0.0
    job.cancel()
    assert job.is_cancelled()
