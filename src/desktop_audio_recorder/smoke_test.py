"""``desktop-audio-recorder --smoke-test FOLDER``: an end-to-end check of a build.

It drives the real window on the real default playback device: plays a quiet 1 kHz tone (through
sounddevice) while recording 2 s, checks the recording's length and that the tone is in it,
plays the clip back for a second, exports it as 16-bit WAV and 192 kbps MP3 and reads both back
(soundfile: libsndfile + mpg123), opens Help > About (in a build, LICENSE and
THIRD-PARTY-NOTICES.txt must be next to the exe), checks that the log file is written, saves
screenshots, writes
``FOLDER/report.json`` and quits (exit code 0 when every check passed).

Settings go to ``FOLDER/settings.ini``, never the registry (``app.main`` passes them in). It is
meant for checking a PyInstaller build: every native library the app needs (both PortAudio DLLs,
libsndfile, LAME, Qt plugins) is exercised.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QApplication

from desktop_audio_recorder import __version__
from desktop_audio_recorder.export.writers import ExportFormat, ExportSettings
from desktop_audio_recorder.ui.export_dialog import ExportRequest, UnsavedChoice
from desktop_audio_recorder.ui.state import AppState

log = logging.getLogger(__name__)

TONE_HZ = 1000.0
TONE_AMPLITUDE = 0.05  # -26 dBFS
RECORD_SECONDS = 2.0

Step = tuple[Callable[[], bool], float, str]  # (done?, timeout seconds, what)


def tone_prominence_db(samples: np.ndarray, rate: int, hz: float = TONE_HZ) -> float:
    """Power at ``hz`` over the median power of the band around it (other apps may be
    playing too, so the overall dominant frequency is not a reliable test)."""
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    if len(mono) < rate // 10:
        return 0.0
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2
    freqs = np.fft.rfftfreq(len(mono), 1 / rate)
    at = (freqs > hz - 5) & (freqs < hz + 5)
    around = (freqs > hz - 300) & (freqs < hz + 300) & ~((freqs > hz - 20) & (freqs < hz + 20))
    floor = np.median(spectrum[around]) + 1e-20
    return float(10 * np.log10(spectrum[at].max() / floor))


class SmokeTest(QObject):
    def __init__(self, window, folder: Path) -> None:
        super().__init__(window)
        self.window = window
        self.folder = Path(folder)
        self.exit_code = 1
        self.report: dict = {
            "version": __version__,
            "frozen": bool(getattr(sys, "frozen", False)),
            "checks": {},
            "error": None,
        }
        self._steps: Iterator[Step] = self._run()
        self._waiting: tuple[Callable[[], bool], float, str] | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._advance)
        self._started = time.perf_counter()
        # Never block on a dialog.
        window.prompt_unsaved = lambda prompt: UnsavedChoice.DISCARD
        window.prompt_export = lambda: None

    def start(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        log.info("Smoke test writing to %s", self.folder)
        self._timer.start()

    # ------------------------------------------------------------------ driver

    def _advance(self) -> None:
        try:
            if self._waiting is not None:
                done, deadline, what = self._waiting
                if done():
                    self._waiting = None
                elif time.perf_counter() > deadline:
                    raise TimeoutError(f"timed out waiting for {what}")
                else:
                    return
            done, timeout, what = next(self._steps)
            self._waiting = (done, time.perf_counter() + timeout, what)
        except StopIteration:
            self._finish()
        except Exception as exc:
            log.exception("Smoke test failed")
            self.report["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self._finish()

    def check(self, check: str, ok: bool, **details) -> None:
        self.report["checks"][check] = {"ok": bool(ok), **details}
        log.info("Smoke check %s: %s %s", check, "ok" if ok else "FAILED", details)

    @staticmethod
    def sleep(seconds: float) -> Step:
        until = time.perf_counter() + seconds
        return (lambda: time.perf_counter() >= until), seconds + 5, f"{seconds} s"

    def state_is(self, state: AppState, timeout: float = 10.0) -> Step:
        return (lambda: self.window.state is state), timeout, f"state {state.value}"

    # ------------------------------------------------------------------ the test

    def _run(self) -> Iterator[Step]:
        import sounddevice as sd
        import soundfile as sf

        window = self.window
        yield (lambda: window.device is not None or window.status_is_error), 10, "the device"
        self.check(
            "device",
            window.device is not None,
            name=window.device.output_name if window.device else window.status_text,
        )
        if window.device is None:
            return

        # Record a tone.
        rate = 48000
        t = np.arange(int(rate * (RECORD_SECONDS + 2.0))) / rate
        tone = (TONE_AMPLITUDE * np.sin(2 * np.pi * TONE_HZ * t)).astype(np.float32)
        sd.play(np.column_stack([tone, tone]), rate)
        yield self.sleep(0.3)
        window.start_recording()
        yield self.state_is(AppState.RECORDING)
        yield self.sleep(RECORD_SECONDS)
        window.stop_recording()
        yield self.state_is(AppState.STOPPED, 20)
        sd.stop()
        clip = window.clip
        prominence = tone_prominence_db(clip.samples, clip.sample_rate)
        self.check(
            "record",
            abs(clip.duration - RECORD_SECONDS) < 0.15 and prominence > 20,
            duration=round(clip.duration, 3),
            rate=clip.sample_rate,
            tone_prominence_db=round(prominence, 1),
            overflows=window.result.input_overflows,
        )
        window.grab().save(str(self.folder / "window.png"))

        # Play a second of it.
        played_from = time.perf_counter()
        window.play()
        yield (lambda: window.player.streaming or window.state is not AppState.PLAYING), 5, "play"
        opened_ms = (time.perf_counter() - played_from) * 1000
        yield self.sleep(1.0)
        position = window.player.position
        error = window.player.last_error
        window.pause_playback()
        self.check(
            "play",
            error is None and 0.7 < position < 1.3,
            stream_open_ms=round(opened_ms),
            position=round(position, 3),
            device=window.player.device_name,
            underflows=window.player.underflows,
            error=error,
        )

        # Export both formats and read them back.
        for name, settings in (
            ("smoke.wav", ExportSettings(ExportFormat.WAV, "PCM_16")),
            ("smoke.mp3", ExportSettings(ExportFormat.MP3, mp3_bitrate=192)),
        ):
            path = self.folder / name
            path.unlink(missing_ok=True)
            window.export_clip(ExportRequest(path, settings))  # enters EXPORTING at once
            yield self.state_is(AppState.STOPPED, 60)
            ok, details = path.exists(), {"status": window.status_text}
            if ok:
                data, file_rate = sf.read(str(path), dtype="float32", always_2d=True)
                details.update(
                    frames=len(data),
                    clip_frames=clip.frames,
                    rate=file_rate,
                    bytes=path.stat().st_size,
                    tone_prominence_db=round(tone_prominence_db(data, file_rate), 1),
                )
                ok = len(data) == clip.frames and file_rate == clip.sample_rate
            self.check(f"export {settings.format.value}", ok, **details)
        self.check("saved", not window.has_unsaved_changes)

        # Help > About: in a release build, LICENSE and the notices are next to the exe.
        from desktop_audio_recorder.ui import about

        window.show_about()
        yield (lambda: window.about_box is not None and window.about_box.isVisible()), 5, "About"
        box = window.about_box
        box.grab().save(str(self.folder / "about.png"))
        buttons = [b.text() for b in box.buttons()]
        notices, license_file = about.notices_path(), about.license_path()
        notices_ok = notices is not None and notices.read_text(encoding="utf-8").startswith(
            f"Desktop Audio Recorder {__version__}: third-party notices"
        )
        frozen = bool(getattr(sys, "frozen", False))
        self.check(
            "about",
            f"{__version__}</h3>" in box.text()
            and (not frozen or (notices_ok and license_file is not None))
            and ("Third-party notices" in buttons) == (notices is not None),
            frozen=frozen,
            notices=str(notices) if notices else None,
            license=str(license_file) if license_file else None,
            buttons=buttons,
        )
        box.accept()

        # The log.
        from desktop_audio_recorder.app import log_dir

        for handler in logging.getLogger().handlers:
            handler.flush()
        log_file = log_dir() / "app.log"
        text = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""
        self.check(
            "log",
            "Starting Desktop Audio Recorder" in text and "Smoke check export mp3" in text,
            path=str(log_file),
        )

    def _finish(self) -> None:
        self._timer.stop()
        self.report["seconds"] = round(time.perf_counter() - self._started, 2)
        checks = self.report["checks"]
        passed = self.report["error"] is None and checks and all(c["ok"] for c in checks.values())
        self.report["passed"] = bool(passed)
        self.exit_code = 0 if passed else 1
        (self.folder / "report.json").write_text(json.dumps(self.report, indent=2), "utf-8")
        log.info("Smoke test %s", "passed" if passed else "FAILED")
        QApplication.quit()
