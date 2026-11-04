"""Application bootstrap: logging, global exception dialog, QApplication.

Logs go to ``%LOCALAPPDATA%\\DesktopAudioRecorder\\logs\\app.log`` (rotating, 4 x 1 MB) and,
when there is a console, to stderr. A native crash (access violation in a DLL) is dumped by
``faulthandler`` into ``crash.log`` in the same folder.
"""

from __future__ import annotations

import argparse
import faulthandler
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from desktop_audio_recorder import APP_NAME, __version__

log = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_crash_file = None  # kept open for faulthandler
# The window and taskbar icon (the exe embeds the same file; packaging/make_icon.py draws it).
ICON_PATH = Path(__file__).resolve().parent / "resources" / "app.ico"
# Groups the app's windows on the taskbar under its own icon (not python.exe's) when it runs
# from source; the exe would get its own anyway.
APP_USER_MODEL_ID = "DesktopAudioRecorder.App"


def set_app_user_model_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except (AttributeError, OSError):
        log.debug("Could not set the AppUserModelID", exc_info=True)


def log_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "DesktopAudioRecorder" / "logs"


def setup_logging(folder: Path | None = None) -> Path | None:
    """Log to the rotating file (and to stderr when there is one). Returns the log folder, or
    ``None`` if it couldn't be created."""
    global _crash_file
    fmt = logging.Formatter(LOG_FORMAT)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if sys.stderr is not None:  # a windowed exe has no console
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)
    folder = folder or log_dir()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            folder / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
    except OSError:
        log.warning("File logging unavailable", exc_info=True)
        return None
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    try:
        _crash_file = open(folder / "crash.log", "a", encoding="utf-8")  # noqa: SIM115
        faulthandler.enable(_crash_file)
    except OSError:
        log.warning("Crash dumps unavailable", exc_info=True)
    return folder


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="desktop-audio-recorder", description=APP_NAME)
    parser.add_argument(
        "--smoke-test",
        metavar="FOLDER",
        type=Path,
        help="record, play and export a test tone into FOLDER, write report.json there, quit",
    )
    parser.add_argument(
        "file", nargs="?", type=Path, help="an audio file to open (WAV, MP3, FLAC, OGG, AIFF)"
    )
    args, _qt_args = parser.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = _parse_args(argv)
    folder = setup_logging()
    try:
        return _run(args, argv, folder)
    except Exception:
        # Before the error dialog exists (e.g. a module missing from a build): at least log it.
        log.critical("The app failed to start", exc_info=True)
        raise


def _run(args: argparse.Namespace, argv: list[str], folder: Path | None) -> int:
    log.info(
        "Starting %s %s (Python %s%s)",
        APP_NAME,
        __version__,
        sys.version.split()[0],
        ", frozen" if getattr(sys, "frozen", False) else "",
    )

    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from desktop_audio_recorder.ui.error_reporter import ErrorReporter
    from desktop_audio_recorder.ui.main_window import MainWindow

    set_app_user_model_id()
    app = QApplication([sys.argv[0], *argv])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("DesktopAudioRecorder")
    app.setApplicationVersion(__version__)
    icon = QIcon(str(ICON_PATH))
    if icon.isNull():
        log.warning("The app icon %s is missing", ICON_PATH)
    else:
        app.setWindowIcon(icon)

    reporter = ErrorReporter(folder, app)
    reporter.install()
    app.aboutToQuit.connect(reporter.begin_shutdown)

    settings = None
    if args.smoke_test is not None:  # never the user's settings (the registry)
        from PySide6.QtCore import QSettings

        from desktop_audio_recorder.settings import Settings

        args.smoke_test.mkdir(parents=True, exist_ok=True)
        ini = str(args.smoke_test / "settings.ini")
        settings = Settings(QSettings(ini, QSettings.Format.IniFormat))

    window = MainWindow(settings=settings)
    # closeEvent covers the normal path; this covers app.quit() and session end.
    app.aboutToQuit.connect(window.shutdown)
    app.commitDataRequest.connect(window.commit_session_data)
    window.show()
    if args.smoke_test is not None:
        from desktop_audio_recorder.smoke_test import SmokeTest

        smoke = SmokeTest(window, args.smoke_test)
        smoke.start()
    else:
        if args.file is not None:
            if args.file.is_file():
                window.open_file(args.file.resolve())
            else:  # e.g. the value of a Qt option argparse doesn't know
                log.warning("Not opening %s: no such file", args.file)
        window.offer_recovery_later()
    code = app.exec()
    if args.smoke_test is not None:
        code = smoke.exit_code
    reporter.uninstall()
    log.info("Exiting with code %d", code)
    return code
