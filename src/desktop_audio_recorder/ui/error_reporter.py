"""The global exception dialog.

``ErrorReporter.install()`` hooks ``sys.excepthook`` (uncaught exceptions on the main thread,
which in PySide6 includes exceptions raised in slots, timer callbacks, ``SerialTaskRunner``
callbacks and Python overrides such as ``paintEvent``) and ``threading.excepthook`` (plain Python
threads). Every exception is logged with its traceback. The dialog:

* is shown on the GUI thread only: a report from another thread is queued to it;
* is **non-modal** (``show()``, not ``exec()``), so there is no nested event loop: the code that
  raised has unwound, nothing re-enters it, and a quit or logoff isn't held up by an open dialog;
* is shown once per distinct error (type, message, place): an error that repeats on every
  repaint doesn't bring up a dialog after every dismissal;
* while one is open, further errors are only counted (the dialog says how many);
* at most ``MAX_DIALOGS`` per session; after that, errors are only logged;
* never during shutdown (``begin_shutdown()``, or ``QCoreApplication.closingDown()``).
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
import traceback
from collections.abc import Callable
from pathlib import Path
from types import TracebackType

from PySide6.QtCore import QCoreApplication, QObject, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from desktop_audio_recorder import APP_NAME

log = logging.getLogger(__name__)

MAX_DIALOGS = 3

ExcInfo = tuple[type[BaseException], BaseException, TracebackType | None]


def error_key(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None):
    """What makes two errors "the same": type, message and the innermost frame."""
    where = None
    if tb is not None:
        frames = traceback.extract_tb(tb)
        if frames:
            last = frames[-1]
            where = (last.filename, last.lineno)
    return exc_type.__qualname__, str(exc), where


class ErrorReporter(QObject):
    """See the module docstring. ``show_dialog(summary, details)`` is replaceable (tests)."""

    _report = Signal(str, str, object)  # summary, details, key (queued to the GUI thread)

    def __init__(self, log_dir: Path | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.log_dir = log_dir
        self.shown: list[str] = []  # summaries of the dialogs shown
        self.suppressed = 0  # errors logged but not shown
        self._seen: set = set()
        self._dialog: QMessageBox | None = None
        self._pending_extra = 0
        self._shutting_down = False
        self._previous_hooks: tuple | None = None
        self.show_dialog: Callable[[str, str], None] = self._show_message_box
        self._report.connect(self._on_report, Qt.ConnectionType.QueuedConnection)

    # ------------------------------------------------------------------ install

    def install(self) -> None:
        self._previous_hooks = (sys.excepthook, threading.excepthook)
        sys.excepthook = self.excepthook
        threading.excepthook = self.thread_excepthook

    def uninstall(self) -> None:
        if self._previous_hooks is not None:
            sys.excepthook, threading.excepthook = self._previous_hooks
            self._previous_hooks = None

    def begin_shutdown(self) -> None:
        """From now on errors are only logged (connected to ``aboutToQuit``)."""
        self._shutting_down = True

    # ------------------------------------------------------------------ hooks (any thread)

    def excepthook(self, exc_type, exc, tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        self.report(exc_type, exc, tb)

    def thread_excepthook(self, args: threading.ExceptHookArgs) -> None:
        if args.exc_type is SystemExit:
            return
        thread = args.thread.name if args.thread is not None else "?"
        self.report(args.exc_type, args.exc_value, args.exc_traceback, thread=thread)

    def report(
        self,
        exc_type: type[BaseException],
        exc: BaseException | None,
        tb: TracebackType | None,
        *,
        thread: str | None = None,
    ) -> None:
        """Log the exception and (maybe) show the dialog. Safe to call from any thread."""
        if exc is None:
            exc = exc_type()
        where = f" in thread {thread}" if thread else ""
        with contextlib.suppress(Exception):  # never raise from a hook, even if logging fails
            log.critical("Unhandled exception%s", where, exc_info=(exc_type, exc, tb))
        try:
            summary = "".join(traceback.format_exception_only(exc_type, exc)).strip()
            details = "".join(traceback.format_exception(exc_type, exc, tb))
            key = error_key(exc_type, exc, tb)
        except Exception:
            summary, details, key = repr(exc), "", None
        if self._quiet():
            self.suppressed += 1
            return
        app = QCoreApplication.instance()
        if app is not None and QThread.currentThread() is app.thread():
            self._on_report(summary, details, key)
        else:
            with contextlib.suppress(RuntimeError):  # the reporter was deleted (shutting down)
                self._report.emit(summary, details, key)

    # ------------------------------------------------------------------ GUI thread

    def _quiet(self) -> bool:
        return (
            self._shutting_down
            or not isinstance(QCoreApplication.instance(), QApplication)
            or QCoreApplication.closingDown()
        )

    @Slot(str, str, object)
    def _on_report(self, summary: str, details: str, key) -> None:
        if self._quiet():
            self.suppressed += 1
            return
        if self._dialog is not None and self._dialog.isVisible():
            self._pending_extra += 1  # one dialog at a time
            self.suppressed += 1
            self._dialog.setInformativeText(self._informative(self._pending_extra))
            return
        if key is not None and key in self._seen:
            self.suppressed += 1
            return
        if len(self.shown) >= MAX_DIALOGS:
            self.suppressed += 1
            return
        if key is not None:
            self._seen.add(key)
        self._pending_extra = 0
        self.shown.append(summary)
        try:
            self.show_dialog(summary, details)
        except Exception:
            log.exception("Could not show the error dialog")

    def _informative(self, extra: int) -> str:
        text = "The app may not work correctly until you restart it."
        if self.log_dir is not None:
            text += f"\nDetails were written to the log in {self.log_dir}."
        if extra:
            text += f"\n\n{extra} more error(s) occurred since (see the log)."
        return text

    def _show_message_box(self, summary: str, details: str) -> None:
        parent = QApplication.activeWindow()
        box = QMessageBox(parent if isinstance(parent, QWidget) else None)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle(f"{APP_NAME} - Error")
        box.setText(f"An unexpected error occurred:\n\n{summary}")
        box.setInformativeText(self._informative(0))
        box.setDetailedText(details)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        if self.log_dir is not None:
            open_logs = box.addButton("Open log folder", QMessageBox.ButtonRole.ActionRole)
            open_logs.clicked.connect(self._open_log_folder)
        box.setWindowModality(Qt.WindowModality.NonModal)
        box.finished.connect(self._on_dialog_finished)
        self._dialog = box
        box.show()

    @Slot(int)
    def _on_dialog_finished(self, _result: int) -> None:
        self._dialog = None

    def _open_log_folder(self) -> None:
        if self.log_dir is None:
            return
        if sys.platform == "win32":
            os.startfile(self.log_dir)  # noqa: S606 - opens Explorer on our own folder
        else:  # pragma: no cover
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.log_dir)))
