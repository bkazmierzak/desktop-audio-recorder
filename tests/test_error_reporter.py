"""The global exception dialog (ui/error_reporter.py) and the logging set-up (app.py)."""

import logging
import sys
import threading

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

from desktop_audio_recorder import app as app_module
from desktop_audio_recorder.ui import error_reporter as er
from desktop_audio_recorder.ui.error_reporter import ErrorReporter
from desktop_audio_recorder.ui.workers import SerialTaskRunner

pytestmark = pytest.mark.qt_no_exception_capture


@pytest.fixture
def reporter(qapp, tmp_path):
    r = ErrorReporter(tmp_path)
    r.dialogs = []

    def fake_dialog(summary, details):
        r.dialogs.append((summary, details, threading.current_thread() is threading.main_thread()))

    r.show_dialog = fake_dialog
    r.install()
    yield r
    r.uninstall()
    r.deleteLater()


def raise_(exc):
    raise exc


def test_install_and_uninstall_restore_the_hooks(qapp, tmp_path):
    before = (sys.excepthook, threading.excepthook)
    r = ErrorReporter(tmp_path)
    r.install()
    assert sys.excepthook == r.excepthook
    assert threading.excepthook == r.thread_excepthook
    r.uninstall()
    assert (sys.excepthook, threading.excepthook) == before


def test_exception_in_a_timer_slot_shows_the_dialog(reporter, qtbot, caplog):
    QTimer.singleShot(0, lambda: raise_(ValueError("boom in a slot")))
    qtbot.waitUntil(lambda: bool(reporter.dialogs), timeout=2000)
    summary, details, on_gui = reporter.dialogs[0]
    assert summary == "ValueError: boom in a slot"
    assert "Traceback" in details and "raise_" in details
    assert on_gui
    assert any(r.levelno == logging.CRITICAL for r in caplog.records)


def test_exception_in_a_clicked_slot_and_a_virtual_override(reporter, qtbot):
    button = QPushButton()
    qtbot.addWidget(button)
    button.clicked.connect(lambda: raise_(KeyError("clicked")))
    button.click()

    class Broken(QPushButton):
        def paintEvent(self, event):
            raise RuntimeError("paint failed")

    widget = Broken()
    qtbot.addWidget(widget)
    QTimer.singleShot(0, widget.grab)  # a paint from the event loop, not from Python
    qtbot.waitUntil(lambda: len(reporter.dialogs) == 2, timeout=2000)
    assert "KeyError" in reporter.dialogs[0][0]
    assert "paint failed" in reporter.dialogs[1][0]


def test_exception_in_a_runner_callback(reporter, qtbot):
    runner = SerialTaskRunner()
    runner.submit(lambda: 41, lambda result: raise_(ArithmeticError(f"got {result}")))
    qtbot.waitUntil(lambda: bool(reporter.dialogs), timeout=2000)
    assert reporter.dialogs[0][0] == "ArithmeticError: got 41"
    runner.deleteLater()


def test_runner_task_errors_without_handler_are_only_logged(reporter, qtbot, caplog):
    runner = SerialTaskRunner()
    runner.submit(lambda: raise_(OSError("task failed")))
    qtbot.waitUntil(lambda: "Background task failed" in caplog.text, timeout=2000)
    assert reporter.dialogs == []
    runner.deleteLater()


def test_exception_in_a_python_thread_is_shown_on_the_gui_thread(reporter, qtbot):
    thread = threading.Thread(target=lambda: raise_(RuntimeError("worker died")), name="w1")
    thread.start()
    thread.join()
    qtbot.waitUntil(lambda: bool(reporter.dialogs), timeout=2000)
    summary, _, on_gui = reporter.dialogs[0]
    assert summary == "RuntimeError: worker died"
    assert on_gui


def test_a_repeating_error_shows_one_dialog(reporter, qtbot):
    for _ in range(5):
        QTimer.singleShot(0, lambda: raise_(ValueError("same")))
    qtbot.waitUntil(lambda: reporter.suppressed == 4, timeout=2000)
    assert len(reporter.dialogs) == 1


def test_at_most_max_dialogs(reporter, qtbot):
    for i in range(er.MAX_DIALOGS + 2):
        QTimer.singleShot(0, lambda i=i: raise_(ValueError(f"error {i}")))
    qtbot.waitUntil(lambda: reporter.suppressed == 2, timeout=2000)
    assert len(reporter.dialogs) == er.MAX_DIALOGS


def test_nothing_is_shown_while_shutting_down(reporter, qtbot):
    reporter.begin_shutdown()
    QTimer.singleShot(0, lambda: raise_(ValueError("late")))
    qtbot.waitUntil(lambda: reporter.suppressed == 1, timeout=2000)
    assert reporter.dialogs == []


def test_keyboard_interrupt_goes_to_the_default_hook(reporter, monkeypatch):
    called = []
    monkeypatch.setattr(sys, "__excepthook__", lambda *a: called.append(a[0]))
    reporter.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    assert called == [KeyboardInterrupt]
    assert reporter.shown == []


def test_real_dialog_is_non_modal_and_counts_later_errors(qapp, qtbot, tmp_path):
    r = ErrorReporter(tmp_path)
    r.install()
    try:
        QTimer.singleShot(0, lambda: raise_(ValueError("first")))
        qtbot.waitUntil(lambda: r._dialog is not None, timeout=2000)
        box = r._dialog
        assert isinstance(box, QMessageBox)
        assert box.isVisible() and not box.isModal()
        assert "ValueError: first" in box.text()
        assert str(tmp_path) in box.informativeText()
        assert "Traceback" in box.detailedText()
        QTimer.singleShot(0, lambda: raise_(TypeError("second")))
        qtbot.waitUntil(lambda: r.suppressed == 1, timeout=2000)
        assert "1 more error(s)" in box.informativeText()
        assert len(r.shown) == 1
        box.accept()
        qtbot.waitUntil(lambda: r._dialog is None, timeout=2000)
        QTimer.singleShot(0, lambda: raise_(TypeError("third")))
        qtbot.waitUntil(lambda: len(r.shown) == 2, timeout=2000)
        r._dialog.accept()
    finally:
        r.uninstall()


def test_setup_logging_writes_the_log_file(tmp_path):
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        folder = app_module.setup_logging(tmp_path / "logs")
        assert folder == tmp_path / "logs"
        logging.getLogger("probe").warning("hello log")
        for handler in root.handlers:
            handler.flush()
        assert "hello log" in (folder / "app.log").read_text(encoding="utf-8")
        assert (folder / "crash.log").exists()
    finally:
        for handler in root.handlers[:]:
            if handler not in before:
                root.removeHandler(handler)
                handler.close()
        import faulthandler

        faulthandler.disable()


def test_log_dir_is_under_localappdata(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert app_module.log_dir() == tmp_path / "DesktopAudioRecorder" / "logs"


def test_qapplication_exists(qapp):
    assert isinstance(QApplication.instance(), QApplication)
