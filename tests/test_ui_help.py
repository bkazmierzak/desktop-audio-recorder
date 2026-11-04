"""Help menu: About, About Qt, Open log folder, Report a problem (and the error dialog's link)."""

import pytest
from conftest import FakeBackend, FakeDeviceManager, FakeRecorder
from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QLabel, QMessageBox

from desktop_audio_recorder import (
    HOMEPAGE_URL,
    LICENSE_FILE,
    NEW_ISSUE_URL,
    NOTICES_FILE,
    __version__,
)
from desktop_audio_recorder.audio.player import ClipPlayer
from desktop_audio_recorder.settings import Settings
from desktop_audio_recorder.ui import about
from desktop_audio_recorder.ui.error_reporter import ErrorReporter
from desktop_audio_recorder.ui.main_window import MainWindow

TIMEOUT = 2000


@pytest.fixture
def make_window(qtbot, tmp_path):
    def make(**kwargs):
        window = MainWindow(
            recorder=FakeRecorder(tmp_path),
            device_manager=FakeDeviceManager(),
            player=ClipPlayer(FakeBackend()),
            settings=Settings(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)),
            **kwargs,
        )
        qtbot.addWidget(window)
        window.opened = []
        window.open_url = lambda url: window.opened.append(("url", url))
        window.open_folder = lambda path: window.opened.append(("folder", path))
        window.open_path = lambda path: window.opened.append(("file", path))
        return window

    return make


@pytest.fixture
def app_folder(monkeypatch, tmp_path):
    """A fake release folder (empty: no LICENSE, no notices)."""
    folder = tmp_path / "release"
    folder.mkdir()
    monkeypatch.setattr(about, "app_folder", lambda: folder)
    return folder


def help_menu(window):
    menus = {a.text(): a.menu() for a in window.menuBar().actions()}
    assert list(menus) == ["&File", "&Recording", "&Help"]
    return menus["&Help"]


def test_help_menu_items(make_window):
    window = make_window()
    texts = [a.text() for a in help_menu(window).actions()]
    assert texts == [
        "&About Desktop Audio Recorder",
        "About &Qt",
        "",  # separator
        "Open &log folder",
        "&Report a problem…",
    ]


def test_report_a_problem_opens_a_new_issue(make_window):
    window = make_window()
    window.report_action.trigger()
    assert window.opened == [("url", NEW_ISSUE_URL)]
    assert NEW_ISSUE_URL.startswith(HOMEPAGE_URL + "/issues/new")


def test_open_log_folder(make_window, tmp_path):
    window = make_window(log_dir=tmp_path / "logs")
    assert window.log_folder_action.isEnabled()
    window.log_folder_action.trigger()
    assert window.opened == [("folder", tmp_path / "logs")]


def test_open_log_folder_is_disabled_without_a_log(make_window):
    window = make_window()
    assert not window.log_folder_action.isEnabled()
    window.open_log_folder()
    assert window.opened == []


def test_about_qt_is_connected(make_window, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "desktop_audio_recorder.ui.main_window.QApplication.aboutQt", lambda: calls.append(1)
    )
    window = make_window()
    window.about_qt_action.trigger()
    assert len(calls) == 1


def test_about_box_from_source_has_no_notices_button(make_window, app_folder, qtbot):
    window = make_window()
    window.about_action.trigger()
    box = window.about_box
    assert isinstance(box, QMessageBox)
    qtbot.waitUntil(box.isVisible, timeout=TIMEOUT)
    assert box.windowTitle() == "About Desktop Audio Recorder"
    text = box.text()
    assert f"Desktop Audio Recorder {__version__}" in text
    assert "MIT License" in text
    assert f'href="{HOMEPAGE_URL}"' in text
    assert "LAME" in text and "PortAudio" in text and "libsndfile" in text and "Qt" in text
    labels = [b.text() for b in box.buttons()]
    assert "Third-party notices" not in labels and "License" not in labels
    # The link opens in the browser.
    label = next(lab for lab in box.findChildren(QLabel) if "href" in lab.text())
    assert label.openExternalLinks()
    window.about_action.trigger()  # a second About while one is open: the same box
    assert window.about_box is box
    box.accept()
    qtbot.waitUntil(lambda: window.about_box is None, timeout=TIMEOUT)


def test_about_box_opens_the_notices_and_license(make_window, app_folder, qtbot):
    (app_folder / NOTICES_FILE).write_text("notices", encoding="utf-8")
    (app_folder / LICENSE_FILE).write_text("MIT", encoding="utf-8")
    window = make_window()
    for label, name in (("Third-party notices", NOTICES_FILE), ("License", LICENSE_FILE)):
        window.opened.clear()
        window.show_about()
        box = window.about_box
        qtbot.waitUntil(box.isVisible, timeout=TIMEOUT)
        button = next(b for b in box.buttons() if b.text() == label)
        button.click()
        assert window.opened == [("file", app_folder / name)]
        qtbot.waitUntil(lambda: window.about_box is None, timeout=TIMEOUT)


def test_app_folder_from_source_is_the_repository():
    folder = about.app_folder()
    assert (folder / "pyproject.toml").is_file()
    assert about.license_path() == folder / LICENSE_FILE


def test_error_dialog_has_a_report_link(qapp, qtbot, tmp_path):
    reporter = ErrorReporter(tmp_path)
    opened = []
    reporter.open_url = opened.append
    reporter.open_folder = lambda path: opened.append(("folder", path))
    reporter.install()
    try:
        QTimer.singleShot(0, lambda: (_ for _ in ()).throw(ValueError("boom")))
        qtbot.waitUntil(lambda: reporter._dialog is not None, timeout=TIMEOUT)
        box = reporter._dialog
        buttons = {b.text(): b for b in box.buttons()}
        assert "Report a problem" in buttons and "Open log folder" in buttons
        buttons["Report a problem"].click()
        assert opened == [NEW_ISSUE_URL]
        qtbot.waitUntil(lambda: reporter._dialog is None, timeout=TIMEOUT)
    finally:
        reporter.uninstall()
