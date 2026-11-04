"""Help > About: version, license, project link and the third-party notices.

A release build has ``LICENSE`` and ``THIRD-PARTY-NOTICES.txt`` next to the exe
(``packaging/build.py`` writes the notices). Run from source, ``LICENSE`` is the repository's
and there are no notices, so the About box has no button for them.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from html import escape
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from desktop_audio_recorder import (
    APP_NAME,
    COPYRIGHT,
    HOMEPAGE_URL,
    LICENSE_FILE,
    NOTICES_FILE,
    __version__,
)

ABOUT_TITLE = f"About {APP_NAME}"
BUNDLED = (
    "Qt (PySide6), PortAudio (PyAudioWPatch, sounddevice), libsndfile with mpg123, LAME,"
    " NumPy, SciPy, noisereduce and pyqtgraph"
)


def app_folder() -> Path:
    """The exe's folder in a release build; the repository root when run from source."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def notices_path() -> Path | None:
    path = app_folder() / NOTICES_FILE
    return path if path.is_file() else None


def license_path() -> Path | None:
    path = app_folder() / LICENSE_FILE
    return path if path.is_file() else None


def about_html() -> str:
    link = escape(HOMEPAGE_URL.removeprefix("https://"))
    return (
        f"<h3>{escape(APP_NAME)} {escape(__version__)}</h3>"
        "<p>Records what your PC plays (the Windows default playback device), lets you edit"
        " and clean up the clip, and saves it as WAV or MP3.</p>"
        f"<p>{escape(COPYRIGHT.replace('(c)', '©'))}.<br>"
        "Free software under the MIT License.</p>"
        f'<p><a href="{escape(HOMEPAGE_URL)}">{link}</a></p>'
        f"<p>Uses {escape(BUNDLED)}. Their licenses are in the third-party notices.</p>"
    )


def open_file(path: Path) -> None:
    """Open a file with its default program (Notepad for the .txt notices)."""
    os.startfile(path)  # noqa: S606 - our own file


def make_about_box(parent: QWidget | None, open_path: Callable[[Path], None]) -> QMessageBox:
    """The About box (not shown yet). Its buttons: Third-party notices (in a release build),
    License, OK. Both file buttons close the box and open the file."""
    box = QMessageBox(parent)
    box.setWindowTitle(ABOUT_TITLE)
    box.setTextFormat(Qt.TextFormat.RichText)
    box.setText(about_html())
    box.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    icon = QApplication.windowIcon()
    if not icon.isNull():
        box.setIconPixmap(icon.pixmap(64, 64))
    for label, path in (("Third-party notices", notices_path()), ("License", license_path())):
        if path is not None:
            button = box.addButton(label, QMessageBox.ButtonRole.ActionRole)
            button.clicked.connect(lambda _=False, p=path: open_path(p))
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.setDefaultButton(QMessageBox.StandardButton.Ok)
    return box
