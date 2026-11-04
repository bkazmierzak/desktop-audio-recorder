"""Phase 0 smoke tests: dependencies import and the app window can be built."""

import importlib

import pytest

DEPENDENCIES = [
    "pyaudiowpatch",
    "sounddevice",
    "soundfile",
    "numpy",
    "scipy",
    "noisereduce",
    "lameenc",
    "PySide6.QtWidgets",
    "pyqtgraph",
]


@pytest.mark.parametrize("module", DEPENDENCIES)
def test_dependency_imports(module):
    importlib.import_module(module)


def test_wasapi_loopback_available():
    pyaudio = importlib.import_module("pyaudiowpatch")
    pa = pyaudio.PyAudio()
    try:
        try:
            pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            pytest.skip("WASAPI host API not available on this machine")
        loopback = pa.get_default_wasapi_loopback()
        assert loopback["maxInputChannels"] > 0
    finally:
        pa.terminate()


def test_main_window_builds():
    from PySide6.QtWidgets import QApplication

    from desktop_audio_recorder.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert "Desktop Audio Recorder" in window.windowTitle()
    window.close()
    app.processEvents()


def test_package_version_matches_the_project_version():
    """``__version__`` (the window title, About, the exe's version resource) and pyproject.toml
    (``uv version``) must not drift apart."""
    import importlib.metadata

    from desktop_audio_recorder import __version__

    assert __version__ == importlib.metadata.version("desktop-audio-recorder")
