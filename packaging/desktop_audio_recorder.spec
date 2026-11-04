# PyInstaller spec: a single-folder, windowed build of Desktop Audio Recorder.
#
#   uv run pyinstaller packaging/desktop_audio_recorder.spec --noconfirm
#
# Output: dist/DesktopAudioRecorder/DesktopAudioRecorder.exe (run the whole folder; it isn't a
# one-file exe). Check a build with:
#
#   dist/DesktopAudioRecorder/DesktopAudioRecorder.exe --smoke-test <folder>
#
# packaging/build.py does both, writes THIRD-PARTY-NOTICES.txt and LICENSE next to the exe, and
# zips the folder for distribution. For the notices, this spec lists the top-level modules and
# folders in the bundle in <workpath>/bundled-top-level.json.
#
# The exe carries the app icon (src/desktop_audio_recorder/resources/app.ico, drawn by
# packaging/make_icon.py; the window uses the same file at run time) and a version resource
# built from desktop_audio_recorder.__version__ (Explorer > Properties > Details).
#
# Native libraries and how they get in:
# * pyaudiowpatch: PortAudio is linked into _portaudiowpatch.pyd (found by import analysis).
# * sounddevice: _sounddevice_data/portaudio-binaries/libportaudio64bit.dll (contrib hook; the
#   ASIO variant is left out: sounddevice loads it only when SD_ENABLE_ASIO is set, and it
#   carries Steinberg's ASIO SDK, which has its own license).
# * soundfile: _soundfile_data/libsndfile_x64.dll, with mpg123/LAME inside (contrib hook).
# * lameenc: lameenc.cp312-win_amd64.pyd (import analysis).
# * PySide6 essentials: QtCore/QtGui/QtWidgets plus the platforms/windows, styles and
#   imageformats plugins (PyInstaller's PySide6 hook).
# * pyqtgraph and comtypes: contrib hooks.
# noisereduce depends on matplotlib but never imports it at run time (plotting is optional);
# joblib and tqdm are imported, so they stay.

import json
import re
from pathlib import Path

from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo,
    StringFileInfo,
    StringStruct,
    StringTable,
    VarFileInfo,
    VarStruct,
    VSVersionInfo,
)

ROOT = Path(SPECPATH).parent  # noqa: F821 - defined by PyInstaller
PACKAGE = ROOT / "src" / "desktop_audio_recorder"
ICON = PACKAGE / "resources" / "app.ico"
VERSION = re.search(
    r'^__version__ = "([^"]+)"', (PACKAGE / "__init__.py").read_text(encoding="utf-8"), re.M
).group(1)
_numbers = tuple(int(n) for n in re.findall(r"\d+", VERSION)[:4])
VERSION_TUPLE = _numbers + (0,) * (4 - len(_numbers))

version_info = VSVersionInfo(
    ffi=FixedFileInfo(filevers=VERSION_TUPLE, prodvers=VERSION_TUPLE),
    kids=[
        StringFileInfo(
            [
                StringTable(
                    "040904B0",  # US English, Unicode
                    [
                        StringStruct("CompanyName", "bkazmierzak"),
                        StringStruct("FileDescription", "Desktop Audio Recorder"),
                        StringStruct("FileVersion", VERSION),
                        StringStruct("InternalName", "DesktopAudioRecorder"),
                        StringStruct("LegalCopyright", "Copyright (c) 2026 bkazmierzak"),
                        StringStruct("OriginalFilename", "DesktopAudioRecorder.exe"),
                        StringStruct("ProductName", "Desktop Audio Recorder"),
                        StringStruct("ProductVersion", VERSION),
                    ],
                )
            ]
        ),
        VarFileInfo([VarStruct("Translation", [0x0409, 1200])]),
    ],
)

excludes = [
    "matplotlib",
    "tkinter",
    "_tkinter",
    "IPython",
    "jupyter_client",
    "notebook",
    "pandas",
    "torch",
    "numba",
    "cupy",
    "h5py",
    "OpenGL",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "pyqtgraph.opengl",
    "pyqtgraph.examples",
    "pyqtgraph.jupyter",
    "scipy.io.matlab.tests",
    "pytest",
]
# No PySide6 module is excluded: pyqtgraph imports QtOpenGL, QtOpenGLWidgets, QtSvg and QtTest
# at start-up (excluding QtOpenGL broke the first build), and PyInstaller's PySide6 hook only
# collects the Qt modules that are imported anyway.

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "launch.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[(str(ICON), "desktop_audio_recorder/resources")],
    hiddenimports=[
        "desktop_audio_recorder.smoke_test",  # imported lazily for --smoke-test
        "comtypes.stream",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

# The 32-bit, ARM and macOS PortAudio variants sounddevice ships are never loaded on x64, and
# the ASIO one only with SD_ENABLE_ASIO.
def _wanted(entry):
    return not any(tag in Path(entry[0]).name for tag in ("32bit", "arm64", ".dylib", "-asio"))


a.binaries = [entry for entry in a.binaries if _wanted(entry)]
a.datas = [entry for entry in a.datas if _wanted(entry)]


def _top_level(name):
    """'numpy.linalg' -> 'numpy'; 'PySide6/Qt6Core.dll' -> 'PySide6'; 'lameenc.cp312.pyd' ->
    'lameenc'."""
    return re.split(r"[\\/]", name, maxsplit=1)[0].split(".")[0]


_bundled = sorted({_top_level(entry[0]) for entry in (*a.pure, *a.binaries, *a.datas)})
(Path(workpath) / "bundled-top-level.json").write_text(  # noqa: F821 - defined by PyInstaller
    json.dumps(_bundled, indent=0), encoding="utf-8"
)

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DesktopAudioRecorder",
    console=False,  # a windowed app; logs go to %LOCALAPPDATA%\DesktopAudioRecorder\logs
    icon=str(ICON),
    version=version_info,
    upx=False,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    upx=False,
    name="DesktopAudioRecorder",
)
