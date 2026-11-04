"""Build the exe, check it, and zip it for distribution.

    uv run python packaging/build.py              # build + smoke test + zip
    uv run python packaging/build.py --no-smoke   # without the smoke test (it plays a quiet tone)
    uv run python packaging/build.py --zip-only   # zip an existing build

1. PyInstaller with ``packaging/desktop_audio_recorder.spec`` -> ``dist/DesktopAudioRecorder/``.
2. ``LICENSE`` and ``THIRD-PARTY-NOTICES.txt`` next to the exe (``write_notices``): the license
   files of every Python distribution in the bundle (from their installed metadata), notes on
   the native libraries inside them (Qt, libsndfile with mpg123 and LAME, LAME in lameenc,
   PortAudio, the Python runtime) and the full GPL/LGPL/Apache texts from
   ``packaging/licenses/``. The spec lists what is in the bundle.
3. ``DesktopAudioRecorder.exe --smoke-test <temp folder>`` (records a tone from the default
   playback device, plays it, exports WAV and MP3; exit code 0 = pass). Needs a playback device.
4. ``dist/DesktopAudioRecorder-<version>-win64.zip``: the ``DesktopAudioRecorder`` folder plus
   ``README.md``, ``LICENSE`` and ``THIRD-PARTY-NOTICES.txt``. Unzip anywhere and run
   ``DesktopAudioRecorder.exe``; nothing is installed.

The zip is written to a temporary name and renamed when complete. Exit code 0 when every step
passed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "packaging" / "desktop_audio_recorder.spec"
LICENSES = ROOT / "packaging" / "licenses"
DIST = ROOT / "dist"
APP_DIR = DIST / "DesktopAudioRecorder"
EXE = APP_DIR / "DesktopAudioRecorder.exe"
# Written by the spec: the top-level modules and folders in the bundle.
BUNDLED = ROOT / "build" / "desktop_audio_recorder" / "bundled-top-level.json"
LICENSE = ROOT / "LICENSE"
NOTICES = "THIRD-PARTY-NOTICES.txt"
APP_DIST = "desktop-audio-recorder"
# Files in a dist-info folder that are license texts (besides its License-File entries).
_LICENSE_NAME = re.compile(r"^(licen[cs]e|copying|notice|authors)", re.I)
_RULE = "=" * 78
# Bundled names that no distribution provides, and that's expected: the app itself, the Python
# runtime's DLLs and archive (covered by the Python notes), and dist-info folders.
_EXPECTED_UNMAPPED = re.compile(
    r"^(desktop_audio_recorder|python3\d*|vcruntime140(_1)?|ucrtbase|base_library|api-ms-win-.*"
    r"|libcrypto-.*|libssl-.*|libffi-.*|.*-\d+)$",
    re.I,
)


def version() -> str:
    text = (ROOT / "src" / "desktop_audio_recorder" / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'^__version__ = "([^"]+)"', text, re.M).group(1)


def build() -> None:
    started = time.monotonic()
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", str(SPEC), "--noconfirm"], cwd=ROOT, check=True
    )
    print(f"Built {EXE} in {time.monotonic() - started:.0f} s")


# ---------------------------------------------------------------------- notices


def bundled_distributions(names: list[str]) -> tuple[list[metadata.Distribution], list[str]]:
    """The installed distributions that provide the bundle's top-level ``names``, and the names
    no distribution provides that aren't part of the standard library (for a look)."""
    providers = metadata.packages_distributions()
    found: dict[str, metadata.Distribution] = {}
    unknown = []
    for name in names:
        dists = providers.get(name)
        if not dists:
            stdlib = name in sys.stdlib_module_names or name.startswith("_")
            if not stdlib and not _EXPECTED_UNMAPPED.match(name):
                unknown.append(name)
            continue
        for dist_name in dists:
            dist = metadata.distribution(dist_name)
            key = dist.metadata["Name"].lower().replace("_", "-")
            if key != APP_DIST:
                found[key] = dist
    return [found[key] for key in sorted(found)], unknown


def license_texts(dist: metadata.Distribution) -> list[tuple[str, str]]:
    """``(file name, text)`` for each license file in the distribution's dist-info folder."""
    texts: dict[str, str] = {}
    for file in dist.files or ():
        parts = file.parts
        if not parts or not parts[0].endswith(".dist-info"):
            continue
        inner = parts[1:]
        if not inner:
            continue
        if inner[0] == "licenses" or (len(inner) == 1 and _LICENSE_NAME.match(inner[0])):
            path = Path(file.locate())
            if path.is_file():
                name = "/".join(inner[1:] if inner[0] == "licenses" else inner)
                texts[name] = path.read_text(encoding="utf-8", errors="replace")
    if not texts:  # e.g. an old-style package with the text in the License field
        field = dist.metadata.get("License") or ""
        if "\n" in field.strip():
            texts["License (from the package metadata)"] = field
    return sorted(texts.items())


def _license_name(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    if meta.get("License-Expression"):
        return meta["License-Expression"]
    field = (meta.get("License") or "").strip()
    if field and "\n" not in field and len(field) < 80:
        return field
    classifiers = [
        c.split(" :: ")[-1] for c in meta.get_all("Classifier") or () if c.startswith("License ::")
    ]
    return ", ".join(classifiers) or "see the license text below"


def _homepage(dist: metadata.Distribution) -> str:
    urls = {}
    for entry in dist.metadata.get_all("Project-URL") or ():
        label, _, url = entry.partition(",")
        urls[label.strip().lower()] = url.strip()
    for label in ("homepage", "home", "source", "source code", "repository", "github"):
        if label in urls:
            return urls[label]
    return dist.metadata.get("Home-page") or (next(iter(urls.values())) if urls else "")


def _native_versions() -> dict[str, str]:
    """Versions of the native libraries, read from the installed packages (best effort)."""
    found = {"python": sys.version.split()[0], "openssl": ssl.OPENSSL_VERSION}
    probes = {
        "qt": "from PySide6.QtCore import qVersion; v = qVersion()",
        "libsndfile": "import soundfile; v = soundfile.__libsndfile_version__",
        "portaudio_sd": "import sounddevice; v = sounddevice.get_portaudio_version()[1]",
        "portaudio_pa": "import pyaudiowpatch; v = pyaudiowpatch.get_portaudio_version_text()",
    }
    for key, code in probes.items():
        scope: dict = {}
        try:
            exec(code, scope)  # noqa: S102 - our own probe strings
            found[key] = str(scope["v"])
        except Exception as exc:  # noqa: BLE001 - a missing version only shortens the notes
            found[key] = f"unknown ({type(exc).__name__})"
    return found


def _dist_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "?"


def native_notes() -> str:
    v = _native_versions()
    pyside = _dist_version("pyside6-essentials")
    lameenc = _dist_version("lameenc")
    soundfile = _dist_version("soundfile")
    return f"""\
LGPL components: separate, replaceable files
--------------------------------------------
Qt (with PySide6 and shiboken6), libsndfile (with mpg123 and LAME built into it) and LAME (built
into lameenc) are licensed under the GNU Lesser General Public License. Each one is a separate
file in the _internal folder next to DesktopAudioRecorder.exe, loaded when the app starts, and
none of them is modified: they are the files from the PyPI wheels named below. You may replace
any of them with your own build of a compatible version of the same library, at the same path,
and you may reverse-engineer the app as far as needed to debug such a change. The source code is
at the links below; if a link stops working, ask in a GitHub issue
(https://github.com/bkazmierzak/desktop-audio-recorder/issues) and the source of the exact
versions used will be provided. The full LGPL and GPL texts are under "Full license texts".

* Qt {v["qt"]}, PySide6 and shiboken6 {pyside} (wheels pyside6-essentials, shiboken6)
  License: LGPL-3.0-only (also offered under GPL-2.0/GPL-3.0 and commercially; used here under
  the LGPL-3.0).
  Files: _internal\\PySide6\\ (Qt6*.dll, the plugins folder, the PySide6 modules) and
  _internal\\shiboken6\\.
  Source: https://download.qt.io/official_releases/qt/ (Qt) and
  https://code.qt.io/cgit/pyside/pyside-setup.git/ (PySide6, shiboken6).
  Qt contains third-party code (FreeType, HarfBuzz, libpng, libjpeg, zlib, PCRE2, md4c and
  others); their licenses are listed at https://doc.qt.io/qt-6/licenses-used-in-qt.html.
  Help > About Qt in the app shows Qt's own notice.

* libsndfile {v["libsndfile"]} (from soundfile {soundfile})
  License: LGPL-2.1-or-later. Built into the same DLL: mpg123 (LGPL-2.1), LAME (LGPL-2.0), and
  libogg, libvorbis, FLAC and Opus (BSD-3-Clause).
  File: _internal\\_soundfile_data\\libsndfile_x64.dll.
  Source: https://github.com/libsndfile/libsndfile, and the build scripts for this DLL at
  https://github.com/bastibe/libsndfile-binaries (mpg123: https://www.mpg123.de/, LAME:
  https://lame.sourceforge.io/, Xiph codecs: https://xiph.org/downloads/).

* LAME, in lameenc {lameenc}
  License: lameenc is LGPL-3.0-or-later; the LAME encoder compiled into it is LGPL-2.0-or-later.
  File: _internal\\lameenc.cp312-win_amd64.pyd (LAME is linked into this module; to change
  LAME, rebuild lameenc from its source against your LAME and replace this file).
  Source: https://github.com/chrisstaite/lameenc and https://lame.sourceforge.io/.

Other native libraries
----------------------
* PortAudio, twice:
  - linked into PyAudioWPatch (_internal\\_portaudiowpatch.cp312-win_amd64.pyd), which
    records the playback device: {v["portaudio_pa"]};
  - in sounddevice (_internal\\_sounddevice_data\\portaudio-binaries\\libportaudio64bit.dll),
    which plays clips: {v["portaudio_sd"]}.
  License: the PortAudio license (MIT style; text under "Full license texts").
  Source: https://github.com/PortAudio/portaudio, https://github.com/s0d3s/PyAudioWPatch,
  https://github.com/spatialaudio/portaudio-binaries.

* Python {v["python"]} (python312.dll, the standard library in base_library.zip and the .pyd
  modules in _internal). License: the Python Software Foundation License. Its text, and
  those of the libraries the Python runtime contains, are under "Full license texts".
  The runtime includes {v["openssl"]} (libssl-3-x64.dll, libcrypto-3-x64.dll; Apache-2.0).

* The Microsoft Visual C++ runtime (VCRUNTIME140*.dll, ucrtbase.dll, api-ms-win-*.dll),
  redistributed as part of the Python runtime under the terms in the Python license section.

* PyInstaller's bootloader (inside DesktopAudioRecorder.exe), GPL-2.0-or-later with the
  PyInstaller bootloader exception, which allows distributing it with any program.
  https://pyinstaller.org/
"""


def notices_text(dists: list[metadata.Distribution]) -> str:
    out = [
        f"Desktop Audio Recorder {version()}: third-party notices",
        _RULE,
        "",
        "Desktop Audio Recorder is free software under the MIT License (the LICENSE file next to",
        "this one). https://github.com/bkazmierzak/desktop-audio-recorder",
        "",
        "This release build contains the software listed below, each under its own license.",
        "",
        "Python packages in this build:",
        "",
    ]
    for dist in dists:
        out.append(f"  {dist.metadata['Name']} {dist.version}  ({_license_name(dist)})")
    out += ["", native_notes(), ""]

    seen: dict[str, str] = {}  # text hash -> where it first appeared
    for dist in dists:
        title = f"{dist.metadata['Name']} {dist.version}"
        out += [_RULE, title, _RULE, f"License: {_license_name(dist)}"]
        homepage = _homepage(dist)
        if homepage:
            out.append(f"Home: {homepage}")
        texts = license_texts(dist)
        if not texts:
            out += ["", "(The package includes no license file.)"]
        for name, text in texts:
            digest = hashlib.sha256(text.strip().encode()).hexdigest()
            out += ["", f"--- {name} ---", ""]
            if digest in seen:
                out.append(f"(The same text as {seen[digest]}.)")
            else:
                seen[digest] = f"{name} of {title}"
                out.append(text.strip())
        out.append("")

    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    full = [
        ("Python license (CPython, with its Windows build terms, bzip2 and Tcl/Tk)", None),
        ("Libraries inside the Python runtime", LICENSES / "python-runtime.txt"),
        ("PortAudio license", LICENSES / "PortAudio.txt"),
        ("GNU Lesser General Public License, version 3", LICENSES / "LGPL-3.0.txt"),
        ("GNU General Public License, version 3", LICENSES / "GPL-3.0.txt"),
        ("GNU Lesser General Public License, version 2.1", LICENSES / "LGPL-2.1.txt"),
        ("Apache License, version 2.0 (OpenSSL, PyAudioWPatch)", LICENSES / "Apache-2.0.txt"),
    ]
    out += [_RULE, "Full license texts", _RULE, ""]
    for title, path in full:
        if path is None:
            path = python_license
            if not path.is_file():
                raise SystemExit(f"The Python license isn't at {path}")
        out += ["-" * 78, title, "-" * 78, "", path.read_text(encoding="utf-8").strip(), ""]
    return "\n".join(out).replace("\r\n", "\n") + "\n"


def write_notices(target: Path = APP_DIR) -> Path:
    """Write ``THIRD-PARTY-NOTICES.txt`` and copy ``LICENSE`` into ``target``."""
    if not BUNDLED.exists():
        raise SystemExit(f"{BUNDLED} doesn't exist: build first (the spec writes it)")
    names = json.loads(BUNDLED.read_text(encoding="utf-8"))
    dists, unknown = bundled_distributions(names)
    text = notices_text(dists)
    path = target / NOTICES
    path.write_text(text, encoding="utf-8", newline="\r\n")  # Notepad-friendly
    shutil.copyfile(LICENSE, target / "LICENSE")
    print(
        f"Wrote {path.name} ({len(text) / 1024:.0f} KB, {len(dists)} distributions:"
        f" {', '.join(d.metadata['Name'] for d in dists)}) and LICENSE"
    )
    if unknown:
        print(f"  Bundled names without a distribution (check them): {', '.join(unknown)}")
    return path


# ---------------------------------------------------------------------- check and zip


def smoke_test() -> bool:
    folder = Path(tempfile.mkdtemp(prefix="dar-smoke-"))
    started = time.monotonic()
    code = subprocess.run([str(EXE), "--smoke-test", str(folder)], timeout=300).returncode
    report = folder / "report.json"
    summary = ""
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        summary = ", ".join(
            f"{name}: {'ok' if check.get('ok') else 'FAILED'}"
            for name, check in data.get("checks", {}).items()
            if isinstance(check, dict)
        )
        if data.get("error"):
            summary += f"; error: {data['error']}"
    print(f"Smoke test exit code {code} in {time.monotonic() - started:.1f} s ({summary})")
    print(f"  report: {report}")
    return code == 0


def make_zip() -> Path:
    if not EXE.exists():
        raise SystemExit(f"{EXE} doesn't exist: build first")
    if not (APP_DIR / NOTICES).exists() or not (APP_DIR / "LICENSE").exists():
        write_notices()
    target = DIST / f"DesktopAudioRecorder-{version()}-win64.zip"
    partial = target.with_name(target.name + ".part")
    started = time.monotonic()
    files = sorted(p for p in APP_DIR.rglob("*") if p.is_file())
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, Path("DesktopAudioRecorder") / path.relative_to(APP_DIR))
        archive.write(ROOT / "README.md", "README.md")
        archive.write(APP_DIR / "LICENSE", "LICENSE")
        archive.write(APP_DIR / NOTICES, NOTICES)
    os.replace(partial, target)
    unpacked = sum(p.stat().st_size for p in files)
    print(
        f"Wrote {target}: {target.stat().st_size / 2**20:.0f} MB"
        f" ({len(files)} files, {unpacked / 2**20:.0f} MB unpacked) in"
        f" {time.monotonic() - started:.0f} s"
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-smoke", action="store_true", help="skip the smoke test")
    parser.add_argument("--zip-only", action="store_true", help="zip an existing build")
    args = parser.parse_args(argv)
    if not args.zip_only:
        if APP_DIR.exists():
            shutil.rmtree(APP_DIR)  # PyInstaller's --noconfirm would, too; be explicit
        build()
        write_notices()
        if not args.no_smoke and not smoke_test():
            print("The smoke test failed: not zipping this build.")
            return 1
    make_zip()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
