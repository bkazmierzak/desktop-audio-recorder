"""Build the exe, check it, and zip it for distribution.

    uv run python packaging/build.py              # build + smoke test + zip
    uv run python packaging/build.py --no-smoke   # without the smoke test (it plays a quiet tone)
    uv run python packaging/build.py --zip-only   # zip an existing build

1. PyInstaller with ``packaging/desktop_audio_recorder.spec`` -> ``dist/DesktopAudioRecorder/``.
2. ``DesktopAudioRecorder.exe --smoke-test <temp folder>`` (records a tone from the default
   playback device, plays it, exports WAV and MP3; exit code 0 = pass). Needs a playback device.
3. ``dist/DesktopAudioRecorder-<version>-win64.zip``: the ``DesktopAudioRecorder`` folder plus
   ``README.md``. Unzip anywhere and run ``DesktopAudioRecorder.exe``; nothing is installed.

The zip is written to a temporary name and renamed when complete. Exit code 0 when every step
passed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "packaging" / "desktop_audio_recorder.spec"
DIST = ROOT / "dist"
APP_DIR = DIST / "DesktopAudioRecorder"
EXE = APP_DIR / "DesktopAudioRecorder.exe"


def version() -> str:
    text = (ROOT / "src" / "desktop_audio_recorder" / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'^__version__ = "([^"]+)"', text, re.M).group(1)


def build() -> None:
    started = time.monotonic()
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", str(SPEC), "--noconfirm"], cwd=ROOT, check=True
    )
    print(f"Built {EXE} in {time.monotonic() - started:.0f} s")


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
    target = DIST / f"DesktopAudioRecorder-{version()}-win64.zip"
    partial = target.with_name(target.name + ".part")
    started = time.monotonic()
    files = sorted(p for p in APP_DIR.rglob("*") if p.is_file())
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, Path("DesktopAudioRecorder") / path.relative_to(APP_DIR))
        archive.write(ROOT / "README.md", "README.md")
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
        if not args.no_smoke and not smoke_test():
            print("The smoke test failed: not zipping this build.")
            return 1
    make_zip()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
