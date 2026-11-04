"""Helpers for the release workflow (``.github/workflows/release.yml``).

    uv run python packaging/release_tools.py check-tag v1.0.0-rc1
    uv run python packaging/release_tools.py notes v1.0.0 --out notes.md
    uv run python packaging/release_tools.py checksums dist/X.zip --out dist/SHA256SUMS.txt

- ``check-tag``: the tag is ``v<version>`` or ``v<version>-<pre-release>`` and ``<version>`` is
  the package ``__version__``, so a tag can't publish a build with another version inside.
- ``notes``: the tag's section of ``CHANGELOG.md``, matched on the version and not the date
  (``## [1.0.0] - unreleased`` and ``## [1.0.0] - 2026-11-05`` both match). A pre-release tag
  without a section of its own (``v1.0.0-rc1``) gets its base version's section.
- ``checksums``: ``SHA256SUMS.txt`` in ``sha256sum`` format (``<hash>  <name>``), which
  ``sha256sum -c`` and ``Get-FileHash`` comparisons both read.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
INIT = ROOT / "src" / "desktop_audio_recorder" / "__init__.py"

_TAG = re.compile(r"^v(?P<version>\d+\.\d+\.\d+)(?:-(?P<pre>[0-9A-Za-z.]+))?$")
_SECTION = re.compile(r"^## \[(?P<version>[^\]]+)\]")
_LINK_REFERENCE = re.compile(r"^\[[^\]]+\]:\s")


class ReleaseError(Exception):
    pass


def package_version() -> str:
    text = INIT.read_text(encoding="utf-8")
    return re.search(r'^__version__ = "([^"]+)"', text, re.M).group(1)


def parse_tag(tag: str) -> tuple[str, str | None]:
    """``"v1.0.0-rc1"`` -> ``("1.0.0", "rc1")``; ``"v1.0.0"`` -> ``("1.0.0", None)``."""
    match = _TAG.match(tag)
    if match is None:
        raise ReleaseError(f"tag {tag!r} isn't v<major>.<minor>.<patch>[-<pre-release>]")
    return match.group("version"), match.group("pre")


def check_tag(tag: str, version: str) -> None:
    base, _ = parse_tag(tag)
    if base != version:
        raise ReleaseError(f"tag {tag!r} is for {base}, but the package version is {version}")


def changelog_section(text: str, version: str) -> str:
    """The body of the ``## [version]`` section (without its heading and link references)."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        match = _SECTION.match(line)
        if start is None:
            if match and match.group("version") == version:
                start = i + 1
        elif match or line.startswith("# "):
            lines = lines[start:i]
            break
    else:
        if start is None:
            raise ReleaseError(f"CHANGELOG.md has no '## [{version}]' section")
        lines = lines[start:]
    body = [line for line in lines if not _LINK_REFERENCE.match(line)]
    result = "\n".join(body).strip()
    if not result:
        raise ReleaseError(f"CHANGELOG.md's '## [{version}]' section is empty")
    return result + "\n"


def release_notes(tag: str, text: str) -> str:
    base, pre = parse_tag(tag)
    full = f"{base}-{pre}" if pre else base
    try:
        return changelog_section(text, full)
    except ReleaseError:
        if pre is None:
            raise
    return changelog_section(text, base)


def sha256sums(paths: list[Path]) -> str:
    lines = []
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1 << 20), b""):
                digest.update(block)
        lines.append(f"{digest.hexdigest()}  {path.name}\n")
    return "".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    tag_parser = commands.add_parser("check-tag", help="the tag matches __version__")
    tag_parser.add_argument("tag")
    notes_parser = commands.add_parser("notes", help="the tag's CHANGELOG.md section")
    notes_parser.add_argument("tag")
    notes_parser.add_argument("--out", type=Path, required=True)
    sums_parser = commands.add_parser("checksums", help="write SHA256SUMS.txt")
    sums_parser.add_argument("files", type=Path, nargs="+")
    sums_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "check-tag":
            check_tag(args.tag, package_version())
            print(f"Tag {args.tag} matches version {package_version()}")
        elif args.command == "notes":
            notes = release_notes(args.tag, CHANGELOG.read_text(encoding="utf-8"))
            args.out.write_text(notes, encoding="utf-8", newline="\n")
            print(f"Wrote {args.out.name}: {len(notes.splitlines())} lines from CHANGELOG.md")
        else:
            sums = sha256sums(args.files)
            args.out.write_text(sums, encoding="utf-8", newline="\n")
            print(f"Wrote {args.out.name}:\n{sums}", end="")
    except ReleaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
