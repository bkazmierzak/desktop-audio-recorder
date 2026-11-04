"""packaging/release_tools.py: the release workflow's tag check, notes and checksums."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

CHANGELOG = """# Changelog

Intro text.

## [Unreleased]

### Fixed
- Not yet released.

## [1.1.0] - 2027-01-02

### Added
- Something new.

## [1.0.0] - unreleased

The first public release.

### Added
- Recording.
- Export.

[1.1.0]: https://example.invalid/releases/tag/v1.1.0
[1.0.0]: https://example.invalid/releases/tag/v1.0.0
"""


@pytest.fixture(scope="module")
def tools():
    spec = importlib.util.spec_from_file_location(
        "release_tools", ROOT / "packaging" / "release_tools.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("v1.0.0", ("1.0.0", None)),
        ("v1.0.0-rc1", ("1.0.0", "rc1")),
        ("v12.3.45-beta.2", ("12.3.45", "beta.2")),
    ],
)
def test_parse_tag(tools, tag, expected):
    assert tools.parse_tag(tag) == expected


@pytest.mark.parametrize("tag", ["1.0.0", "v1.0", "v1.0.0-", "v1.0.0 rc1", "vX.Y.Z", "v1.0.0/x"])
def test_parse_tag_rejects(tools, tag):
    with pytest.raises(tools.ReleaseError):
        tools.parse_tag(tag)


def test_check_tag(tools):
    tools.check_tag("v1.0.0", "1.0.0")
    tools.check_tag("v1.0.0-rc1", "1.0.0")
    with pytest.raises(tools.ReleaseError, match="package version is 1.0.0"):
        tools.check_tag("v1.0.1", "1.0.0")


def test_the_real_tag_check(tools):
    version = tools.package_version()
    assert tools.main(["check-tag", f"v{version}"]) == 0
    assert tools.main(["check-tag", f"v{version}-rc1"]) == 0
    assert tools.main(["check-tag", "v0.0.1"]) == 1


def test_section_matches_the_version_not_the_date(tools):
    notes = tools.changelog_section(CHANGELOG, "1.0.0")
    assert notes == "The first public release.\n\n### Added\n- Recording.\n- Export.\n"
    assert tools.changelog_section(CHANGELOG, "1.1.0") == "### Added\n- Something new.\n"
    assert tools.changelog_section(CHANGELOG, "Unreleased").startswith("### Fixed")


def test_section_missing_or_empty(tools):
    with pytest.raises(tools.ReleaseError, match=r"no '## \[2.0.0\]' section"):
        tools.changelog_section(CHANGELOG, "2.0.0")
    with pytest.raises(tools.ReleaseError, match="empty"):
        tools.changelog_section("## [1.0.0] - x\n\n[1.0.0]: https://x.invalid\n", "1.0.0")
    with pytest.raises(tools.ReleaseError):
        tools.changelog_section(CHANGELOG, "1.0")  # not a prefix match


def test_release_notes_for_a_pre_release_fall_back_to_the_base_version(tools):
    assert tools.release_notes("v1.0.0-rc1", CHANGELOG).startswith("The first public release.")
    own = CHANGELOG.replace("## [1.1.0] - 2027-01-02", "## [1.0.0-rc1] - 2027-01-02")
    assert tools.release_notes("v1.0.0-rc1", own) == "### Added\n- Something new.\n"
    with pytest.raises(tools.ReleaseError):
        tools.release_notes("v2.0.0", CHANGELOG)


def test_the_real_changelog_has_this_versions_notes(tools, tmp_path):
    out = tmp_path / "notes.md"
    assert tools.main(["notes", f"v{tools.package_version()}", "--out", str(out)]) == 0
    notes = out.read_text(encoding="utf-8")
    assert "### Added" in notes and "## [" not in notes and "releases/tag" not in notes


def test_checksums(tools, tmp_path):
    first = tmp_path / "a.zip"
    second = tmp_path / "b.bin"
    first.write_bytes(b"zip bytes")
    second.write_bytes(b"x" * (3 << 20))  # more than one read block
    out = tmp_path / "SHA256SUMS.txt"
    assert tools.main(["checksums", str(first), str(second), "--out", str(out)]) == 0
    data = out.read_bytes()
    assert b"\r" not in data
    expected = (
        f"{hashlib.sha256(b'zip bytes').hexdigest()}  a.zip\n"
        f"{hashlib.sha256(b'x' * (3 << 20)).hexdigest()}  b.bin\n"
    )
    assert data.decode("ascii") == expected
