"""packaging/build.py's third-party notices (without building anything)."""

import decimal
import importlib.metadata as metadata
import importlib.util
import pyexpat
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def build():
    spec = importlib.util.spec_from_file_location("build_script", ROOT / "packaging" / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("name", "first_line"),
    [
        ("GPL-3.0.txt", "GNU GENERAL PUBLIC LICENSE"),
        ("LGPL-3.0.txt", "GNU LESSER GENERAL PUBLIC LICENSE"),
        ("LGPL-2.1.txt", "GNU LESSER GENERAL PUBLIC LICENSE"),
        ("Apache-2.0.txt", "Apache License"),
        ("PortAudio.txt", "PortAudio Portable Real-Time Audio Library"),
    ],
)
def test_vendored_license_texts(build, name, first_line):
    data = (build.LICENSES / name).read_bytes()
    assert b"\r" not in data
    assert data.decode("utf-8").strip().splitlines()[0].strip() == first_line


def test_bundled_names_map_to_distributions(build):
    names = ["PySide6", "shiboken6", "_soundfile_data", "lameenc", "_portaudiowpatch", "numpy"]
    names += ["desktop_audio_recorder", "python312", "VCRUNTIME140", "json", "_ctypes", "mystery"]
    dists, unknown = build.bundled_distributions(names)
    found = sorted(d.metadata["Name"].lower().replace("_", "-") for d in dists)
    expected = ["lameenc", "numpy", "pyaudiowpatch", "pyside6-essentials", "shiboken6"]
    assert found == sorted([*expected, "soundfile"])
    assert unknown == ["mystery"]  # the app, the runtime DLLs and the stdlib are expected


def test_license_texts_come_from_the_dist_info(build):
    texts = dict(build.license_texts(metadata.distribution("lameenc")))
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in texts["LICENSE"]
    numpy_texts = dict(build.license_texts(metadata.distribution("numpy")))
    assert "LICENSE.txt" in numpy_texts and len(numpy_texts) > 5  # nested licenses/ files
    scipy_texts = build.license_texts(metadata.distribution("scipy"))  # no License-File entries
    assert scipy_texts and "SciPy" in scipy_texts[0][1]


def test_notices_text(build):
    names = ("lameenc", "pyside6-essentials", "shiboken6", "soundfile")
    dists = [metadata.distribution(n) for n in names]
    text = build.notices_text(dists)
    assert text.startswith(f"Desktop Audio Recorder {build.version()}: third-party notices")
    for needle in (
        "LGPL components: separate, replaceable files",
        r"_internal\PySide6",
        r"_internal\_soundfile_data\libsndfile_x64.dll",
        r"_internal\lameenc.cp312-win_amd64.pyd",
        "PortAudio, twice",
        "lameenc 1.9.0",
        "Full license texts",
        "PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2",
        "Version 3, 29 June 2007",  # GPL-3.0 and LGPL-3.0
        "Version 2.1, February 1999",  # LGPL-2.1
        "Apache License",
        "Copyright (c) 1999-2011 Ross Bencina and Phil Burk",  # PortAudio
        "Copyright (c) 1996-2021  Anthony Green",  # libffi 3.4.2, in the Python runtime
        "Copyright (c) 2008-2020 Stefan Krah",  # libmpdec 2.5.1
        "Copyright (C) 1995-2026 Jean-loup Gailly and Mark Adler",  # zlib 1.3.2
    ):
        assert needle in text, needle
    # A license file that two distributions share is written once, then referred to.
    assert text.count("Licensees holding valid commercial Qt licenses") == 1


def test_python_runtime_licenses_match_this_python(build):
    """python-runtime.txt names the library versions its copyright lines were checked against.

    If this fails, the uv-managed Python changed: compare each section of
    packaging/licenses/python-runtime.txt with the new version's upstream license, then update
    the versions in its headings.
    """
    text = (build.LICENSES / "python-runtime.txt").read_text(encoding="utf-8")
    expat = pyexpat.EXPAT_VERSION.removeprefix("expat_")
    for heading in (
        f"Python {sys.version.split()[0]}",
        f"zlib {zlib.ZLIB_RUNTIME_VERSION} (inside python312.dll)",
        f"Expat {expat} (pyexpat.pyd)",
        f"libmpdec {decimal.__libmpdec_version__} (_decimal.pyd)",
    ):
        assert heading in text, heading
    # libffi and XZ don't report their versions; python-build-standalone's build tag pins them.
    tag_file = Path(sys.base_prefix) / "BUILD"
    if not tag_file.exists():
        pytest.skip("not a python-build-standalone Python (no BUILD file)")
    assert f"python-build-standalone {tag_file.read_text().strip()}" in text
