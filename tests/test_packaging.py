"""packaging/build.py's third-party notices (without building anything)."""

import importlib.metadata as metadata
import importlib.util
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
        "Copyright (c) 1996-2024  Anthony Green",  # libffi, in the Python runtime
    ):
        assert needle in text, needle
    # A license file that two distributions share is written once, then referred to.
    assert text.count("Licensees holding valid commercial Qt licenses") == 1
