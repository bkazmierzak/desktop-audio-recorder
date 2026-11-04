"""Leftover temp recordings (audio/recovery.py)."""

import os
import shutil
import struct

import numpy as np
import pytest
import soundfile as sf

from desktop_audio_recorder.audio import recovery
from desktop_audio_recorder.audio.recorder import TEMP_PREFIX

RATE = 48000


def crashed_recording(folder, name="recording-20261103-101500-abc.wav", seconds=1.0):
    """What a killed app leaves behind: a float WAV copied while still open (sizes unwritten)."""
    live = folder / "live.wav"
    frames = round(seconds * RATE)
    data = np.column_stack([np.linspace(-0.5, 0.5, frames), np.linspace(0.5, -0.5, frames)]).astype(
        np.float32
    )
    f = sf.SoundFile(live, mode="w", samplerate=RATE, channels=2, subtype="FLOAT", format="WAV")
    f.write(data)
    f.flush()
    target = folder / name
    shutil.copy(live, target)
    f.close()
    live.unlink()
    return target, data


def test_a_crashed_file_has_no_sizes_but_is_found_with_its_length(tmp_path):
    path, data = crashed_recording(tmp_path)
    layout = recovery.read_layout(path)
    assert layout.declared_frames == 0
    assert (layout.sample_rate, layout.channels, layout.block_align) == (RATE, 2, 8)
    (leftover,) = recovery.find_leftovers(tmp_path)
    assert leftover.path == path
    assert leftover.frames == len(data)
    assert leftover.duration == pytest.approx(1.0)
    assert not leftover.empty


def test_repair_writes_the_sizes(tmp_path):
    path, data = crashed_recording(tmp_path)
    assert recovery.repair(path) == len(data)
    layout = recovery.read_layout(path)
    assert layout.declared_frames == len(data)
    raw = path.read_bytes()
    assert struct.unpack_from("<I", raw, 4)[0] == len(raw) - 8
    assert struct.unpack_from("<I", raw, layout.fact_offset)[0] == len(data)
    back, rate = sf.read(path, dtype="float32")
    assert rate == RATE
    np.testing.assert_array_equal(back, data)


def test_repair_cuts_a_partial_last_frame(tmp_path):
    path, data = crashed_recording(tmp_path)
    with open(path, "ab") as handle:
        handle.write(b"\x01\x02\x03")  # half a frame
    assert recovery.repair(path) == len(data)
    assert os.path.getsize(path) == recovery.read_layout(path).data_offset + len(data) * 8


def test_repair_leaves_a_good_file_alone(tmp_path):
    path = tmp_path / f"{TEMP_PREFIX}good.wav"
    sf.write(path, np.zeros((100, 2), np.float32), RATE, subtype="FLOAT")
    before = path.read_bytes()
    assert recovery.repair(path) == 100
    assert path.read_bytes() == before


def test_newest_first_and_only_our_files(tmp_path):
    old, _ = crashed_recording(tmp_path, f"{TEMP_PREFIX}old.wav")
    new, _ = crashed_recording(tmp_path, f"{TEMP_PREFIX}new.wav")
    os.utime(old, (1_700_000_000, 1_700_000_000))
    os.utime(new, (1_800_000_000, 1_800_000_000))
    (tmp_path / "other.wav").write_bytes(b"not ours")
    assert [leftover.path for leftover in recovery.find_leftovers(tmp_path)] == [new, old]


def test_excluded_and_in_use_files_are_skipped(tmp_path):
    mine, _ = crashed_recording(tmp_path, f"{TEMP_PREFIX}mine.wav")
    busy = tmp_path / f"{TEMP_PREFIX}busy.wav"
    with sf.SoundFile(busy, mode="w", samplerate=RATE, channels=2, subtype="FLOAT") as f:
        f.write(np.zeros((10, 2), np.float32))
        f.flush()
        if os.name == "nt":  # another instance is recording into it
            assert recovery.is_in_use(busy)
            assert recovery.find_leftovers(tmp_path, exclude=(mine,)) == []
    assert not recovery.is_in_use(busy)
    assert [x.path for x in recovery.find_leftovers(tmp_path, exclude=(mine,))] == [busy]


def test_junk_and_empty_files_are_listed_as_empty(tmp_path):
    junk = tmp_path / f"{TEMP_PREFIX}junk.wav"
    junk.write_bytes(b"RIFF\x00\x00")
    empty, _ = crashed_recording(tmp_path, f"{TEMP_PREFIX}empty.wav", seconds=0)
    leftovers = {x.path: x for x in recovery.find_leftovers(tmp_path)}
    assert leftovers[junk].empty
    assert leftovers[empty].empty
    with pytest.raises(recovery.NotAWave):
        recovery.read_layout(junk)


def test_missing_folder(tmp_path):
    assert recovery.find_leftovers(tmp_path / "nope") == []


def test_delete(tmp_path):
    a, _ = crashed_recording(tmp_path, f"{TEMP_PREFIX}a.wav")
    assert recovery.delete([a, tmp_path / "missing.wav"]) == 2
    assert not a.exists()


def test_is_in_use_for_a_missing_file(tmp_path):
    assert not recovery.is_in_use(tmp_path / "missing.wav")
