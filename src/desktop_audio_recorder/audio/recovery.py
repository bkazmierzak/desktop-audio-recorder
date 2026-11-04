"""Recordings left in the temp folder by a crash (no Qt here).

The recorder writes ``%TEMP%/DesktopAudioRecorder/recording-<stamp>-<random>.wav`` and the
window deletes it when the clip is discarded or the app closes. If the app is killed (crash,
Task Manager, power loss, a forced logoff), the file stays behind. libsndfile writes the WAV
sizes only when the file is closed, so such a file says "0 frames" (RIFF size 8, data size 0)
although the audio is all there (it is flushed every second). ``repair()`` rewrites the sizes
from the file length so any reader accepts it.

A file that is still open (another instance of the app is recording into it) is skipped:
``is_in_use()`` tries to open it with no sharing, which Windows refuses while any other handle
is open.
"""

from __future__ import annotations

import logging
import os
import struct
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from desktop_audio_recorder.audio.recorder import TEMP_PREFIX, default_temp_dir

log = logging.getLogger(__name__)

_HEADER_BYTES = 64 * 1024  # the data chunk starts within this (libsndfile: byte 80)
_WAVE_FORMAT_PCM = 1
_WAVE_FORMAT_IEEE_FLOAT = 3
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE


class NotAWave(ValueError):
    """The file isn't a WAV this app could have written."""


@dataclass(frozen=True, slots=True)
class WaveLayout:
    sample_rate: int
    channels: int
    block_align: int  # bytes per frame
    data_offset: int  # where the samples start
    data_size_offset: int  # where the data chunk's size field is
    fact_offset: int | None  # the fact chunk's sample count field, if any
    declared_frames: int  # what the header says


@dataclass(frozen=True, slots=True)
class Leftover:
    path: Path
    modified: datetime
    size: int
    sample_rate: int
    frames: int  # from the file length (what repair() will declare)

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0

    @property
    def empty(self) -> bool:
        return self.frames == 0


def read_layout(path: str | Path) -> WaveLayout:
    """Parse the RIFF header up to the data chunk."""
    with open(path, "rb") as handle:
        head = handle.read(_HEADER_BYTES)
    if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
        raise NotAWave(f"{Path(path).name} is not a WAV file")
    pos = 12
    fmt = None
    fact_offset = None
    while pos + 8 <= len(head):
        chunk, size = head[pos : pos + 4], struct.unpack_from("<I", head, pos + 4)[0]
        if chunk == b"fmt " and size >= 16:
            fmt = struct.unpack_from("<HHIIHH", head, pos + 8)
        elif chunk == b"fact" and size >= 4:
            fact_offset = pos + 8
        elif chunk == b"data":
            if fmt is None:
                raise NotAWave(f"{Path(path).name}: no format chunk before the data")
            tag, channels, rate, _byte_rate, block_align, _bits = fmt
            if tag not in (_WAVE_FORMAT_PCM, _WAVE_FORMAT_IEEE_FLOAT, _WAVE_FORMAT_EXTENSIBLE):
                raise NotAWave(f"{Path(path).name}: unsupported format {tag:#x}")
            if channels < 1 or rate < 1 or block_align < 1:
                raise NotAWave(f"{Path(path).name}: invalid format chunk")
            declared = 0 if size == 0xFFFFFFFF else size // block_align
            return WaveLayout(rate, channels, block_align, pos + 8, pos + 4, fact_offset, declared)
        pos += 8 + size + (size & 1)
    raise NotAWave(f"{Path(path).name}: no data chunk")


def frames_on_disk(path: str | Path, layout: WaveLayout) -> int:
    return max(0, os.path.getsize(path) - layout.data_offset) // layout.block_align


def repair(path: str | Path) -> int:
    """Make the header's sizes match the audio in the file; returns the number of frames.

    A trailing partial frame is cut off. RF64 files (over 4 GB) are left as they are: libsndfile
    reads those from the file length anyway.
    """
    layout = read_layout(path)
    frames = frames_on_disk(path, layout)
    data_bytes = frames * layout.block_align
    end = layout.data_offset + data_bytes
    if data_bytes > 0xFFFFFFFF - layout.data_offset:
        log.warning("%s is too big for a plain WAV header; not repaired", path)
        return frames
    if os.path.getsize(path) != end:
        os.truncate(path, end)
    if layout.declared_frames == frames and end - 8 == _riff_size(path):
        return frames
    with open(path, "r+b") as handle:
        handle.seek(4)
        handle.write(struct.pack("<I", end - 8))
        handle.seek(layout.data_size_offset)
        handle.write(struct.pack("<I", data_bytes))
        if layout.fact_offset is not None:
            handle.seek(layout.fact_offset)
            handle.write(struct.pack("<I", frames))
    log.info("Repaired %s: %d frames (header said %d)", path, frames, layout.declared_frames)
    return frames


def _riff_size(path: str | Path) -> int:
    with open(path, "rb") as handle:
        handle.seek(4)
        return struct.unpack("<I", handle.read(4))[0]


def is_in_use(path: str | Path) -> bool:
    """Whether another handle has the file open (Windows; elsewhere always ``False``)."""
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    generic_read, open_existing, no_sharing = 0x80000000, 3, 0
    handle = kernel32.CreateFileW(str(path), generic_read, no_sharing, None, open_existing, 0, None)
    if handle == wintypes.HANDLE(-1).value:
        return ctypes.get_last_error() == 32  # ERROR_SHARING_VIOLATION
    kernel32.CloseHandle(handle)
    return False


def find_leftovers(
    folder: str | Path | None = None, exclude: tuple[str | Path, ...] = ()
) -> list[Leftover]:
    """Temp recordings that no running recording is using, newest first.

    Files that aren't readable WAVs are listed as empty (``frames == 0``), so the caller can
    delete them.
    """
    folder = Path(folder) if folder is not None else default_temp_dir()
    excluded = {os.path.normcase(os.path.abspath(p)) for p in exclude}
    found = []
    try:
        candidates = list(folder.glob(f"{TEMP_PREFIX}*.wav"))
    except OSError:
        return []
    for path in candidates:
        if os.path.normcase(os.path.abspath(path)) in excluded:
            continue
        try:
            if is_in_use(path):
                log.info("Skipping %s: in use", path.name)
                continue
            stat = path.stat()
        except OSError:
            continue
        try:
            layout = read_layout(path)
            rate, frames = layout.sample_rate, frames_on_disk(path, layout)
        except (OSError, NotAWave) as exc:
            log.info("Unreadable leftover %s: %s", path.name, exc)
            rate, frames = 0, 0
        found.append(
            Leftover(path, datetime.fromtimestamp(stat.st_mtime), stat.st_size, rate, frames)
        )
    found.sort(key=lambda leftover: leftover.modified, reverse=True)
    return found


def delete(paths: list[Path]) -> int:
    """Delete these files; returns how many were deleted."""
    deleted = 0
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
            deleted += 1
        except OSError:
            log.warning("Could not delete %s", path, exc_info=True)
    return deleted
