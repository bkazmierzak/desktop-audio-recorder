"""Undo data that can move to a temporary file when memory runs short (no Qt).

An hour-long clip is 1.3 GB, so one whole-clip gain or crop keeps 1.3 GB for undo; with the
in-memory cap (500 MB) that edit alone used to evict every older undo step. Now ``EditHistory``
first *spills* the oldest large undo buffers (``Payload``) to disk through a ``SpillStore``, and
evicts only when the disk budget (or the disk itself) is full too.

* **Payload** wraps one undo buffer (a float32 ``[frames, 2]`` array). It is in memory, being
  written (``spilling``: still in memory, readable), or on disk. Commands read it with
  ``read_into(out)`` (straight from the file into the clip, no temporary), ``array()`` (a fresh
  copy) or ``take()`` (the buffer becomes the caller's, e.g. a crop's undo returning the old
  array as the live clip, so it must not stay on disk: the file is dropped and the payload holds
  the array in memory again).
* **SpillStore** owns a folder ``<temp>/undo-<pid>-<random>`` with a ``.lock`` file held open, so
  another instance's start-up clean-up (``remove_stale``) can't delete it while it is in use. One
  background thread writes the files (``write`` releases the GIL), so an edit never waits for
  the disk just to push an old entry out: writing an hour's undo copy (1.3 GB) takes 3-10 s
  once the drive's write cache is full. A big edit (already off the GUI thread) only waits
  while *more than one* write is pending (``wait_pending(1)``), so at most one buffer on its way
  to disk is in memory besides the clip, the newest undo copy and the one being made.
* Writes fail softly: if a file can't be written (disk full, folder gone), the payload stays in
  memory and is marked unspillable; the history then evicts as it did before.
* Files go away when their payload is discarded or garbage-collected, and the folder when the
  store closes (the window does that at shutdown).
"""

from __future__ import annotations

import logging
import os
import queue
import secrets
import shutil
import threading
import weakref
from pathlib import Path

import numpy as np

from desktop_audio_recorder.audio import buffers

log = logging.getLogger(__name__)

SPILL_PREFIX = "undo-"
LOCK_NAME = ".lock"
MIN_SPILL_BYTES = 32 * 1024 * 1024  # smaller buffers stay in memory (~87 s of stereo float32)
DEFAULT_MAX_DISK_BYTES = 10 * 1024**3  # ~7 whole-clip undo steps of an hour-long clip
MIN_FREE_BYTES = 2 * 1024**3  # never fill the disk below this
IO_CHUNK_BYTES = 64 * 1024 * 1024

_MEMORY, _SPILLING, _DISK, _GONE = "memory", "spilling", "disk", "gone"


def _remove_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        log.warning("Could not delete the undo file %s", path, exc_info=True)


def _cleanup(path: str, store: SpillStore, nbytes: int) -> None:
    """A spilled payload was garbage-collected (or discarded): its file and budget go."""
    _remove_file(path)
    store.release(nbytes)


def _write_array(path: Path, array: np.ndarray) -> None:
    view = memoryview(np.ascontiguousarray(array)).cast("B")
    with open(path, "wb") as f:
        for i in range(0, len(view), IO_CHUNK_BYTES):
            f.write(view[i : i + IO_CHUNK_BYTES])


def _read_into(path: str, out: np.ndarray) -> None:
    if not out.flags.c_contiguous:
        raise ValueError("read_into needs a C-contiguous array")
    view = memoryview(out).cast("B")
    with open(path, "rb") as f:
        pos = 0
        while pos < len(view):
            n = f.readinto(view[pos : pos + IO_CHUNK_BYTES])
            if not n:
                raise OSError(f"the undo file {path} is shorter than expected")
            pos += n


class Payload:
    """One undo buffer, in memory or in a ``SpillStore`` file (see the module docstring).

    Thread safety: the history's thread calls everything; the store's writer thread only runs
    ``_write``. The state lock is never held during disk I/O except to finish a read of a file
    that a concurrent ``discard`` would otherwise delete.
    """

    def __init__(self, array: np.ndarray) -> None:
        self._array: np.ndarray | None = array
        self.shape = array.shape
        self.dtype = array.dtype
        self.size_bytes = int(array.nbytes)
        self._state = _MEMORY
        self._path: str | None = None
        self._store: SpillStore | None = None
        self._lock = threading.Lock()
        self._done = threading.Event()  # set when no write is pending
        self._done.set()
        self.unspillable = False  # a write failed: keep it in memory
        self._finalizer: weakref.finalize | None = None  # owns the file once it is on disk
        self._reserved = False  # holds a share of the store's budget for a pending write

    # ------------------------------------------------------------------ queries

    @property
    def frames(self) -> int:
        return int(self.shape[0])

    @property
    def nbytes(self) -> int:
        """Memory this payload counts for (0 once a spill has been requested)."""
        return self.size_bytes if self._state == _MEMORY else 0

    @property
    def disk_bytes(self) -> int:
        return self.size_bytes if self._state in (_SPILLING, _DISK) else 0

    @property
    def on_disk(self) -> bool:
        """Whether reading it means reading a file (it is spilled, or being spilled)."""
        return self._state in (_SPILLING, _DISK)

    @property
    def state(self) -> str:
        return self._state

    def spillable(self, min_bytes: int = MIN_SPILL_BYTES) -> bool:
        return self._state == _MEMORY and not self.unspillable and self.size_bytes >= min_bytes

    # ------------------------------------------------------------------ reading

    def read_into(self, out: np.ndarray) -> None:
        """Copy the payload into ``out`` (same shape and dtype)."""
        if out.shape != self.shape or out.dtype != self.dtype:
            raise ValueError(f"payload {self.shape} {self.dtype} doesn't fit {out.shape}")
        with self._lock:
            state, array, path = self._state, self._array, self._path
            if state == _GONE:
                raise RuntimeError("the undo data was discarded")
            if array is not None:  # in memory, or still being written
                np.copyto(out, array)
                return
            assert path is not None
            # On disk: read under the lock, so a concurrent discard can't delete the file.
            _read_into(path, out)

    def array(self) -> np.ndarray:
        """The array (in memory; don't modify it) or a freshly read copy (on disk)."""
        if self._array is not None:
            return self._array
        out = buffers.empty(self.shape, self.dtype)
        self.read_into(out)
        return out

    def take(self) -> np.ndarray:
        """The buffer itself, back in memory; any file is dropped. The caller may modify it
        (the payload then holds whatever the caller makes of it, as a plain array would)."""
        with self._lock:
            if self._state == _GONE:
                raise RuntimeError("the undo data was discarded")
            if self._state == _MEMORY:
                assert self._array is not None
                return self._array
            if self._state == _SPILLING:  # the writer finishes, sees MEMORY, drops its file
                self._state = _MEMORY
                self._unreserve()
                assert self._array is not None
                return self._array
            # _DISK
            assert self._path is not None
            out = buffers.empty(self.shape, self.dtype)
            _read_into(self._path, out)
            self._drop_file()
            self._array = out
            self._state = _MEMORY
            return out

    # ------------------------------------------------------------------ spilling

    def spill(self, store: SpillStore) -> bool:
        """Ask ``store`` to write this payload to disk (returns at once). False if it can't."""
        with self._lock:
            if self._state != _MEMORY or self.unspillable:
                return False
            if not store.reserve(self.size_bytes):
                return False
            self._store = store
            self._state = _SPILLING
            self._reserved = True
            self._done.clear()
        store.submit(self)
        return True

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for a pending write to finish."""
        return self._done.wait(timeout)

    def _unreserve(self) -> None:  # with the lock held
        """Give the pending write's budget back now (taken back, discarded or failed), so the
        history can use it at once instead of after the write."""
        if self._reserved and self._store is not None:
            self._store.release(self.size_bytes)
        self._reserved = False

    def _write(self) -> None:  # the store's writer thread
        store = self._store
        assert store is not None
        try:
            with self._lock:
                array = self._array
                if self._state != _SPILLING or array is None:
                    self._unreserve()  # taken back or discarded before the write started
                    return
            path = store.new_path()
            try:
                _write_array(path, array)
            except Exception:
                _remove_file(str(path))
                with self._lock:
                    if self._state == _SPILLING:
                        self._state = _MEMORY
                    self.unspillable = True
                    self._unreserve()
                log.warning(
                    "Could not write %.0f MB of undo data to %s; keeping it in memory",
                    self.size_bytes / 2**20,
                    path,
                    exc_info=True,
                )
                return
            with self._lock:
                if self._state == _SPILLING:
                    self._path = str(path)
                    self._array = None  # the memory goes (via buffers' release thread)
                    self._state = _DISK
                    self._finalizer = weakref.finalize(
                        self, _cleanup, str(path), store, self.size_bytes
                    )
                    self._finalizer.atexit = False
                    self._reserved = False  # the finalizer releases the budget now
                    log.info("Moved %.0f MB of undo data to disk", self.size_bytes / 2**20)
                    return
                self._unreserve()  # taken back (or discarded) while it was being written
            _remove_file(str(path))
        finally:
            self._done.set()

    def _drop_file(self) -> None:  # with the lock held
        if self._path is not None:
            finalizer, self._finalizer = self._finalizer, None
            if finalizer is not None:
                finalizer()  # deletes the file and releases the budget (once)
            else:
                _remove_file(self._path)
            self._path = None

    def discard(self) -> None:
        """Drop the data (memory and file); the payload can't be read any more."""
        with self._lock:
            state = self._state
            self._state = _GONE
            self._array = None
            if state == _DISK:
                self._drop_file()
            elif state == _SPILLING:  # the writer sees GONE and deletes its file
                self._unreserve()


class SpillStore:
    """A folder of undo files with a disk budget, and the thread that writes them."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_bytes: int = DEFAULT_MAX_DISK_BYTES,
        min_free_bytes: int = MIN_FREE_BYTES,
        min_spill_bytes: int = MIN_SPILL_BYTES,
    ) -> None:
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes
        self.min_spill_bytes = min_spill_bytes
        self._folder: Path | None = None
        self._lock_file = None
        self._lock = threading.Lock()
        self._used = 0
        self._counter = 0
        self._queue: queue.Queue[Payload | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._closed = False
        self.failed = False  # the folder couldn't be created: never try again
        self._pending = 0  # writes submitted and not finished
        self._idle = threading.Condition(self._lock)

    @property
    def used_bytes(self) -> int:
        return self._used

    @property
    def folder(self) -> Path | None:
        return self._folder

    def _ensure_folder(self) -> Path:  # with self._lock held
        if self._folder is None:
            self.root.mkdir(parents=True, exist_ok=True)
            folder = self.root / f"{SPILL_PREFIX}{os.getpid()}-{secrets.token_hex(4)}"
            folder.mkdir()
            # Held open (without delete sharing) for as long as the store lives, so a second
            # instance's ``remove_stale`` sees the folder is in use.
            self._lock_file = open(folder / LOCK_NAME, "wb")  # noqa: SIM115
            self._folder = folder
            log.info("Undo data can spill to %s", folder)
        return self._folder

    def reserve(self, nbytes: int) -> bool:
        """Claim ``nbytes`` of the budget for a new file (False: over budget, low disk)."""
        with self._lock:
            if self._closed or self.failed:
                return False
            if self._used + nbytes > self.max_bytes:
                return False
            try:
                folder = self._ensure_folder()
                free = shutil.disk_usage(folder).free
            except OSError:
                log.warning("Undo data can't spill to %s", self.root, exc_info=True)
                self.failed = True
                return False
            if free - nbytes < self.min_free_bytes:
                log.info("Not moving undo data to disk: only %.1f GB free", free / 1024**3)
                return False
            self._used += nbytes
            return True

    def release(self, nbytes: int) -> None:
        with self._lock:
            self._used = max(0, self._used - nbytes)

    def new_path(self) -> Path:
        with self._lock:
            self._counter += 1
            folder = self._ensure_folder()
            return folder / f"{self._counter:06d}.f32"

    @property
    def pending(self) -> int:
        """Writes submitted and not finished yet."""
        return self._pending

    def submit(self, payload: Payload) -> None:
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._writer, name="undo-spill", daemon=True)
                self._thread.start()
            self._pending += 1
        self._queue.put(payload)

    def _writer(self) -> None:
        while True:
            payload = self._queue.get()
            try:
                if payload is None:
                    return
                payload._write()
            except Exception:  # never let the thread die
                log.exception("Undo spill failed")
            finally:
                if payload is not None:
                    with self._idle:
                        self._pending -= 1
                        self._idle.notify_all()
                payload = None  # don't keep the last one alive until the next write
                self._queue.task_done()

    def wait_pending(self, at_most: int, timeout: float | None = 60.0) -> bool:
        """Wait until at most ``at_most`` writes are pending."""
        with self._idle:
            return self._idle.wait_for(lambda: self._pending <= at_most, timeout)

    def wait_idle(self, timeout: float | None = 60.0) -> bool:
        """Wait until every submitted write has finished."""
        return self.wait_pending(0, timeout)

    def close(self) -> None:
        """Stop the writer and delete the folder (payloads still alive can't be read after)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
        if thread is not None:
            self._queue.put(None)
            thread.join(timeout=30)
        if self._lock_file is not None:
            self._lock_file.close()
            self._lock_file = None
        if self._folder is not None:
            shutil.rmtree(self._folder, ignore_errors=True)
            log.info("Removed the undo folder %s", self._folder)

    @staticmethod
    def remove_stale(root: str | Path) -> int:
        """Delete undo folders left behind by instances that are gone (crashed or killed);
        returns how many. A folder whose ``.lock`` can't be deleted is in use and is kept."""
        removed = 0
        try:
            folders = [p for p in Path(root).glob(f"{SPILL_PREFIX}*") if p.is_dir()]
        except OSError:
            return 0
        for folder in folders:
            try:
                (folder / LOCK_NAME).unlink(missing_ok=True)
            except OSError:  # held open by a running instance
                continue
            shutil.rmtree(folder, ignore_errors=True)
            if not folder.exists():
                removed += 1
                log.info("Removed the stale undo folder %s", folder)
        return removed


__all__ = [
    "DEFAULT_MAX_DISK_BYTES",
    "MIN_SPILL_BYTES",
    "Payload",
    "SpillStore",
]
