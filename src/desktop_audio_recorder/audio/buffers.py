"""Large sample buffers whose memory goes back to Windows without stalling the GUI (no Qt).

Freeing a large numpy array holds the GIL while Windows releases its pages: ~45-55 ms per hour
of stereo float32 audio (1.29 GiB), measured on this machine. Dropping an old clip after an
edit, or an evicted undo copy, would freeze every Python thread (the GUI included) for that
long; a 3-hour clip, ~150 ms.

``empty()``/``copy()`` therefore allocate arrays of ``LARGE_BYTES`` or more straight from
``VirtualAlloc``. When the last view of such an array goes away, a ``weakref.finalize`` hands the
address to a background thread that calls ``VirtualFree`` through ctypes, which releases the GIL
during the call. Smaller arrays (and other platforms) use ``np.empty``.

Lifetime: the numpy array's ``base`` chain ends at the ctypes buffer the finalizer watches, so
the memory stays valid as long as any view of it exists (the playback callback, the waveform
pyramid, an undo command, …).
"""

from __future__ import annotations

import logging
import math
import queue
import sys
import threading
import weakref

import numpy as np

log = logging.getLogger(__name__)

LARGE_BYTES = 64 * 1024 * 1024

_MEM_COMMIT = 0x1000
_MEM_RESERVE = 0x2000
_MEM_RELEASE = 0x8000
_PAGE_READWRITE = 0x04

_kernel32 = None
_queue: queue.Queue[tuple[int, int]] = queue.Queue()
_thread: threading.Thread | None = None
_lock = threading.Lock()
_stats = {"allocated": 0, "released": 0, "live_bytes": 0}


def _api():
    global _kernel32
    if _kernel32 is None:
        import ctypes
        import ctypes.wintypes as wt

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.VirtualAlloc.restype = ctypes.c_void_p
        k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD, wt.DWORD]
        k32.VirtualFree.restype = wt.BOOL
        k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD]
        _kernel32 = k32
    return _kernel32


def _releaser() -> None:  # background thread
    k32 = _api()
    while True:
        addr, nbytes = _queue.get()
        try:
            if not k32.VirtualFree(addr, 0, _MEM_RELEASE):
                log.error("VirtualFree failed for %d bytes", nbytes)
        finally:
            with _lock:
                _stats["released"] += 1
                _stats["live_bytes"] -= nbytes
            _queue.task_done()


def _schedule_release(addr: int, nbytes: int) -> None:  # any thread, from the finalizer
    _queue.put((addr, nbytes))


def _ensure_thread() -> None:
    global _thread
    with _lock:
        if _thread is None:
            _thread = threading.Thread(target=_releaser, name="buffer-release", daemon=True)
            _thread.start()


def _virtual_empty(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    import ctypes

    nbytes = math.prod(shape) * dtype.itemsize
    addr = _api().VirtualAlloc(None, nbytes, _MEM_COMMIT | _MEM_RESERVE, _PAGE_READWRITE)
    if not addr:
        raise MemoryError(f"VirtualAlloc of {nbytes / 2**20:.0f} MiB failed")
    _ensure_thread()
    raw = (ctypes.c_char * nbytes).from_address(addr)
    finalizer = weakref.finalize(raw, _schedule_release, addr, nbytes)
    finalizer.atexit = False  # never free memory that may still be in use at exit
    with _lock:
        _stats["allocated"] += 1
        _stats["live_bytes"] += nbytes
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def empty(shape: int | tuple[int, ...], dtype=np.float32) -> np.ndarray:
    """An uninitialised C-contiguous array (see module docstring for large ones)."""
    shape = (shape,) if isinstance(shape, int) else tuple(int(s) for s in shape)
    dtype = np.dtype(dtype)
    if sys.platform == "win32" and math.prod(shape) * dtype.itemsize >= max(1, LARGE_BYTES):
        return _virtual_empty(shape, dtype)
    return np.empty(shape, dtype=dtype)


def copy(a: np.ndarray) -> np.ndarray:
    """A C-contiguous copy of ``a``."""
    out = empty(a.shape, a.dtype)
    np.copyto(out, a)
    return out


def wait_released(timeout: float = 5.0) -> bool:
    """Wait until every buffer handed to the background thread has been freed (for tests)."""
    done = threading.Event()

    def join() -> None:
        _queue.join()
        done.set()

    threading.Thread(target=join, daemon=True).start()
    return done.wait(timeout)


def stats() -> dict[str, int]:
    """Counts of large buffers allocated and released, and bytes still live."""
    with _lock:
        return dict(_stats)
