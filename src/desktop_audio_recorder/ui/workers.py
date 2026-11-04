"""Run blocking work off the GUI thread and deliver the outcome back on it."""

from __future__ import annotations

import itertools
import logging
import threading
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QCoreApplication, QObject, Qt, QThreadPool, Signal, Slot

log = logging.getLogger(__name__)

SuccessCallback = Callable[[Any], None]
ErrorCallback = Callable[[BaseException], None]


class ProgressJob:
    """Progress and cancellation shared by a background task and the GUI (any thread).

    The task calls ``report(fraction)`` and polls ``is_cancelled()``; the GUI polls ``fraction``
    from a timer and calls ``cancel()``. A float assignment is atomic in CPython, and the cancel
    flag is a ``threading.Event``.
    """

    def __init__(self) -> None:
        self._fraction = 0.0
        self._cancel = threading.Event()

    @property
    def fraction(self) -> float:
        return self._fraction

    def report(self, fraction: float) -> None:
        self._fraction = min(1.0, max(0.0, float(fraction)))

    def cancel(self) -> None:
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()


class SerialTaskRunner(QObject):
    """Runs callables one at a time, in submission order, on a single long-lived thread.

    All PortAudio work (device refresh, ``recorder.start()``, ``recorder.stop()``) goes through
    one runner. That serialises ``Pa_Initialize``/``Pa_Terminate`` (not thread-safe) and keeps
    stream open/close on the same thread (WASAPI/COM is happiest that way).

    ``on_success(result)`` / ``on_error(exc)`` are called on the thread that owns the runner
    (the GUI thread), via a queued signal.
    """

    _finished = Signal(int, object, object)  # task id, result, exception

    def __init__(self, parent: QObject | None = None, name: str = "audio") -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._pool.setExpiryTimeout(-1)  # keep the one thread forever
        self._pool.setObjectName(name)
        self._ids = itertools.count(1)
        self._callbacks: dict[int, tuple[SuccessCallback | None, ErrorCallback | None]] = {}
        self._pending = 0
        self._pending_lock = threading.Lock()
        self._finished.connect(self._dispatch, Qt.ConnectionType.QueuedConnection)

    @property
    def busy(self) -> bool:
        """Whether a submitted task hasn't run to completion yet."""
        with self._pending_lock:
            return self._pending > 0

    def submit(
        self,
        fn: Callable[[], Any],
        on_success: SuccessCallback | None = None,
        on_error: ErrorCallback | None = None,
    ) -> int:
        task_id = next(self._ids)
        self._callbacks[task_id] = (on_success, on_error)
        with self._pending_lock:
            self._pending += 1
        self._pool.start(lambda: self._run(task_id, fn))
        return task_id

    def run_blocking(self, fn: Callable[[], Any], timeout_ms: int = 10_000) -> Any:
        """Run ``fn`` on the worker thread and wait for it (for shutdown). Re-raises errors."""
        box: dict[str, Any] = {}

        def task() -> None:
            try:
                box["result"] = fn()
            except BaseException as exc:
                box["error"] = exc

        with self._pending_lock:
            self._pending += 1
        self._pool.start(lambda: self._run(0, task))
        if not self._pool.waitForDone(timeout_ms):
            raise TimeoutError("audio thread did not finish in time")
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def wait(self, timeout_ms: int = -1) -> bool:
        return self._pool.waitForDone(timeout_ms)

    def drain(self, timeout_ms: int = 10_000) -> bool:
        """Wait for all tasks, then deliver their queued callbacks now (for shutdown)."""
        done = self._pool.waitForDone(timeout_ms)
        QCoreApplication.sendPostedEvents(self, 0)
        return done

    # ------------------------------------------------------------------ internals

    def _run(self, task_id: int, fn: Callable[[], Any]) -> None:  # worker thread
        result: Any = None
        error: BaseException | None = None
        try:
            result = fn()
        except BaseException as exc:
            error = exc
        finally:
            with self._pending_lock:
                self._pending -= 1
        if task_id:
            try:
                self._finished.emit(task_id, result, error)
            except RuntimeError:  # runner deleted while the task ran (app shutting down)
                log.debug("Dropping result of task %d: runner deleted", task_id)

    @Slot(int, object, object)
    def _dispatch(self, task_id: int, result: Any, error: BaseException | None) -> None:
        on_success, on_error = self._callbacks.pop(task_id, (None, None))
        if error is None:
            if on_success is not None:
                on_success(result)
        elif on_error is not None:
            on_error(error)
        else:
            log.error("Background task failed", exc_info=error)
