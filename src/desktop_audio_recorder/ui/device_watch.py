"""Qt side of ``audio/endpoints.py``: endpoint changes as a signal on the GUI thread.

``DeviceWatch.changed`` carries an ``EndpointEvent``. The monitor calls back on a Core Audio
system thread; emitting a signal from there queues it to the receivers on the GUI thread.
Without Core Audio (``EndpointError``) the watch is inert: ``available`` is false and every
query returns ``None``, so the app works as before (no device-change warning).
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, Signal

from desktop_audio_recorder.audio.endpoints import (
    Endpoint,
    EndpointError,
    EndpointEvent,
    EndpointMonitor,
)

log = logging.getLogger(__name__)

MonitorFactory = Callable[[Callable[[EndpointEvent], None]], EndpointMonitor]


class DeviceWatch(QObject):
    changed = Signal(object)  # EndpointEvent, delivered on the GUI thread

    def __init__(
        self, parent: QObject | None = None, monitor_factory: MonitorFactory = EndpointMonitor
    ) -> None:
        super().__init__(parent)
        self._closed = False
        self._monitor = None
        try:
            monitor = monitor_factory(self._on_event)
            monitor.start()
            self._monitor = monitor
        except (EndpointError, OSError) as exc:
            log.warning("Device changes can't be watched: %s", exc)
        except Exception:
            log.warning("Device changes can't be watched", exc_info=True)

    @property
    def available(self) -> bool:
        return self._monitor is not None

    def _on_event(self, event: EndpointEvent) -> None:  # Core Audio thread
        if self._closed:
            return
        with contextlib.suppress(RuntimeError):  # deleted while the callback ran (shutdown)
            self.changed.emit(event)

    def _query(self, name: str, *args) -> Endpoint | None:
        monitor = self._monitor
        if monitor is None:
            return None
        try:
            return getattr(monitor, name)(*args)
        except Exception:
            log.warning("Endpoint query %s failed", name, exc_info=True)
            return None

    def default_render(self) -> Endpoint | None:
        return self._query("default_render")

    def endpoint(self, device_id: str) -> Endpoint | None:
        return self._query("endpoint", device_id)

    def find_render(self, name: str) -> Endpoint | None:
        return self._query("find_render", name)

    def close(self) -> None:
        self._closed = True
        monitor, self._monitor = self._monitor, None
        if monitor is not None:
            try:
                monitor.close()
            except Exception:
                log.warning("Could not close the endpoint monitor", exc_info=True)
