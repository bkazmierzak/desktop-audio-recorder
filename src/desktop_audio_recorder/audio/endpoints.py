"""Windows audio endpoints through the Core Audio ``IMMDeviceEnumerator`` (comtypes).

PortAudio snapshots the device list when it initialises, and while the recorder's PyAudio
instance is alive a new one doesn't re-enumerate. So the recorder can't notice through PortAudio
that Windows switched the default playback device, or that the device it records went away.
Core Audio can: ``IMMNotificationClient`` is called (on a system thread) when the default
endpoint changes and when an endpoint is added, removed, unplugged or disabled.

``EndpointMonitor`` wraps that, Qt-free: ``on_event(EndpointEvent)`` is called on the system's
notification thread, so the UI must marshal it (``ui/device_watch.py`` does, via a queued Qt
signal). The callback must return quickly and must not call back into the enumerator's
(un)register methods.

Roles: PortAudio's WASAPI host API (and so pyaudiowpatch and sounddevice) uses the
**multimedia** default (``eRender``, ``eMultimedia``); Windows' Sound settings change the
console and multimedia defaults together. Only multimedia changes are reported.

Endpoint IDs vs PortAudio: PortAudio doesn't expose the endpoint ID, but a WASAPI device's name
is the endpoint's friendly name (``PKEY_Device_FriendlyName``), e.g.
"Speakers (INZONE H9 II - Game)", so ``find_render(name)`` maps one to the other.
"""

from __future__ import annotations

import ctypes
import enum
import logging
from collections.abc import Callable
from ctypes import POINTER, Structure, c_int, c_ushort, c_void_p
from ctypes.wintypes import DWORD, LPCWSTR
from dataclasses import dataclass

log = logging.getLogger(__name__)

# EDataFlow / ERole / DEVICE_STATE_*
E_RENDER = 0
E_CAPTURE = 1
E_CONSOLE = 0
E_MULTIMEDIA = 1
E_COMMUNICATIONS = 2
DEVICE_STATE_ACTIVE = 0x1
DEVICE_STATE_DISABLED = 0x2
DEVICE_STATE_NOTPRESENT = 0x4
DEVICE_STATE_UNPLUGGED = 0x8
DEVICE_STATEMASK_ALL = 0xF
_STGM_READ = 0
_VT_LPWSTR = 31
_E_NOTFOUND = -2147023728  # HRESULT_FROM_WIN32(ERROR_NOT_FOUND) = 0x80070490


def describe_state(state: int) -> str:
    return {
        DEVICE_STATE_ACTIVE: "active",
        DEVICE_STATE_DISABLED: "disabled",
        DEVICE_STATE_NOTPRESENT: "not present",
        DEVICE_STATE_UNPLUGGED: "unplugged",
    }.get(state, f"state {state:#x}")


@dataclass(frozen=True, slots=True)
class Endpoint:
    id: str
    name: str  # friendly name, the same as PortAudio's WASAPI device name
    state: int

    @property
    def active(self) -> bool:
        return self.state == DEVICE_STATE_ACTIVE


class EventKind(enum.Enum):
    DEFAULT_CHANGED = "default changed"  # device_id: the new default ("" if there is none)
    STATE_CHANGED = "state changed"  # device_id, state
    REMOVED = "removed"
    ADDED = "added"


@dataclass(frozen=True, slots=True)
class EndpointEvent:
    kind: EventKind
    device_id: str
    state: int | None = None


EventCallback = Callable[[EndpointEvent], None]


class EndpointError(RuntimeError):
    """Core Audio isn't available (not Windows, or the audio service isn't running)."""


# --------------------------------------------------------------------------- COM declarations


def _declare():
    """The COM interfaces, declared on first use (comtypes initialises COM when imported)."""
    import comtypes
    from comtypes import COMMETHOD, GUID, HRESULT, IUnknown

    class PROPERTYKEY(Structure):
        _fields_ = [("fmtid", GUID), ("pid", DWORD)]

    class PROPVARIANT(Structure):
        # vt + 3 reserved words, then a 16-byte union; only the LPWSTR member is read here.
        _fields_ = [
            ("vt", c_ushort),
            ("reserved1", c_ushort),
            ("reserved2", c_ushort),
            ("reserved3", c_ushort),
            ("pwszVal", c_void_p),
            ("pad", c_void_p),
        ]

    class IPropertyStore(IUnknown):
        _iid_ = GUID("{886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99}")
        _methods_ = [
            COMMETHOD([], HRESULT, "GetCount", (["out"], POINTER(DWORD), "count")),
            COMMETHOD(
                [], HRESULT, "GetAt", (["in"], DWORD, "i"), (["out"], POINTER(PROPERTYKEY), "key")
            ),
            COMMETHOD(
                [],
                HRESULT,
                "GetValue",
                (["in"], POINTER(PROPERTYKEY), "key"),
                (["out"], POINTER(PROPVARIANT), "value"),
            ),
        ]

    class IMMDevice(IUnknown):
        _iid_ = GUID("{D666063F-1587-4E43-81F1-B948E807363F}")
        _methods_ = [
            COMMETHOD(
                [],
                HRESULT,
                "Activate",
                (["in"], POINTER(GUID), "iid"),
                (["in"], DWORD, "context"),
                (["in"], c_void_p, "params"),
                (["out"], POINTER(c_void_p), "interface"),
            ),
            COMMETHOD(
                [],
                HRESULT,
                "OpenPropertyStore",
                (["in"], DWORD, "access"),
                (["out"], POINTER(POINTER(IPropertyStore)), "store"),
            ),
            COMMETHOD([], HRESULT, "GetId", (["out"], POINTER(c_void_p), "id")),
            COMMETHOD([], HRESULT, "GetState", (["out"], POINTER(DWORD), "state")),
        ]

    class IMMDeviceCollection(IUnknown):
        _iid_ = GUID("{0BD7A1BE-7A1A-44DB-8397-CC5392387B5E}")
        _methods_ = [
            COMMETHOD([], HRESULT, "GetCount", (["out"], POINTER(DWORD), "count")),
            COMMETHOD(
                [],
                HRESULT,
                "Item",
                (["in"], DWORD, "i"),
                (["out"], POINTER(POINTER(IMMDevice)), "device"),
            ),
        ]

    class IMMNotificationClient(IUnknown):
        _iid_ = GUID("{7991EEC9-7E89-4D85-8390-6C703CEC60C0}")
        _methods_ = [
            COMMETHOD(
                [],
                HRESULT,
                "OnDeviceStateChanged",
                (["in"], LPCWSTR, "device_id"),
                (["in"], DWORD, "state"),
            ),
            COMMETHOD([], HRESULT, "OnDeviceAdded", (["in"], LPCWSTR, "device_id")),
            COMMETHOD([], HRESULT, "OnDeviceRemoved", (["in"], LPCWSTR, "device_id")),
            COMMETHOD(
                [],
                HRESULT,
                "OnDefaultDeviceChanged",
                (["in"], c_int, "flow"),
                (["in"], c_int, "role"),
                (["in"], LPCWSTR, "device_id"),
            ),
            COMMETHOD(
                [],
                HRESULT,
                "OnPropertyValueChanged",
                (["in"], LPCWSTR, "device_id"),
                (["in"], PROPERTYKEY, "key"),
            ),
        ]

    class IMMDeviceEnumerator(IUnknown):
        _iid_ = GUID("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
        _methods_ = [
            COMMETHOD(
                [],
                HRESULT,
                "EnumAudioEndpoints",
                (["in"], c_int, "flow"),
                (["in"], DWORD, "state_mask"),
                (["out"], POINTER(POINTER(IMMDeviceCollection)), "devices"),
            ),
            COMMETHOD(
                [],
                HRESULT,
                "GetDefaultAudioEndpoint",
                (["in"], c_int, "flow"),
                (["in"], c_int, "role"),
                (["out"], POINTER(POINTER(IMMDevice)), "device"),
            ),
            COMMETHOD(
                [],
                HRESULT,
                "GetDevice",
                (["in"], LPCWSTR, "device_id"),
                (["out"], POINTER(POINTER(IMMDevice)), "device"),
            ),
            COMMETHOD(
                [],
                HRESULT,
                "RegisterEndpointNotificationCallback",
                (["in"], POINTER(IMMNotificationClient), "client"),
            ),
            COMMETHOD(
                [],
                HRESULT,
                "UnregisterEndpointNotificationCallback",
                (["in"], POINTER(IMMNotificationClient), "client"),
            ),
        ]

    class NotificationClient(comtypes.COMObject):
        _com_interfaces_ = [IMMNotificationClient]

        def __init__(self, sink: Callable[[EndpointEvent], None]) -> None:
            super().__init__()
            self._sink = sink

        # comtypes passes ``this`` first for these "low-level" implementations.
        def OnDeviceStateChanged(self, this, device_id, state):  # noqa: N802
            self._sink(EndpointEvent(EventKind.STATE_CHANGED, device_id or "", int(state)))
            return 0

        def OnDeviceAdded(self, this, device_id):  # noqa: N802
            self._sink(EndpointEvent(EventKind.ADDED, device_id or ""))
            return 0

        def OnDeviceRemoved(self, this, device_id):  # noqa: N802
            self._sink(EndpointEvent(EventKind.REMOVED, device_id or ""))
            return 0

        def OnDefaultDeviceChanged(self, this, flow, role, device_id):  # noqa: N802
            if flow == E_RENDER and role == E_MULTIMEDIA:
                self._sink(EndpointEvent(EventKind.DEFAULT_CHANGED, device_id or ""))
            return 0

        def OnPropertyValueChanged(self, this, device_id, key):  # noqa: N802
            return 0

    friendly_name = PROPERTYKEY()
    friendly_name.fmtid = GUID("{A45C254E-DF1C-4EFD-8020-67D146A850E0}")
    friendly_name.pid = 14
    return {
        "comtypes": comtypes,
        "GUID": GUID,
        "PROPVARIANT": PROPVARIANT,
        "IMMDeviceEnumerator": IMMDeviceEnumerator,
        "NotificationClient": NotificationClient,
        "PKEY_FRIENDLY_NAME": friendly_name,
        "CLSID": GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}"),
    }


_com: dict | None = None


def _com_api() -> dict:
    global _com
    if _com is None:
        try:
            _com = _declare()
        except (ImportError, OSError) as exc:
            raise EndpointError(f"Core Audio is not available: {exc}") from exc
    return _com


# --------------------------------------------------------------------------- monitor


class EndpointMonitor:
    """Queries render endpoints and (after ``start()``) reports endpoint changes.

    Create and use it on the GUI thread (COM is initialised there); ``on_event`` is called on a
    system thread.
    """

    def __init__(self, on_event: EventCallback | None = None) -> None:
        api = _com_api()
        self._api = api
        self._on_event = on_event
        self._client = None
        self._client_ptr = None
        try:
            self._enumerator = api["comtypes"].CoCreateInstance(
                api["CLSID"], interface=api["IMMDeviceEnumerator"]
            )
        except (OSError, api["comtypes"].COMError) as exc:
            raise EndpointError(f"Core Audio is not available: {exc}") from exc

    # ------------------------------------------------------------ notifications

    @property
    def watching(self) -> bool:
        return self._client is not None

    def start(self) -> None:
        if self._client is not None or self._on_event is None:
            return
        client = self._api["NotificationClient"](self._dispatch)
        pointer = client.QueryInterface(self._api["NotificationClient"]._com_interfaces_[0])
        self._enumerator.RegisterEndpointNotificationCallback(pointer)
        self._client, self._client_ptr = client, pointer  # keep both alive while registered
        log.info("Watching audio endpoint changes")

    def stop(self) -> None:
        if self._client_ptr is None:
            return
        try:
            self._enumerator.UnregisterEndpointNotificationCallback(self._client_ptr)
        except Exception:
            log.warning("Could not unregister the endpoint callback", exc_info=True)
        self._client = self._client_ptr = None

    def _dispatch(self, event: EndpointEvent) -> None:  # system thread
        try:
            log.debug("Endpoint event %s", event)
            callback = self._on_event
            if callback is not None:
                callback(event)
        except Exception:  # never raise into COM
            log.exception("Endpoint event handler failed")

    # ------------------------------------------------------------ queries

    def default_render(self) -> Endpoint | None:
        """The multimedia default playback endpoint (``None`` if there is none)."""
        comtypes = self._api["comtypes"]
        try:
            device = self._enumerator.GetDefaultAudioEndpoint(E_RENDER, E_MULTIMEDIA)
        except comtypes.COMError as exc:
            if exc.hresult == _E_NOTFOUND:
                return None
            raise EndpointError(f"Could not get the default playback device: {exc}") from exc
        return self._describe(device)

    def endpoint(self, device_id: str) -> Endpoint | None:
        """The endpoint with this ID (``None`` if Windows doesn't know it any more)."""
        comtypes = self._api["comtypes"]
        try:
            device = self._enumerator.GetDevice(device_id)
            return self._describe(device)
        except comtypes.COMError:
            return None

    def render_endpoints(self, state_mask: int = DEVICE_STATE_ACTIVE) -> list[Endpoint]:
        collection = self._enumerator.EnumAudioEndpoints(E_RENDER, state_mask)
        return [self._describe(collection.Item(i)) for i in range(collection.GetCount())]

    def find_render(self, name: str) -> Endpoint | None:
        """The active playback endpoint called ``name`` (PortAudio's WASAPI device name);
        the default one wins if several share the name."""
        default = self.default_render()
        if default is not None and default.name == name:
            return default
        for endpoint in self.render_endpoints():
            if endpoint.name == name:
                return endpoint
        return None

    def _describe(self, device) -> Endpoint:
        ole32 = ctypes.windll.ole32
        raw_id = device.GetId()
        try:
            device_id = ctypes.wstring_at(raw_id)
        finally:
            ole32.CoTaskMemFree(c_void_p(raw_id))
        state = int(device.GetState())
        return Endpoint(device_id, self._friendly_name(device), state)

    def _friendly_name(self, device) -> str:
        api = self._api
        try:
            store = device.OpenPropertyStore(_STGM_READ)
            value = store.GetValue(ctypes.byref(api["PKEY_FRIENDLY_NAME"]))
        except Exception:
            log.debug("No friendly name", exc_info=True)
            return ""
        try:
            if value.vt == _VT_LPWSTR and value.pwszVal:
                return ctypes.wstring_at(value.pwszVal)
            return ""
        finally:
            ctypes.windll.ole32.PropVariantClear(ctypes.byref(value))

    def close(self) -> None:
        self.stop()
        self._enumerator = None
