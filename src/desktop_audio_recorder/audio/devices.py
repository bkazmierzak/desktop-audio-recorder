"""Resolve the Windows default playback device to its WASAPI loopback capture device.

PortAudio snapshots the device list when ``PyAudio()`` is created, so every public call here
creates a fresh instance and terminates it afterwards. That way a change of the Windows default
output (e.g. speakers -> headphones) is picked up on the next call.

No Qt here: this module is used by the headless recorder and wrapped by ``ui/`` later.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import pyaudiowpatch

LOOPBACK_SUFFIX = " [Loopback]"


class DeviceError(RuntimeError):
    """No usable WASAPI output / loopback device could be found."""


@dataclass(frozen=True, slots=True)
class LoopbackDevice:
    """The loopback capture endpoint paired with a WASAPI output device."""

    index: int  # PortAudio index of the loopback (input) device
    name: str  # e.g. "Speakers (Realtek(R) Audio) [Loopback]"
    sample_rate: int  # native mix rate, e.g. 48000
    channels: int  # native mix channel count (may be 1, 2, 6, 8, ...)
    output_index: int  # PortAudio index of the paired output device (for the keep-alive stream)
    output_name: str  # e.g. "Speakers (Realtek(R) Audio)"
    output_channels: int

    @property
    def display_name(self) -> str:
        return self.output_name


@dataclass(frozen=True, slots=True)
class OutputDevice:
    """A WASAPI playback device."""

    index: int
    name: str
    channels: int
    sample_rate: int
    is_default: bool
    loopback_index: int | None  # None when no loopback endpoint was found for it


def _wasapi_info(pa: Any, module: ModuleType) -> dict:
    try:
        return pa.get_host_api_info_by_type(module.paWASAPI)
    except OSError as exc:
        raise DeviceError("The Windows WASAPI audio API is not available.") from exc


def _wasapi_devices(pa: Any, host_api_index: int) -> Iterator[dict]:
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["hostApi"] == host_api_index:
            yield info


def _is_loopback(info: dict) -> bool:
    return bool(info.get("isLoopbackDevice", False))


def _find_loopback_for(output: dict, candidates: list[dict]) -> dict | None:
    """Find the loopback device for ``output``.

    PyAudioWPatch names loopbacks ``"<output name> [Loopback]"``. Prefer the exact name and fall
    back to a substring match (the same rule PyAudioWPatch itself uses).
    """
    exact = output["name"] + LOOPBACK_SUFFIX
    for info in candidates:
        if info["name"] == exact:
            return info
    for info in candidates:
        if output["name"] in info["name"]:
            return info
    return None


def resolve_default_loopback(pa: Any, module: ModuleType = pyaudiowpatch) -> LoopbackDevice:
    """Resolve the default WASAPI output device to its loopback device using ``pa``."""
    wasapi = _wasapi_info(pa, module)
    default_index = wasapi.get("defaultOutputDevice", -1)
    if default_index is None or default_index < 0:
        raise DeviceError("No default playback device is set in Windows.")
    return _resolve_loopback(pa, wasapi, default_index)


def resolve_loopback(
    pa: Any, output_index: int, module: ModuleType = pyaudiowpatch
) -> LoopbackDevice:
    """Resolve a specific WASAPI output device (by PortAudio index) to its loopback device."""
    return _resolve_loopback(pa, _wasapi_info(pa, module), output_index)


def resolve_named_loopback(
    pa: Any, output_name: str, module: ModuleType = pyaudiowpatch
) -> LoopbackDevice:
    """Resolve the WASAPI output device called ``output_name`` (indexes change whenever
    PortAudio re-reads the device list; names don't)."""
    wasapi = _wasapi_info(pa, module)
    for info in _wasapi_devices(pa, wasapi["index"]):
        if (
            info["name"] == output_name
            and not _is_loopback(info)
            and int(info["maxOutputChannels"]) > 0
        ):
            return _resolve_loopback(pa, wasapi, int(info["index"]))
    raise DeviceError(f"No playback device called {output_name!r}.")


def _resolve_loopback(pa: Any, wasapi: dict, output_index: int) -> LoopbackDevice:
    output = pa.get_device_info_by_index(output_index)
    if output["hostApi"] != wasapi["index"]:
        raise DeviceError(f"{output['name']!r} is not a WASAPI device.")
    devices = list(_wasapi_devices(pa, wasapi["index"]))
    if _is_loopback(output):
        loopback = output
    else:
        loopback = _find_loopback_for(output, [d for d in devices if _is_loopback(d)])
    if loopback is None:
        raise DeviceError(f"No loopback capture device found for {output['name']!r}.")
    if int(loopback["maxInputChannels"]) < 1:
        raise DeviceError(f"Loopback device {loopback['name']!r} reports no input channels.")

    output_name = output["name"]
    if output_name.endswith(LOOPBACK_SUFFIX):
        output_name = output_name[: -len(LOOPBACK_SUFFIX)]
    return LoopbackDevice(
        index=int(loopback["index"]),
        name=loopback["name"],
        sample_rate=int(round(loopback["defaultSampleRate"])),
        channels=int(loopback["maxInputChannels"]),
        output_index=int(output["index"]),
        output_name=output_name,
        output_channels=int(output.get("maxOutputChannels", 0))
        or int(loopback["maxInputChannels"]),
    )


def list_output_devices(pa: Any, module: ModuleType = pyaudiowpatch) -> list[OutputDevice]:
    """List WASAPI playback devices (not their loopback twins) using ``pa``."""
    wasapi = _wasapi_info(pa, module)
    default_index = wasapi.get("defaultOutputDevice", -1)
    devices = list(_wasapi_devices(pa, wasapi["index"]))
    loopbacks = [d for d in devices if _is_loopback(d)]
    result = []
    for info in devices:
        if _is_loopback(info) or int(info["maxOutputChannels"]) < 1:
            continue
        loopback = _find_loopback_for(info, loopbacks)
        result.append(
            OutputDevice(
                index=int(info["index"]),
                name=info["name"],
                channels=int(info["maxOutputChannels"]),
                sample_rate=int(round(info["defaultSampleRate"])),
                is_default=int(info["index"]) == default_index,
                loopback_index=None if loopback is None else int(loopback["index"]),
            )
        )
    return result


class DeviceManager:
    """Device queries that always see the current Windows device list.

    ``module`` is the PyAudio module (``pyaudiowpatch``); tests inject a fake one.
    """

    def __init__(self, module: ModuleType = pyaudiowpatch) -> None:
        self.module = module

    def create_pyaudio(self) -> Any:
        """Return a fresh ``PyAudio`` instance. The caller must ``terminate()`` it."""
        return self.module.PyAudio()

    @contextlib.contextmanager
    def session(self) -> Iterator[Any]:
        pa = self.create_pyaudio()
        try:
            yield pa
        finally:
            pa.terminate()

    def resolve_default_loopback(self, pa: Any | None = None) -> LoopbackDevice:
        """Resolve the current default output's loopback device.

        If ``pa`` is given it is used (and not terminated); otherwise a fresh instance is created.
        Raises ``DeviceError`` if there is no usable device.
        """
        if pa is not None:
            return resolve_default_loopback(pa, self.module)
        with self.session() as fresh:
            return resolve_default_loopback(fresh, self.module)

    def resolve_named_loopback(self, pa: Any, output_name: str) -> LoopbackDevice:
        """The loopback of the playback device called ``output_name``, using ``pa``."""
        return resolve_named_loopback(pa, output_name, self.module)

    def list_output_devices(self) -> list[OutputDevice]:
        with self.session() as pa:
            return list_output_devices(pa, self.module)
