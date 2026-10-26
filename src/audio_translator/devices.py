"""Audio device lookup (thin wrapper over sounddevice)."""

from __future__ import annotations

from dataclasses import dataclass


class DeviceError(RuntimeError):
    """A device is missing, unusable, or disappeared mid-session."""


@dataclass(frozen=True)
class DeviceInfo:
    index: int | None
    name: str
    samplerate: float
    max_channels: int


def _sd():
    import sounddevice as sd

    return sd


def list_devices(kind: str) -> list[DeviceInfo]:
    """Return devices with at least one channel of ``kind`` ("input"/"output")."""
    sd = _sd()
    key = f"max_{kind}_channels"
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d[key] > 0:
            out.append(DeviceInfo(i, d["name"], float(d["default_samplerate"]), int(d[key])))
    return out


def resolve_device(name: str | None, kind: str) -> DeviceInfo:
    """Find a device by case-insensitive substring; ``None`` = system default.

    The default is only allowed for input; output must be named explicitly so
    translated speech is never routed to the user's speakers by accident.
    """
    sd = _sd()
    key = f"max_{kind}_channels"
    if name is None:
        if kind != "input":
            raise DeviceError("An output device must be chosen explicitly (e.g. 'BlackHole 2ch').")
        try:
            d = sd.query_devices(kind="input")
        except Exception as exc:  # no default input
            raise DeviceError(f"No default input device: {exc}") from exc
        return DeviceInfo(None, d["name"], float(d["default_samplerate"]), int(d[key]))
    needle = name.lower()
    for dev in list_devices(kind):
        if dev.name.lower() == needle:
            return dev
    for dev in list_devices(kind):
        if needle in dev.name.lower():
            return dev
    hint = " (install with: brew install blackhole-2ch)" if "blackhole" in needle else ""
    raise DeviceError(f"{kind.capitalize()} device '{name}' not found{hint}")
