"""USB device discovery + persistent registry.

Two-layer identity: USB serial number (preferred when available) falls back
to USB port path (KERNELS, e.g. "3-1" or "3-1.2"). CH340-based DevKits don't
expose a serial, so port path is the only stable handle for them — provided
the user keeps the board in the same physical RPi USB socket.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, model_validator

from .config import get_settings

# Known USB-serial bridge / native-USB vendors found on ESP32 DevKits.
ESP_VENDOR_IDS: frozenset[int] = frozenset(
    {
        0x10C4,  # Silicon Labs CP210x family
        0x1A86,  # QinHeng Electronics — CH340/CH341/CH9102
        0x303A,  # Espressif (native USB on ESP32-S2/S3/C3)
        0x0403,  # FTDI — older boards
    }
)

REGISTRY_FILENAME = "devices.json"


# ── Models ───────────────────────────────────────────────────────────────


class DiscoveredPort(BaseModel):
    device: str
    vid: int | None = None
    pid: int | None = None
    serial_number: str | None = None
    manufacturer: str | None = None
    product: str | None = None
    description: str = ""
    usb_port: str | None = None
    id_path: str | None = None
    chip: str | None = None  # Inferred (CH340 / CP2102 / ESP32-USB / FTDI / None)


class DeviceRegistration(BaseModel):
    alias: str
    serial_number: str | None = None
    usb_port: str | None = None
    vid: int | None = None
    pid: int | None = None
    chip: str | None = None
    udev_symlink: str | None = None  # filled by Task 3
    created_at: datetime
    notes: str | None = None

    @model_validator(mode="after")
    def _identity_required(self) -> "DeviceRegistration":
        if not self.serial_number and not self.usb_port:
            raise ValueError("registration requires serial_number or usb_port")
        return self


class DeviceStatus(BaseModel):
    alias: str
    identity_type: Literal["serial", "usb_port"]
    identity_value: str
    udev_symlink: str | None = None
    current_device: str | None = None
    connected: bool = False
    vid: int | None = None
    pid: int | None = None
    chip: str | None = None


class DiscoveredPortWithRegistration(BaseModel):
    port: DiscoveredPort
    matched_alias: str | None = None


# ── Chip inference ───────────────────────────────────────────────────────


def guess_chip(
    vid: int | None,
    pid: int | None,
    manufacturer: str | None = None,
    product: str | None = None,
) -> str | None:
    if vid == 0x1A86:
        if pid == 0x7523:
            return "CH340"
        if pid == 0x55D4:
            return "CH9102"
        return "CH34x"
    if vid == 0x10C4:
        if pid == 0xEA60:
            return "CP2102"
        if pid == 0xEA70:
            return "CP2105"
        return "CP210x"
    if vid == 0x303A:
        return "ESP32-USB"
    if vid == 0x0403:
        return "FTDI"
    blob = f"{manufacturer or ''} {product or ''}".lower()
    for needle, label in [
        ("ch340", "CH340"),
        ("ch9102", "CH9102"),
        ("cp2104", "CP2104"),
        ("cp2102", "CP2102"),
        ("cp210", "CP210x"),
        ("esp32", "ESP32-USB"),
        ("ftdi", "FTDI"),
    ]:
        if needle in blob:
            return label
    return None


# ── pyudev discovery ─────────────────────────────────────────────────────


def _attr_str(udev_dev, key: str) -> str | None:  # type: ignore[no-untyped-def]
    """Read a sysfs attribute as a stripped str (None on absent/empty)."""
    val = udev_dev.attributes.get(key)
    if val is None:
        return None
    if isinstance(val, bytes):
        try:
            decoded = val.decode("utf-8", errors="replace").strip()
        except Exception:  # pragma: no cover
            return None
        return decoded or None
    s = str(val).strip()
    return s or None


def _hex_attr(udev_dev, key: str) -> int | None:  # type: ignore[no-untyped-def]
    raw = _attr_str(udev_dev, key)
    if raw is None:
        return None
    try:
        return int(raw, 16)
    except ValueError:
        return None


def _build_discovered_port(tty_dev) -> DiscoveredPort | None:  # type: ignore[no-untyped-def]
    node = tty_dev.device_node
    if not node:
        return None
    if not (node.startswith("/dev/ttyUSB") or node.startswith("/dev/ttyACM")):
        return None
    usb_dev = tty_dev.find_parent("usb", "usb_device")
    if usb_dev is None:
        return None
    vid = _hex_attr(usb_dev, "idVendor")
    pid = _hex_attr(usb_dev, "idProduct")
    serial_number = _attr_str(usb_dev, "serial")
    manufacturer = _attr_str(usb_dev, "manufacturer")
    product = _attr_str(usb_dev, "product")
    usb_port = getattr(usb_dev, "sys_name", None)
    id_path = None
    if hasattr(tty_dev, "properties"):
        try:
            id_path = tty_dev.properties.get("ID_PATH")
        except Exception:  # pragma: no cover
            id_path = None
    description = " ".join(filter(None, [manufacturer, product])).strip() or node
    chip = guess_chip(vid, pid, manufacturer, product)
    return DiscoveredPort(
        device=node,
        vid=vid,
        pid=pid,
        serial_number=serial_number,
        manufacturer=manufacturer,
        product=product,
        description=description,
        usb_port=usb_port,
        id_path=id_path,
        chip=chip,
    )


def discover_ports(*, esp_only: bool = True, context=None) -> list[DiscoveredPort]:  # type: ignore[no-untyped-def]
    """Enumerate currently connected serial ports.

    `context` is exposed for tests — production callers leave it None and
    a fresh ``pyudev.Context()`` is created.
    """
    if context is None:
        import pyudev

        context = pyudev.Context()
    ports: list[DiscoveredPort] = []
    for tty_dev in context.list_devices(subsystem="tty"):
        port = _build_discovered_port(tty_dev)
        if port is None:
            continue
        if esp_only and port.vid not in ESP_VENDOR_IDS:
            continue
        ports.append(port)
    ports.sort(key=lambda p: p.device)
    return ports


# ── Identity matching ────────────────────────────────────────────────────


def match_registration(
    port: DiscoveredPort, registry: dict[str, DeviceRegistration]
) -> DeviceRegistration | None:
    """Resolve a live port to a registered alias.

    Priority:
      1. Serial-number match (most stable; survives port changes).
      2. USB-port-path match — but only against registrations that have no
         serial. A registration that *does* carry a serial is intentionally
         pinned to that serial; refusing the port-path fallback prevents a
         CH340 board from being mis-identified as a CP2102 board that
         happens to be plugged into the same socket.
    """
    if port.serial_number:
        for reg in registry.values():
            if reg.serial_number and reg.serial_number == port.serial_number:
                return reg
    if port.usb_port:
        for reg in registry.values():
            if reg.serial_number is None and reg.usb_port == port.usb_port:
                return reg
    return None


# ── Registry persistence ─────────────────────────────────────────────────


def _registry_path() -> Path:
    return get_settings().data_dir / REGISTRY_FILENAME


def load_registry() -> dict[str, DeviceRegistration]:
    path = _registry_path()
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {alias: DeviceRegistration.model_validate(data) for alias, data in raw.items()}


def save_registry(registry: dict[str, DeviceRegistration]) -> None:
    """Atomic JSON write — temp file in same dir, then os.replace()."""
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {alias: reg.model_dump(mode="json") for alias, reg in registry.items()}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


# ── Public CRUD ──────────────────────────────────────────────────────────


class DeviceRegistryError(ValueError):
    """Raised on alias/identity collisions or missing identity."""


def _validate_unique(
    registry: dict[str, DeviceRegistration],
    *,
    alias: str,
    serial_number: str | None,
    usb_port: str | None,
) -> None:
    if alias in registry:
        raise DeviceRegistryError(f"alias {alias!r} already registered")
    if not serial_number and not usb_port:
        raise DeviceRegistryError("registration requires serial_number or usb_port")
    for existing in registry.values():
        if serial_number and existing.serial_number == serial_number:
            raise DeviceRegistryError(
                f"serial_number {serial_number!r} already registered to alias "
                f"{existing.alias!r}"
            )
        if (
            usb_port
            and not serial_number
            and existing.serial_number is None
            and existing.usb_port == usb_port
        ):
            raise DeviceRegistryError(
                f"usb_port {usb_port!r} already registered to alias {existing.alias!r}"
            )


def register_device(
    alias: str,
    *,
    port: DiscoveredPort | None = None,
    serial_number: str | None = None,
    usb_port: str | None = None,
    vid: int | None = None,
    pid: int | None = None,
    chip: str | None = None,
    notes: str | None = None,
) -> DeviceRegistration:
    """Persist a new alias. Pass either a discovered port or explicit fields.

    When both a port and explicit fields are provided, explicit values win.
    Identity precedence (when both serial and usb_port are present): the
    serial is used as primary identifier; the usb_port is recorded as
    metadata for diagnostics but matching will hit serial first.
    """
    registry = load_registry()

    if port is not None:
        serial_number = serial_number if serial_number is not None else port.serial_number
        usb_port = usb_port if usb_port is not None else port.usb_port
        vid = vid if vid is not None else port.vid
        pid = pid if pid is not None else port.pid
        if chip is None:
            chip = port.chip or guess_chip(vid, pid, port.manufacturer, port.product)
    if chip is None:
        chip = guess_chip(vid, pid)

    _validate_unique(registry, alias=alias, serial_number=serial_number, usb_port=usb_port)

    reg = DeviceRegistration(
        alias=alias,
        serial_number=serial_number,
        usb_port=usb_port,
        vid=vid,
        pid=pid,
        chip=chip,
        created_at=datetime.now(timezone.utc),
        notes=notes,
    )
    registry[alias] = reg
    save_registry(registry)
    return reg


def unregister_device(alias: str) -> None:
    registry = load_registry()
    if alias not in registry:
        raise KeyError(alias)
    del registry[alias]
    save_registry(registry)


def get_device_status(
    *,
    ports: Iterable[DiscoveredPort] | None = None,
) -> list[DeviceStatus]:
    """Merge registry + live discovery into a status list."""
    registry = load_registry()
    if ports is None:
        ports = discover_ports(esp_only=False)
    matches: dict[str, DiscoveredPort] = {}
    for port in ports:
        reg = match_registration(port, registry)
        if reg is not None:
            matches[reg.alias] = port

    statuses: list[DeviceStatus] = []
    for alias, reg in registry.items():
        port = matches.get(alias)
        if reg.serial_number:
            identity_type: Literal["serial", "usb_port"] = "serial"
            identity_value = reg.serial_number
        else:
            identity_type = "usb_port"
            assert reg.usb_port is not None  # validator guarantees one or the other
            identity_value = reg.usb_port
        statuses.append(
            DeviceStatus(
                alias=alias,
                identity_type=identity_type,
                identity_value=identity_value,
                udev_symlink=reg.udev_symlink,
                current_device=port.device if port else None,
                connected=port is not None,
                vid=reg.vid,
                pid=reg.pid,
                chip=reg.chip,
            )
        )
    statuses.sort(key=lambda s: s.alias)
    return statuses


def discover_with_registrations(
    *, esp_only: bool = True
) -> list[DiscoveredPortWithRegistration]:
    """Discovery + per-port registration lookup, used by /api/devices/discover."""
    registry = load_registry()
    ports = discover_ports(esp_only=esp_only)
    out: list[DiscoveredPortWithRegistration] = []
    for port in ports:
        reg = match_registration(port, registry)
        out.append(
            DiscoveredPortWithRegistration(
                port=port,
                matched_alias=reg.alias if reg else None,
            )
        )
    return out


__all__ = [
    "DeviceRegistration",
    "DeviceRegistryError",
    "DeviceStatus",
    "DiscoveredPort",
    "DiscoveredPortWithRegistration",
    "ESP_VENDOR_IDS",
    "discover_ports",
    "discover_with_registrations",
    "get_device_status",
    "guess_chip",
    "load_registry",
    "match_registration",
    "register_device",
    "save_registry",
    "unregister_device",
]
