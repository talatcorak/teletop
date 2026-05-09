"""USB device discovery + persistent registry.

Two-layer identity: USB serial number (preferred when available) falls back
to USB port path (KERNELS, e.g. "3-1" or "3-1.2"). CH340-based DevKits don't
expose a serial, so port path is the only stable handle for them — provided
the user keeps the board in the same physical RPi USB socket.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, get_args

from pydantic import BaseModel, model_validator

from .config import get_settings

logger = logging.getLogger("teletop.devices")

# Known USB-serial bridge / native-USB vendors found on ESP32 DevKits.
ESP_VENDOR_IDS: frozenset[int] = frozenset(
    {
        0x10C4,  # Silicon Labs CP210x family
        0x1A86,  # QinHeng Electronics — CH340/CH341/CH9102
        0x303A,  # Espressif (native USB on ESP32-S2/S3/C3)
        0x0403,  # FTDI — older boards
    }
)

# Target ESP chip families. The USB-UART converter is a separate concern
# (see `usb_chip` field) — the target dictates flash binaries and memory
# layout, not how bytes get to the chip.
TargetChip = Literal[
    "esp32",
    "esp8266",
    "esp32s2",
    "esp32s3",
    "esp32c3",
    "esp32c6",
    "esp32h2",
]
TARGET_CHIPS: tuple[str, ...] = tuple(get_args(TargetChip))
DEFAULT_TARGET_CHIP: TargetChip = "esp32"

REGISTRY_FILENAME = "devices.json"

# udev rule destination + symlink directory. Both can be overridden via env
# vars (used in tests; production never sets them).
UDEV_RULES_PATH = Path(
    os.environ.get("TELETOP_UDEV_RULES_PATH", "/etc/udev/rules.d/99-teletop.rules")
)
SYMLINK_DIR = Path(os.environ.get("TELETOP_SYMLINK_DIR", "/dev"))
# `tty-` is ESP-family-agnostic (ESP8266 boards also live here) and reads
# naturally with shell tab-completion against /dev/tty*. Devices registered
# before 0.2 used "esp32-" — udev re-trigger drops the old links.
SYMLINK_PREFIX = "tty-"

# Aliases double as udev SYMLINK names (`/dev/tty-<alias>`); keep them safe.
ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


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
    usb_chip: str | None = None  # CH340 / CP2102 / ESP32-USB / FTDI / None


class DeviceRegistration(BaseModel):
    alias: str
    serial_number: str | None = None
    usb_port: str | None = None
    vid: int | None = None
    pid: int | None = None
    usb_chip: str | None = None  # USB-UART converter (CH340, CP2102, …)
    target_chip: TargetChip  # ESP family — required, drives flash layout
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
    usb_chip: str | None = None
    target_chip: TargetChip = DEFAULT_TARGET_CHIP


class DiscoveredPortWithRegistration(BaseModel):
    port: DiscoveredPort
    matched_alias: str | None = None


# ── Chip inference ───────────────────────────────────────────────────────


def guess_usb_chip(
    vid: int | None,
    pid: int | None,
    manufacturer: str | None = None,
    product: str | None = None,
) -> str | None:
    """Best-effort USB-UART converter chip name from VID/PID and strings."""
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
    usb_chip = guess_usb_chip(vid, pid, manufacturer, product)
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
        usb_chip=usb_chip,
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
    """Load the registry, applying old-format migrations on the way.

    Migrations performed:
      • field rename ``chip`` → ``usb_chip``
      • inject ``target_chip = DEFAULT_TARGET_CHIP`` for entries that lack
        the field, with a warning log so the user knows to run set-target.

    Migrated registries are persisted on the way out so we don't repeat the
    rename next time. The defaulted target_chip is also persisted; the
    warning fires only on the first load that needs the migration.
    """
    path = _registry_path()
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out: dict[str, DeviceRegistration] = {}
    needs_persist = False
    for alias, data in raw.items():
        if "chip" in data and "usb_chip" not in data:
            data["usb_chip"] = data.pop("chip")
            needs_persist = True
        if not data.get("target_chip"):
            logger.warning(
                "device %r has no target_chip in registry, defaulted to %s. "
                "Run `teletop-server set-target %s <chip>` to correct.",
                alias,
                DEFAULT_TARGET_CHIP,
                alias,
            )
            data["target_chip"] = DEFAULT_TARGET_CHIP
            needs_persist = True
        out[alias] = DeviceRegistration.model_validate(data)
    if needs_persist:
        save_registry(out)
    return out


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


def validate_alias(alias: str) -> None:
    """Reject aliases that would produce ill-formed udev SYMLINK names."""
    if not ALIAS_PATTERN.match(alias):
        raise DeviceRegistryError(
            f"alias {alias!r} is invalid — use letters, digits, '_' or '-' only "
            "and start with an alphanumeric character"
        )


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
    target_chip: TargetChip,
    port: DiscoveredPort | None = None,
    serial_number: str | None = None,
    usb_port: str | None = None,
    vid: int | None = None,
    pid: int | None = None,
    usb_chip: str | None = None,
    notes: str | None = None,
) -> DeviceRegistration:
    """Persist a new alias. ``target_chip`` is mandatory.

    Pass either a discovered port or explicit fields. When both are given,
    explicit values win. Identity precedence: serial first, then usb_port.
    """
    registry = load_registry()

    if port is not None:
        serial_number = serial_number if serial_number is not None else port.serial_number
        usb_port = usb_port if usb_port is not None else port.usb_port
        vid = vid if vid is not None else port.vid
        pid = pid if pid is not None else port.pid
        if usb_chip is None:
            usb_chip = port.usb_chip or guess_usb_chip(
                vid, pid, port.manufacturer, port.product
            )
    if usb_chip is None:
        usb_chip = guess_usb_chip(vid, pid)

    validate_alias(alias)
    _validate_unique(registry, alias=alias, serial_number=serial_number, usb_port=usb_port)

    reg = DeviceRegistration(
        alias=alias,
        serial_number=serial_number,
        usb_port=usb_port,
        vid=vid,
        pid=pid,
        usb_chip=usb_chip,
        target_chip=target_chip,
        created_at=datetime.now(timezone.utc),
        notes=notes,
    )
    registry[alias] = reg
    save_registry(registry)
    return reg


def update_device(
    alias: str,
    *,
    target_chip: TargetChip | None = None,
    notes: str | None = None,
) -> DeviceRegistration:
    """Mutate an existing registration. ``serial_number`` and ``usb_port``
    are immutable here — re-register if those need to change."""
    registry = load_registry()
    if alias not in registry:
        raise KeyError(alias)
    updates: dict[str, object] = {}
    if target_chip is not None:
        updates["target_chip"] = target_chip
    if notes is not None:
        updates["notes"] = notes
    if not updates:
        return registry[alias]
    new_reg = registry[alias].model_copy(update=updates)
    registry[alias] = new_reg
    save_registry(registry)
    return new_reg


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
        sym_path = symlink_for(alias)
        # `os.path.lexists` returns True even if the symlink dangles — udev
        # may leave the file on disk for a moment after unplug, but we still
        # want to report the symlink path that the user configured.
        live_symlink = str(sym_path) if os.path.lexists(sym_path) else reg.udev_symlink
        statuses.append(
            DeviceStatus(
                alias=alias,
                identity_type=identity_type,
                identity_value=identity_value,
                udev_symlink=live_symlink,
                current_device=port.device if port else None,
                connected=port is not None,
                vid=reg.vid,
                pid=reg.pid,
                usb_chip=reg.usb_chip,
                target_chip=reg.target_chip,
            )
        )
    statuses.sort(key=lambda s: s.alias)
    return statuses


# ── Target chip auto-detect (esptool) ────────────────────────────────────


def _parse_target_chip_from_esptool(output: str) -> TargetChip:
    """Map an esptool ``chip_id`` stdout/stderr blob to a TargetChip literal.

    esptool prints the chip name in a few different shapes:
      "Chip is ESP32-D0WD-V3 (revision v3.1)"
      "Chip is ESP32-S3 (revision v0.1)"
      "Chip is ESP8266EX"
      "Detecting chip type... Unsupported detection protocol, ..."
    """
    text = output.lower()
    # More specific names first — "esp32-s3" must beat the generic "esp32".
    candidates: list[tuple[str, TargetChip]] = [
        ("esp32-s3", "esp32s3"),
        ("esp32s3", "esp32s3"),
        ("esp32-s2", "esp32s2"),
        ("esp32s2", "esp32s2"),
        ("esp32-c6", "esp32c6"),
        ("esp32c6", "esp32c6"),
        ("esp32-c3", "esp32c3"),
        ("esp32c3", "esp32c3"),
        ("esp32-h2", "esp32h2"),
        ("esp32h2", "esp32h2"),
        ("esp8266", "esp8266"),
        ("esp32", "esp32"),
    ]
    for needle, chip in candidates:
        if needle in text:
            return chip
    raise RuntimeError(
        "could not detect target chip from esptool output:\n" + output[:2000]
    )


def detect_target_chip(device_path: str, *, timeout: float = 12.0) -> TargetChip:
    """Run ``esptool.py --port <device> chip_id`` and parse the result.

    The chip is briefly bounced through bootloader; the device should not be
    actively running important code while this runs.
    """
    result = subprocess.run(
        ["esptool.py", "--port", device_path, "chip_id"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    blob = (result.stdout or "") + (result.stderr or "")
    return _parse_target_chip_from_esptool(blob)


# ── udev rule generation ─────────────────────────────────────────────────


_RULES_HEADER = """\
# Generated by teletop-server — DO NOT EDIT MANUALLY.
# Regenerate with: teletop-server udev-rules
# Reload with:    sudo udevadm control --reload && sudo udevadm trigger
#
"""


def symlink_for(alias: str) -> Path:
    """Path of the stable /dev symlink that udev will create for `alias`."""
    return SYMLINK_DIR / f"{SYMLINK_PREFIX}{alias}"


def _format_rule(reg: DeviceRegistration) -> str:
    """Render one DeviceRegistration as a multi-line udev rule.

    Returns a `# skipped …` comment if the registration lacks VID/PID — a
    rule without a vendor match would be dangerously broad (it could match
    keyboards or hubs that happen to share the same USB port path).
    """
    if reg.vid is None or reg.pid is None:
        return (
            f"# skipped {reg.alias}: missing VID/PID — re-register with the "
            "device plugged in to capture them.\n"
        )
    if reg.serial_number:
        identity_clause = f'ATTRS{{serial}}=="{reg.serial_number}"'
        identity_kind = "serial"
    elif reg.usb_port:
        identity_clause = f'KERNELS=="{reg.usb_port}"'
        identity_kind = "port"
    else:  # pragma: no cover — model validator forbids this state
        return f"# skipped {reg.alias}: no identity\n"

    parts = [
        'SUBSYSTEM=="tty"',
        f'ATTRS{{idVendor}}=="{reg.vid:04x}"',
        f'ATTRS{{idProduct}}=="{reg.pid:04x}"',
        identity_clause,
        f'SYMLINK+="{SYMLINK_PREFIX}{reg.alias}"',
        'GROUP="dialout"',
        'MODE="0660"',
        f'ENV{{TELETOP_ALIAS}}="{reg.alias}"',
        f'ENV{{TELETOP_TARGET}}="{reg.target_chip}"',
    ]
    usb_label = reg.usb_chip or "unknown USB"
    header = (
        f"# {reg.alias} (target={reg.target_chip}, usb={usb_label}, "
        f"{identity_kind}-based identity)\n"
    )
    body = ", \\\n    ".join(parts)
    return header + body + "\n"


def generate_udev_rules(
    registry: dict[str, DeviceRegistration] | None = None,
) -> str:
    """Render the full /etc/udev/rules.d/99-teletop.rules content."""
    if registry is None:
        registry = load_registry()
    out: list[str] = [_RULES_HEADER]
    if not registry:
        out.append("# (registry empty — register a device first)\n")
        return "".join(out)
    for alias in sorted(registry):
        out.append(_format_rule(registry[alias]))
        out.append("\n")
    return "".join(out)


def _require_root(action: str) -> None:
    if os.geteuid() != 0:
        raise PermissionError(
            f"{action} requires root — run as: "
            "sudo $(which uv) run teletop-server " + action.replace("_", "-")
        )


def _udev_reload() -> None:
    subprocess.run(["udevadm", "control", "--reload"], check=True)
    subprocess.run(["udevadm", "trigger"], check=True)


def install_udev_rules(content: str | None = None) -> Path:
    """Write rules atomically + reload udev. Returns the destination path."""
    _require_root("udev-install")
    if content is None:
        content = generate_udev_rules()
    UDEV_RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = UDEV_RULES_PATH.with_suffix(UDEV_RULES_PATH.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, UDEV_RULES_PATH)
    _udev_reload()
    return UDEV_RULES_PATH


def uninstall_udev_rules() -> bool:
    """Remove the rules file and reload. Returns False if nothing to remove."""
    _require_root("udev-uninstall")
    if not UDEV_RULES_PATH.exists():
        return False
    UDEV_RULES_PATH.unlink()
    _udev_reload()
    return True


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
    "ALIAS_PATTERN",
    "DEFAULT_TARGET_CHIP",
    "DeviceRegistration",
    "DeviceRegistryError",
    "DeviceStatus",
    "DiscoveredPort",
    "DiscoveredPortWithRegistration",
    "ESP_VENDOR_IDS",
    "SYMLINK_DIR",
    "SYMLINK_PREFIX",
    "TARGET_CHIPS",
    "TargetChip",
    "UDEV_RULES_PATH",
    "detect_target_chip",
    "discover_ports",
    "discover_with_registrations",
    "generate_udev_rules",
    "get_device_status",
    "guess_usb_chip",
    "install_udev_rules",
    "load_registry",
    "match_registration",
    "register_device",
    "save_registry",
    "symlink_for",
    "uninstall_udev_rules",
    "unregister_device",
    "update_device",
    "validate_alias",
]
