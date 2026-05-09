"""Tests for device discovery, identity matching, and the registry CRUD/API.

pyudev is mocked end-to-end — no real /sys access is required, so the suite
is deterministic on dev machines without ESP32 hardware attached.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from teletop_server import devices
from teletop_server.devices import (
    DeviceRegistration,
    DeviceRegistryError,
    DiscoveredPort,
    discover_ports,
    discover_with_registrations,
    get_device_status,
    guess_chip,
    load_registry,
    match_registration,
    register_device,
    save_registry,
    unregister_device,
)


# ── Fake pyudev objects ──────────────────────────────────────────────────


class _FakeAttrs:
    def __init__(self, attrs: dict[str, str | bytes | None]) -> None:
        self._attrs = attrs

    def get(self, key: str) -> bytes | None:
        v = self._attrs.get(key)
        if v is None:
            return None
        return v.encode() if isinstance(v, str) else v


class _FakeProps(dict):
    pass


class _FakeUSBDevice:
    def __init__(self, sys_name: str, attrs: dict[str, str | None]) -> None:
        self.sys_name = sys_name
        self.attributes = _FakeAttrs(attrs)


class _FakeTTYDevice:
    def __init__(
        self,
        device_node: str | None,
        parent_usb: _FakeUSBDevice | None,
        properties: dict[str, str] | None = None,
    ) -> None:
        self.device_node = device_node
        self._parent_usb = parent_usb
        self.properties = _FakeProps(properties or {})

    def find_parent(self, subsystem: str, device_type: str) -> _FakeUSBDevice | None:
        if subsystem == "usb" and device_type == "usb_device":
            return self._parent_usb
        return None


class _FakeContext:
    def __init__(self, ttys: Iterable[_FakeTTYDevice]) -> None:
        self._ttys = list(ttys)

    def list_devices(self, subsystem: str) -> list[_FakeTTYDevice]:
        return list(self._ttys) if subsystem == "tty" else []


def _ch340_port(node: str = "/dev/ttyUSB0", usb_port: str = "3-1") -> _FakeTTYDevice:
    parent = _FakeUSBDevice(
        sys_name=usb_port,
        attrs={
            "idVendor": "1a86",
            "idProduct": "7523",
            "manufacturer": "QinHeng Electronics",
            "product": "USB Serial",
            "serial": None,
        },
    )
    return _FakeTTYDevice(node, parent, {"ID_PATH": f"pci-0000:00:14.0-usb-0:{usb_port}"})


def _cp2102_port(
    node: str = "/dev/ttyUSB1",
    usb_port: str = "3-2",
    serial: str = "0001",
) -> _FakeTTYDevice:
    parent = _FakeUSBDevice(
        sys_name=usb_port,
        attrs={
            "idVendor": "10c4",
            "idProduct": "ea60",
            "manufacturer": "Silicon Labs",
            "product": "CP2102 USB to UART Bridge Controller",
            "serial": serial,
        },
    )
    return _FakeTTYDevice(node, parent, {"ID_PATH": f"pci-0000:00:14.0-usb-0:{usb_port}"})


def _non_esp_port(node: str = "/dev/ttyUSB9") -> _FakeTTYDevice:
    parent = _FakeUSBDevice(
        sys_name="3-9",
        attrs={
            "idVendor": "1234",
            "idProduct": "5678",
            "manufacturer": "Some Vendor",
            "product": "Random USB Device",
            "serial": None,
        },
    )
    return _FakeTTYDevice(node, parent)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the registry to a clean tmp_path for each test."""
    monkeypatch.setenv("TELETOP_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_ports(monkeypatch: pytest.MonkeyPatch) -> list[_FakeTTYDevice]:
    """Patch discover_ports to use a fake udev context.

    Tests append to the returned list; a fresh _FakeContext is built each
    discovery call so updates take effect immediately.
    """
    devs: list[_FakeTTYDevice] = []

    def _patched_discover(
        *, esp_only: bool = True, context=None  # type: ignore[no-untyped-def]
    ) -> list[DiscoveredPort]:
        ctx = _FakeContext(devs)
        return _real_discover(esp_only=esp_only, context=ctx)

    _real_discover = devices.discover_ports
    monkeypatch.setattr(devices, "discover_ports", _patched_discover)
    # Also patch in main.py since it imports the symbol directly
    from teletop_server import main as main_mod

    monkeypatch.setattr(main_mod, "discover_ports", _patched_discover)
    return devs


# ── Chip inference ───────────────────────────────────────────────────────


def test_guess_chip_known_vendors() -> None:
    assert guess_chip(0x1A86, 0x7523) == "CH340"
    assert guess_chip(0x10C4, 0xEA60) == "CP2102"
    assert guess_chip(0x303A, 0x1001) == "ESP32-USB"
    assert guess_chip(0x0403, 0x6001) == "FTDI"


def test_guess_chip_falls_back_to_strings() -> None:
    assert guess_chip(None, None, "Whoever", "CP2104 USB Bridge") == "CP2104"
    assert guess_chip(None, None, None, None) is None


# ── Discovery ────────────────────────────────────────────────────────────


def test_discover_filters_to_esp_vendors_by_default() -> None:
    ctx = _FakeContext([_ch340_port(), _cp2102_port(), _non_esp_port()])
    ports = discover_ports(esp_only=True, context=ctx)
    assert {p.device for p in ports} == {"/dev/ttyUSB0", "/dev/ttyUSB1"}


def test_discover_with_all_includes_other_vendors() -> None:
    ctx = _FakeContext([_ch340_port(), _non_esp_port()])
    ports = discover_ports(esp_only=False, context=ctx)
    assert {p.device for p in ports} == {"/dev/ttyUSB0", "/dev/ttyUSB9"}


def test_discover_extracts_metadata() -> None:
    ctx = _FakeContext([_cp2102_port(usb_port="3-1.2", serial="ABCD1234")])
    ports = discover_ports(context=ctx)
    assert len(ports) == 1
    p = ports[0]
    assert p.vid == 0x10C4
    assert p.pid == 0xEA60
    assert p.serial_number == "ABCD1234"
    assert p.usb_port == "3-1.2"
    assert p.chip == "CP2102"
    assert p.id_path is not None and "usb-0:3-1.2" in p.id_path


def test_discover_skips_devices_without_usb_parent() -> None:
    orphan = _FakeTTYDevice("/dev/ttyUSB7", parent_usb=None)
    ctx = _FakeContext([orphan, _ch340_port()])
    ports = discover_ports(esp_only=False, context=ctx)
    assert {p.device for p in ports} == {"/dev/ttyUSB0"}


def test_discover_ignores_non_usb_serial_nodes() -> None:
    bogus = _FakeTTYDevice("/dev/tty0", parent_usb=None)
    ctx = _FakeContext([bogus, _ch340_port()])
    ports = discover_ports(esp_only=False, context=ctx)
    assert {p.device for p in ports} == {"/dev/ttyUSB0"}


# ── Identity matching ────────────────────────────────────────────────────


def _reg(
    alias: str,
    *,
    serial: str | None = None,
    usb_port: str | None = None,
    chip: str | None = None,
) -> DeviceRegistration:
    return DeviceRegistration(
        alias=alias,
        serial_number=serial,
        usb_port=usb_port,
        chip=chip,
        created_at=datetime.now(timezone.utc),
    )


def test_match_by_serial_when_available() -> None:
    port = DiscoveredPort(device="/dev/ttyUSB0", serial_number="ABC", usb_port="3-1")
    registry = {"a": _reg("a", serial="ABC", usb_port="9-9")}  # different port, same serial
    assert match_registration(port, registry).alias == "a"


def test_match_by_usb_port_only_when_reg_has_no_serial() -> None:
    """Port-path fallback must only fire for serial-less registrations."""
    port = DiscoveredPort(device="/dev/ttyUSB0", serial_number=None, usb_port="3-1")
    # Registration *does* carry a serial — port is on the same socket but
    # the serial differs (None vs ABC), so we must NOT match.
    registry = {"a": _reg("a", serial="ABC", usb_port="3-1")}
    assert match_registration(port, registry) is None


def test_match_serial_less_port_to_serial_less_reg() -> None:
    port = DiscoveredPort(device="/dev/ttyUSB0", serial_number=None, usb_port="3-1")
    registry = {"agv1": _reg("agv1", usb_port="3-1")}
    assert match_registration(port, registry).alias == "agv1"


def test_match_no_match_returns_none() -> None:
    port = DiscoveredPort(device="/dev/ttyUSB0", serial_number="ZZZ", usb_port="9-9")
    registry = {"a": _reg("a", serial="AAA")}
    assert match_registration(port, registry) is None


# ── Registry CRUD ────────────────────────────────────────────────────────


def test_register_then_load_round_trip(isolated_data_dir: Path) -> None:
    reg = register_device(
        "agv1",
        usb_port="3-1",
        vid=0x1A86,
        pid=0x7523,
        notes="left-side bot",
    )
    assert reg.alias == "agv1"
    assert reg.chip == "CH340"

    loaded = load_registry()
    assert "agv1" in loaded
    assert loaded["agv1"].usb_port == "3-1"
    assert loaded["agv1"].chip == "CH340"
    assert loaded["agv1"].notes == "left-side bot"


def test_register_alias_collision(isolated_data_dir: Path) -> None:
    register_device("agv1", usb_port="3-1")
    with pytest.raises(DeviceRegistryError, match="already registered"):
        register_device("agv1", usb_port="3-2")


def test_register_serial_collision(isolated_data_dir: Path) -> None:
    register_device("a", serial_number="DEAD")
    with pytest.raises(DeviceRegistryError, match="serial_number"):
        register_device("b", serial_number="DEAD")


def test_register_usb_port_collision_only_for_serialless(isolated_data_dir: Path) -> None:
    """Two CH340s plugged into the same socket can't co-exist as registrations."""
    register_device("a", usb_port="3-1")
    with pytest.raises(DeviceRegistryError, match="usb_port"):
        register_device("b", usb_port="3-1")


def test_register_usb_port_reuse_ok_when_existing_has_serial(
    isolated_data_dir: Path,
) -> None:
    """A serial-pinned registration doesn't 'own' the port; another reg can use it."""
    register_device("a", serial_number="AAA", usb_port="3-1")
    # b is a different chip plugged into the same socket; only b lacks a
    # serial, so its port-path identity is its own — no collision with a.
    register_device("b", usb_port="3-1")
    assert set(load_registry().keys()) == {"a", "b"}


def test_register_requires_some_identity(isolated_data_dir: Path) -> None:
    with pytest.raises(DeviceRegistryError, match="requires"):
        register_device("ghost")


def test_register_from_discovered_port_prefers_serial(isolated_data_dir: Path) -> None:
    port = DiscoveredPort(
        device="/dev/ttyUSB0",
        vid=0x10C4,
        pid=0xEA60,
        serial_number="ABCD",
        usb_port="3-1",
        chip="CP2102",
    )
    reg = register_device("cp", port=port)
    # Both fields are stored; matching prioritizes serial.
    assert reg.serial_number == "ABCD"
    assert reg.usb_port == "3-1"
    assert reg.chip == "CP2102"


def test_unregister_removes(isolated_data_dir: Path) -> None:
    register_device("a", usb_port="3-1")
    unregister_device("a")
    assert load_registry() == {}


def test_unregister_missing_raises(isolated_data_dir: Path) -> None:
    with pytest.raises(KeyError):
        unregister_device("nope")


def test_save_registry_atomic(isolated_data_dir: Path) -> None:
    """Verify the write goes through a tmp file that gets renamed."""
    reg_path = isolated_data_dir / "devices.json"
    register_device("a", usb_port="3-1")
    assert reg_path.exists()
    # No leftover .tmp file after success.
    assert not (isolated_data_dir / "devices.json.tmp").exists()


def test_save_registry_round_trip_with_datetime(isolated_data_dir: Path) -> None:
    register_device("a", serial_number="ABC", notes="hello")
    raw = (isolated_data_dir / "devices.json").read_text()
    assert "ABC" in raw and "hello" in raw
    again = load_registry()
    assert isinstance(again["a"].created_at, datetime)


# ── get_device_status merge ──────────────────────────────────────────────


def test_status_marks_connected_for_serial_match(isolated_data_dir: Path) -> None:
    register_device("cp", serial_number="0001", usb_port="3-1", vid=0x10C4, pid=0xEA60)
    port = DiscoveredPort(
        device="/dev/ttyUSB0",
        serial_number="0001",
        usb_port="3-9",  # moved sockets — serial still wins
    )
    statuses = get_device_status(ports=[port])
    assert len(statuses) == 1
    s = statuses[0]
    assert s.connected
    assert s.identity_type == "serial"
    assert s.current_device == "/dev/ttyUSB0"


def test_status_marks_disconnected_when_no_match(isolated_data_dir: Path) -> None:
    register_device("ch", usb_port="3-1")
    statuses = get_device_status(ports=[])
    assert len(statuses) == 1
    assert statuses[0].connected is False
    assert statuses[0].current_device is None
    assert statuses[0].identity_type == "usb_port"


# ── API endpoints ────────────────────────────────────────────────────────


def _client(isolated_data_dir: Path) -> TestClient:
    # main.py was imported before the env var fixture ran for this test,
    # so the static-mount probe used the old data_dir; that's harmless.
    # We only care about API routes that re-read settings on each call.
    from teletop_server.main import app

    return TestClient(app)


def test_api_list_empty(isolated_data_dir: Path) -> None:
    with _client(isolated_data_dir) as client:
        r = client.get("/api/devices")
        assert r.status_code == 200
        assert r.json() == []


def test_api_register_then_list(isolated_data_dir: Path, fake_ports) -> None:  # type: ignore[no-untyped-def]
    fake_ports.append(_ch340_port(usb_port="3-1"))
    with _client(isolated_data_dir) as client:
        r = client.post(
            "/api/devices",
            json={"alias": "agv1", "usb_port": "3-1", "vid": 0x1A86, "pid": 0x7523},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["alias"] == "agv1"
        assert body["chip"] == "CH340"

        r = client.get("/api/devices")
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) == 1
        assert rows[0]["alias"] == "agv1"
        assert rows[0]["connected"] is True
        assert rows[0]["current_device"] == "/dev/ttyUSB0"


def test_api_register_validation_error(isolated_data_dir: Path) -> None:
    with _client(isolated_data_dir) as client:
        r = client.post("/api/devices", json={"alias": "ghost"})
        assert r.status_code == 400
        assert "requires" in r.json()["detail"]


def test_api_register_serial_collision(isolated_data_dir: Path) -> None:
    with _client(isolated_data_dir) as client:
        client.post("/api/devices", json={"alias": "a", "serial_number": "X"})
        r = client.post("/api/devices", json={"alias": "b", "serial_number": "X"})
        assert r.status_code == 400


def test_api_discover(isolated_data_dir: Path, fake_ports) -> None:  # type: ignore[no-untyped-def]
    fake_ports.append(_ch340_port())
    fake_ports.append(_cp2102_port(serial="SN-1"))
    with _client(isolated_data_dir) as client:
        r = client.get("/api/devices/discover")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 2
        # Register one then re-discover; matched_alias should populate.
        client.post("/api/devices", json={"alias": "cp", "serial_number": "SN-1"})
        r = client.get("/api/devices/discover")
        body = r.json()
        matched = {entry["matched_alias"] for entry in body}
        assert "cp" in matched


def test_api_unregister_204(isolated_data_dir: Path) -> None:
    with _client(isolated_data_dir) as client:
        client.post("/api/devices", json={"alias": "a", "usb_port": "3-1"})
        r = client.delete("/api/devices/a")
        assert r.status_code == 204


def test_api_unregister_missing_404(isolated_data_dir: Path) -> None:
    with _client(isolated_data_dir) as client:
        r = client.delete("/api/devices/nope")
        assert r.status_code == 404


# ── CLI ──────────────────────────────────────────────────────────────────


def test_cli_register_with_serial_flag(isolated_data_dir: Path) -> None:
    from teletop_server.main import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["register", "agv1", "--serial", "DEAD", "--note", "n"])
    assert result.exit_code == 0, result.output
    assert "registered" in result.output

    reg = load_registry()
    assert reg["agv1"].serial_number == "DEAD"
    assert reg["agv1"].notes == "n"


def test_cli_register_with_port_flag_warns_about_socket(isolated_data_dir: Path) -> None:
    from teletop_server.main import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["register", "ch", "--port", "3-1"])
    assert result.exit_code == 0, result.output
    assert load_registry()["ch"].usb_port == "3-1"


def test_cli_list_shows_rows(isolated_data_dir: Path) -> None:
    register_device("agv1", usb_port="3-1")
    from teletop_server.main import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["list"])
    assert result.exit_code == 0
    assert "agv1" in result.output


def test_cli_unregister_with_yes_flag(isolated_data_dir: Path) -> None:
    register_device("a", usb_port="3-1")
    from teletop_server.main import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["unregister", "a", "-y"])
    assert result.exit_code == 0
    assert load_registry() == {}


def test_cli_discover_renders_table(isolated_data_dir: Path, fake_ports) -> None:  # type: ignore[no-untyped-def]
    fake_ports.append(_ch340_port())
    fake_ports.append(_cp2102_port(serial="SN-1"))
    from teletop_server.main import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["discover"])
    assert result.exit_code == 0, result.output
    assert "/dev/ttyUSB0" in result.output
    assert "/dev/ttyUSB1" in result.output
    assert "CH340" in result.output
    assert "CP2102" in result.output


def test_cli_discover_empty_message(isolated_data_dir: Path, fake_ports) -> None:  # type: ignore[no-untyped-def]
    from teletop_server.main import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["discover"])
    assert result.exit_code == 0
    assert "No serial ports" in result.output


# ── Bonus: discover_with_registrations integration ───────────────────────


def test_discover_with_registrations_marks_match(
    isolated_data_dir: Path, fake_ports  # type: ignore[no-untyped-def]
) -> None:
    fake_ports.append(_cp2102_port(serial="SN-1"))
    register_device("cp", serial_number="SN-1")
    out = discover_with_registrations()
    assert len(out) == 1
    assert out[0].matched_alias == "cp"
