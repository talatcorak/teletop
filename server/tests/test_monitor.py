"""Tests for the SerialMonitor + MonitorRegistry + monitor REST/CLI surface.

We avoid touching real serial hardware by monkeypatching
``SerialMonitor._open_serial`` to return a controllable
``(asyncio.StreamReader, asyncio.StreamWriter)`` pair. The reader can be
fed bytes with ``feed_data()`` / ``feed_eof()`` and the broadcast side
either uses a captured-list fake or the real ``manager`` against
``TestClient`` WebSockets.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from teletop_server import monitor as monitor_mod
from teletop_server.devices import register_device
from teletop_server.monitor import (
    LogPathError,
    MonitorConfig,
    MonitorRegistry,
    SerialMonitor,
    channel_for,
    list_log_files,
    resolve_log_file,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Reuse the same isolation pattern as test_devices."""
    monkeypatch.setenv("TELETOP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TELETOP_LOG_DIR", str(tmp_path / "logs"))
    return tmp_path


@pytest.fixture(autouse=True)
async def _clear_registry() -> Any:
    """Make sure no leftover monitor from a prior test leaks state."""
    yield
    await monitor_mod.registry.stop_all()


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_fake_pair(loop_limit: int = 1024 * 1024) -> tuple[
    asyncio.StreamReader, "FakeWriter"
]:
    reader = asyncio.StreamReader(limit=loop_limit)

    class FakeWriter:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

        def write(self, data: bytes) -> None:  # pragma: no cover — unused in tests
            pass

    return reader, FakeWriter()


async def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f"timeout waiting for {predicate}")


# ── SerialMonitor — line broadcast ────────────────────────────────────────


@pytest.mark.asyncio
async def test_serial_monitor_broadcasts_lines(tmp_path: Path) -> None:
    """Bytes pushed at the reader become broadcast events on the right channel."""
    reader, writer = _make_fake_pair()
    captured: list[tuple[str, dict[str, Any]]] = []

    async def fake_broadcast(channel: str, payload: dict[str, Any] | str) -> None:
        assert isinstance(payload, dict)
        captured.append((channel, payload))

    cfg = MonitorConfig(
        alias="agv1", device_path="loop://", baudrate=115200, log_dir=tmp_path
    )
    mon = SerialMonitor(cfg, broadcast=fake_broadcast)
    mon._open_serial = AsyncMock(return_value=(reader, writer))  # type: ignore[method-assign]

    await mon.start()
    reader.feed_data(b"line1\nline2\n")
    await _wait_until(
        lambda: sum(1 for c, p in captured if p.get("type") == "line") >= 2
    )
    await mon.stop()

    line_events = [p for _c, p in captured if p["type"] == "line"]
    assert [e["text"] for e in line_events] == ["line1", "line2"]
    for evt in line_events:
        assert evt["alias"] == "agv1"
        assert evt["ts"].endswith("Z")
    # status events sandwich the line events
    states = [p["state"] for _c, p in captured if p["type"] == "status"]
    assert "connected" in states
    assert "stopped" in states
    # all events broadcast on the right channel
    assert {c for c, _ in captured} == {"monitor:agv1"}


@pytest.mark.asyncio
async def test_serial_monitor_handles_partial_lines(tmp_path: Path) -> None:
    """A line split across reads must arrive whole — readline buffers for us."""
    reader, writer = _make_fake_pair()
    captured: list[dict[str, Any]] = []

    async def fake_broadcast(channel: str, payload: dict[str, Any]) -> None:
        captured.append(payload)

    mon = SerialMonitor(
        MonitorConfig("a", "loop://", log_dir=tmp_path), broadcast=fake_broadcast
    )
    mon._open_serial = AsyncMock(return_value=(reader, writer))  # type: ignore[method-assign]
    await mon.start()
    reader.feed_data(b"hello ")
    await asyncio.sleep(0.02)
    # Nothing should have been broadcast as a line yet.
    assert not [p for p in captured if p["type"] == "line"]
    reader.feed_data(b"world\r\n")
    await _wait_until(lambda: any(p["type"] == "line" for p in captured))
    await mon.stop()
    [line] = [p for p in captured if p["type"] == "line"]
    # \r\n stripped, partial pieces stitched.
    assert line["text"] == "hello world"


@pytest.mark.asyncio
async def test_serial_monitor_disconnect_emits_status(tmp_path: Path) -> None:
    """An OSError from the reader surfaces as a 'disconnected' status."""
    reader, writer = _make_fake_pair()
    statuses: list[dict[str, Any]] = []

    async def fake_broadcast(channel: str, payload: dict[str, Any]) -> None:
        if payload["type"] == "status":
            statuses.append(payload)

    mon = SerialMonitor(
        MonitorConfig("a", "loop://", log_dir=tmp_path), broadcast=fake_broadcast
    )
    mon._open_serial = AsyncMock(return_value=(reader, writer))  # type: ignore[method-assign]
    await mon.start()
    # Simulate an EIO-style failure after some traffic.
    reader.set_exception(OSError(5, "Input/output error"))
    await _wait_until(
        lambda: any(s["state"] == "disconnected" for s in statuses)
    )
    await mon.stop()

    disc = [s for s in statuses if s["state"] == "disconnected"]
    assert disc, "expected a disconnected status"
    assert "Input/output error" in disc[0]["reason"]
    assert mon.is_running() is False


@pytest.mark.asyncio
async def test_serial_monitor_eof_treated_as_disconnect(tmp_path: Path) -> None:
    reader, writer = _make_fake_pair()
    statuses: list[dict[str, Any]] = []

    async def fake_broadcast(channel: str, payload: dict[str, Any]) -> None:
        if payload["type"] == "status":
            statuses.append(payload)

    mon = SerialMonitor(
        MonitorConfig("a", "loop://", log_dir=tmp_path), broadcast=fake_broadcast
    )
    mon._open_serial = AsyncMock(return_value=(reader, writer))  # type: ignore[method-assign]
    await mon.start()
    reader.feed_eof()
    await _wait_until(
        lambda: any(s["state"] == "disconnected" for s in statuses)
    )
    await mon.stop()
    disc = [s for s in statuses if s["state"] == "disconnected"]
    assert disc and "EOF" in disc[0]["reason"]


# ── Log files ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_serial_monitor_writes_log_file(tmp_path: Path) -> None:
    reader, writer = _make_fake_pair()
    captured: list[dict[str, Any]] = []

    async def fake_broadcast(channel: str, payload: dict[str, Any]) -> None:
        captured.append(payload)

    cfg = MonitorConfig("agv1", "loop://", log_dir=tmp_path)
    mon = SerialMonitor(cfg, broadcast=fake_broadcast)
    mon._open_serial = AsyncMock(return_value=(reader, writer))  # type: ignore[method-assign]
    await mon.start()
    reader.feed_data(b"hb #1\nhb #2\n")
    await _wait_until(
        lambda: sum(1 for p in captured if p["type"] == "line") >= 2
    )
    log_path = mon.log_path
    assert log_path is not None
    await mon.stop()

    assert log_path.exists()
    contents = log_path.read_text().splitlines()
    assert contents == ["hb #1", "hb #2"]
    # Filename pattern: YYYY-MM-DD_HHMMSS.log inside <root>/<alias>/
    assert log_path.parent.name == "agv1"
    assert log_path.suffix == ".log"


# ── Backpressure ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_serial_monitor_drops_oldest_when_queue_full(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Tiny queue + slow consumer → oldest lines get dropped, warning logged."""
    import logging

    caplog.set_level(logging.WARNING, logger="teletop.monitor")

    reader, writer = _make_fake_pair()
    cfg = MonitorConfig("a", "loop://", log_dir=tmp_path, queue_size=2)

    block_event = asyncio.Event()

    async def slow_broadcast(channel: str, payload: dict[str, Any]) -> None:
        # Block the consumer just long enough for the queue to back up.
        if payload.get("type") == "line":
            await block_event.wait()

    mon = SerialMonitor(cfg, broadcast=slow_broadcast)
    mon._open_serial = AsyncMock(return_value=(reader, writer))  # type: ignore[method-assign]
    await mon.start()

    # Push more lines than the queue can hold; consumer is blocked, so the
    # reader will overflow the queue and trigger drops.
    for i in range(20):
        reader.feed_data(f"line{i}\n".encode())
    await _wait_until(lambda: mon.dropped > 0, timeout=2.0)
    assert mon.dropped > 0
    assert any(
        "queue full" in record.message for record in caplog.records
    ), "expected a queue-full warning"

    block_event.set()
    await mon.stop()


# ── MonitorRegistry — idempotency ────────────────────────────────────────


@pytest.mark.asyncio
async def test_registry_start_is_idempotent(tmp_path: Path) -> None:
    reg = MonitorRegistry()

    async def open_serial(self):
        return _make_fake_pair()

    with patch.object(SerialMonitor, "_open_serial", open_serial):
        cfg = MonitorConfig("agv1", "loop://", log_dir=tmp_path)
        mon1, already1 = await reg.start(cfg)
        mon2, already2 = await reg.start(cfg)

        assert mon1 is mon2
        assert already1 is False
        assert already2 is True
        assert reg.list_active() == ["agv1"]

        await reg.stop_all()
        assert reg.list_active() == []


@pytest.mark.asyncio
async def test_registry_stop_unknown_returns_false() -> None:
    reg = MonitorRegistry()
    assert await reg.stop("nope") is False


# ── REST endpoints ───────────────────────────────────────────────────────


def test_api_monitor_start_unknown_alias_404(isolated_data_dir: Path) -> None:
    from teletop_server.main import app

    with TestClient(app) as client:
        r = client.post("/api/monitor/ghost/start")
        assert r.status_code == 404
        assert "ghost" in r.json()["detail"]


def test_api_monitor_start_disconnected_alias_409(isolated_data_dir: Path) -> None:
    """Registered, but not currently plugged in → 409 Conflict."""
    register_device("offline", usb_port="9-9", target_chip="esp32")
    from teletop_server.main import app

    # No fake_ports fixture means current_device will be None.
    with TestClient(app) as client:
        r = client.post("/api/monitor/offline/start")
        assert r.status_code == 409
        assert "not connected" in r.json()["detail"]


def test_api_monitor_start_idempotent(
    isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Start twice → second call returns already_running=True."""
    register_device("agv1", usb_port="3-1", target_chip="esp32")

    # Pretend the device is connected.
    from teletop_server import devices as devices_mod
    from teletop_server import main as main_mod
    from teletop_server.devices import DeviceStatus

    def fake_status() -> list[DeviceStatus]:
        return [
            DeviceStatus(
                alias="agv1",
                identity_type="usb_port",
                identity_value="3-1",
                udev_symlink="/dev/tty-agv1",
                current_device="/dev/ttyUSB0",
                connected=True,
                target_chip="esp32",
            )
        ]

    monkeypatch.setattr(main_mod, "get_device_status", fake_status)
    monkeypatch.setattr(devices_mod, "get_device_status", fake_status)

    # Patch SerialMonitor._open_serial so no real port is touched. Construct
    # the reader inside the handler so it's bound to the request's event loop
    # (TestClient runs the app on a separate loop from the test thread).
    async def open_serial(self):
        return _make_fake_pair()

    monkeypatch.setattr(SerialMonitor, "_open_serial", open_serial)

    from teletop_server.main import app

    with TestClient(app) as client:
        r1 = client.post("/api/monitor/agv1/start")
        assert r1.status_code == 200, r1.text
        body1 = r1.json()
        assert body1["alias"] == "agv1"
        assert body1["device_path"] == "/dev/tty-agv1"
        assert body1["already_running"] is False

        r2 = client.post("/api/monitor/agv1/start")
        assert r2.status_code == 200
        assert r2.json()["already_running"] is True

        rlist = client.get("/api/monitor")
        assert rlist.status_code == 200
        rows = {row["alias"]: row for row in rlist.json()}
        assert rows["agv1"]["running"] is True

        rstop = client.post("/api/monitor/agv1/stop")
        assert rstop.status_code == 200
        assert rstop.json()["stopped"] is True

        rstop2 = client.post("/api/monitor/agv1/stop")
        # idempotent: 200 with stopped=False
        assert rstop2.status_code == 200
        assert rstop2.json()["stopped"] is False


def test_api_monitor_logs_listing_and_download(
    isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """List + download log files for a known alias."""
    register_device("agv1", usb_port="3-1", target_chip="esp32")
    log_root = isolated_data_dir / "logs"
    (log_root / "agv1").mkdir(parents=True)
    f1 = log_root / "agv1" / "2026-05-09_120000.log"
    f1.write_text("hello\nworld\n")

    monkeypatch.setenv("TELETOP_LOG_DIR", str(log_root))

    from teletop_server.main import app

    with TestClient(app) as client:
        r = client.get("/api/monitor/agv1/logs")
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) == 1
        assert rows[0]["filename"] == "2026-05-09_120000.log"
        assert rows[0]["size"] == len(b"hello\nworld\n")

        rd = client.get("/api/monitor/agv1/logs/2026-05-09_120000.log")
        assert rd.status_code == 200
        assert rd.text == "hello\nworld\n"

        # path traversal must be rejected
        r404 = client.get("/api/monitor/agv1/logs/..%2Fevil")
        assert r404.status_code in (400, 404)


def test_api_monitor_logs_unknown_alias_404(isolated_data_dir: Path) -> None:
    from teletop_server.main import app

    with TestClient(app) as client:
        r = client.get("/api/monitor/ghost/logs")
        assert r.status_code == 404


# ── Path traversal unit tests ────────────────────────────────────────────


def test_resolve_log_file_rejects_traversal(tmp_path: Path) -> None:
    (tmp_path / "agv1").mkdir()
    (tmp_path / "agv1" / "ok.log").write_text("hi")
    (tmp_path / "secret.log").write_text("nope")

    # Bare filename works
    assert resolve_log_file(tmp_path, "agv1", "ok.log").read_text() == "hi"
    # ".." rejected
    with pytest.raises(LogPathError):
        resolve_log_file(tmp_path, "agv1", "..")
    with pytest.raises(LogPathError):
        resolve_log_file(tmp_path, "agv1", "../secret.log")
    with pytest.raises(LogPathError):
        resolve_log_file(tmp_path, "agv1", "subdir/file.log")


def test_list_log_files_empty_when_no_dir(tmp_path: Path) -> None:
    assert list_log_files(tmp_path, "ghost") == []


# ── Lifespan shutdown calls stop_all ─────────────────────────────────────


def test_lifespan_shutdown_calls_stop_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """The FastAPI shutdown hook must drain the registry."""
    called = asyncio.Event()
    original_stop_all = monitor_mod.registry.stop_all

    async def tracking_stop_all() -> None:
        called.set()
        await original_stop_all()

    monkeypatch.setattr(monitor_mod.registry, "stop_all", tracking_stop_all)
    # main.py imported `registry as monitor_registry`, so patch the binding it sees.
    from teletop_server import main as main_mod

    monkeypatch.setattr(main_mod, "monitor_registry", monitor_mod.registry)

    from teletop_server.main import app

    with TestClient(app):
        pass  # __exit__ runs lifespan shutdown
    assert called.is_set()


# ── WebSocket smoke test ─────────────────────────────────────────────────


def test_ws_monitor_emits_initial_status(isolated_data_dir: Path) -> None:
    """On connect, the server pushes the current monitor state."""
    from teletop_server.main import app

    with TestClient(app) as client:
        with client.websocket_connect("/ws/monitor/ghost") as ws:
            initial = ws.receive_json()
            assert initial == {"type": "status", "alias": "ghost", "state": "stopped"}


# ── CLI smoke ────────────────────────────────────────────────────────────


def test_cli_monitor_subgroup_help() -> None:
    from teletop_server.main import cli

    result = CliRunner().invoke(cli, ["monitor", "--help"])
    assert result.exit_code == 0, result.output
    for sub in ("start", "stop", "list", "tail"):
        assert sub in result.output


def test_cli_all_expected_top_level_commands_registered() -> None:
    """Guard against silent loss of CLI commands during refactors."""
    from teletop_server.main import cli

    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    for cmd in (
        "discover",
        "register",
        "unregister",
        "list",
        "serve",
        "udev-rules",
        "udev-install",
        "udev-uninstall",
        "set-target",
        "monitor",
        "flash",
        "projects",
    ):
        assert cmd in result.output, f"missing top-level command: {cmd}"


def test_cli_monitor_start_complains_when_server_down() -> None:
    """The CLI talks to the running server — a clean error if it isn't."""
    from teletop_server.main import cli

    runner = CliRunner()
    # Use an unlikely-to-be-bound port to force ConnectError.
    import os

    os.environ["TELETOP_PORT"] = "59999"
    try:
        result = runner.invoke(cli, ["monitor", "start", "agv1"])
    finally:
        del os.environ["TELETOP_PORT"]
    assert result.exit_code != 0
    assert "could not reach server" in result.output.lower()
