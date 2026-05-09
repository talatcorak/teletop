"""Tests for flash.py + flash REST/CLI surface.

Real esptool is never invoked. We monkeypatch ``asyncio.create_subprocess_exec``
(via FlashRunner's subprocess_factory injection) with a fake process that
yields scripted stdout lines and a configured return code.
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from teletop_server.devices import register_device
from teletop_server.flash import (
    FlashConflictError,
    FlashJob,
    FlashJobManager,
    build_esptool_cmd,
    list_jobs,
    read_job,
    trigger_flash,
    write_job,
)
from teletop_server.monitor import MonitorConfig, MonitorRegistry, SerialMonitor
from teletop_server.projects import upload_project


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TELETOP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TELETOP_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("TELETOP_FLASH_JOBS_DIR", str(tmp_path / "flash_jobs"))
    monkeypatch.setenv("TELETOP_LOG_DIR", str(tmp_path / "logs"))
    return tmp_path


# ── Helpers ──────────────────────────────────────────────────────────────


def _build_tarball(chip: str = "esp32") -> bytes:
    flasher_args = {
        "write_flash_args": ["--flash_mode", "dio", "--flash_size", "2MB"],
        "flash_files": {
            "0x1000": "bootloader/bootloader.bin",
            "0x10000": "app.bin",
        },
        "extra_esptool_args": {
            "before": "default_reset",
            "after": "hard_reset",
            "chip": chip,
        },
    }
    files = {
        "flasher_args.json": json.dumps(flasher_args).encode(),
        "bootloader/bootloader.bin": b"\x00" * 100,
        "app.bin": b"\xaa" * 200,
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel, data in files.items():
            ti = tarfile.TarInfo(name=rel)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


class FakeProc:
    """Minimal asyncio.subprocess-like stand-in.

    `stdout` is an async iterator yielding pre-scripted byte-lines; `wait()`
    returns the configured return code. Captures the argv it was constructed
    with so tests can assert on the command shape.
    """

    last_argv: list[str] = []

    def __init__(
        self, lines: list[bytes], rc: int = 0, *, delay: float = 0.0
    ) -> None:
        self._lines = lines
        self._rc = rc
        self._delay = delay
        self.returncode: int | None = None

        outer = self

        class _Stdout:
            def __aiter__(self) -> "_Stdout":
                self._idx = 0
                return self

            async def __anext__(self) -> bytes:
                if outer._delay:
                    await asyncio.sleep(outer._delay)
                if self._idx >= len(outer._lines):
                    raise StopAsyncIteration
                line = outer._lines[self._idx]
                self._idx += 1
                return line

        self.stdout = _Stdout()

    async def wait(self) -> int:
        self.returncode = self._rc
        return self._rc

    def terminate(self) -> None:  # pragma: no cover — not exercised in default path
        self.returncode = -15

    def kill(self) -> None:  # pragma: no cover
        self.returncode = -9


def make_subprocess_factory(
    *, lines: list[bytes], rc: int = 0
) -> Any:
    async def _factory(*argv: str, **_kwargs: Any) -> FakeProc:
        FakeProc.last_argv = list(argv)
        return FakeProc(lines, rc)

    return _factory


class FakeWS:
    """Captures every payload pushed via send_json — assertion target."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def send_json(self, channel: str, payload: dict[str, Any]) -> None:
        self.calls.append((channel, payload))

    async def broadcast(
        self, channel: str, payload: str | dict[str, Any]
    ) -> None:  # pragma: no cover — not used by flash
        if isinstance(payload, dict):
            self.calls.append((channel, payload))


# ── build_esptool_cmd — pure function ────────────────────────────────────


def test_build_esptool_cmd_esp32_basic(tmp_path: Path) -> None:
    cmd = build_esptool_cmd(
        target_chip="esp32",
        device_path="/dev/ttyUSB0",
        project_build_dir=tmp_path,
        flash_files=[("0x1000", "bootloader.bin"), ("0x10000", "app.bin")],
        extra_esptool_args={"before": "default_reset", "after": "hard_reset"},
        write_flash_args=["--flash_mode", "dio"],
    )
    # Use a stable interpreter path for the snapshot.
    assert cmd[0] == sys.executable
    assert cmd[1:] == [
        "-m", "esptool",
        "--chip", "esp32",
        "--port", "/dev/ttyUSB0",
        "--before", "default_reset",
        "--after", "hard_reset",
        "write_flash",
        "--flash_mode", "dio",
        "0x1000", str(tmp_path / "bootloader.bin"),
        "0x10000", str(tmp_path / "app.bin"),
    ]


def test_build_esptool_cmd_esp8266_adds_flash_size_detect(tmp_path: Path) -> None:
    cmd = build_esptool_cmd(
        target_chip="esp8266",
        device_path="/dev/ttyUSB0",
        project_build_dir=tmp_path,
        flash_files=[("0x0", "blob.bin")],
    )
    assert "--flash_size" in cmd
    assert cmd[cmd.index("--flash_size") + 1] == "detect"
    # And the final args land after write_flash.
    assert cmd.index("write_flash") < cmd.index("0x0")


def test_build_esptool_cmd_esp8266_respects_user_flash_size(tmp_path: Path) -> None:
    """If user already passed --flash_size we shouldn't double-add 'detect'."""
    cmd = build_esptool_cmd(
        target_chip="esp8266",
        device_path="/dev/ttyUSB0",
        project_build_dir=tmp_path,
        flash_files=[("0x0", "blob.bin")],
        write_flash_args=["--flash_size", "4MB"],
    )
    # exactly one --flash_size arg
    assert cmd.count("--flash_size") == 1
    assert cmd[cmd.index("--flash_size") + 1] == "4MB"


# ── Job persistence ──────────────────────────────────────────────────────


def test_flash_job_round_trip(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    job = FlashJob(
        job_id="abc123",
        alias="agv1",
        project="hello",
        target_chip="esp32",
        started_at=datetime.now(timezone.utc),
        status="running",
        cmd=["python", "-m", "esptool"],
    )
    write_job(tmp_path, job)
    loaded = read_job(tmp_path, "abc123")
    assert loaded.job_id == "abc123"
    assert loaded.alias == "agv1"
    assert loaded.cmd == ["python", "-m", "esptool"]
    rows = list_jobs(tmp_path, limit=10)
    assert len(rows) == 1
    assert rows[0].job_id == "abc123"


# ── End-to-end: trigger_flash with mocked subprocess ─────────────────────


@pytest.mark.asyncio
async def test_trigger_flash_success_streams_lines(
    isolated_data_dir: Path,
) -> None:
    projects_root = isolated_data_dir / "projects"
    upload_project(projects_root, "hello", _build_tarball(), max_size_mb=10)

    fake_ws = FakeWS()
    mon_reg = MonitorRegistry()
    factory = make_subprocess_factory(
        lines=[b"Connecting...\n", b"Writing at 0x1000...\n", b"Hash of data verified.\n"],
        rc=0,
    )
    job_mgr = FlashJobManager()
    job, task = await trigger_flash(
        alias="agv1",
        project="hello",
        target_chip="esp32",
        device_path="/dev/ttyUSB0",
        projects_root=projects_root,
        jobs_dir=isolated_data_dir / "flash_jobs",
        ws_manager=fake_ws,
        monitor_registry=mon_reg,
        subprocess_factory=factory,
        job_manager=job_mgr,
    )
    final = await task

    assert final.status == "success"
    assert final.return_code == 0
    assert final.line_count == 3

    states = [p["state"] for _c, p in fake_ws.calls if p.get("type") == "flash_status"]
    assert "running" in states
    assert "success" in states
    # No monitor was running, so no pause/resume events.
    assert "monitor_paused" not in states
    assert "monitor_resumed" not in states

    lines = [p["text"] for _c, p in fake_ws.calls if p.get("type") == "flash_line"]
    assert lines == ["Connecting...", "Writing at 0x1000...", "Hash of data verified."]

    # Channel naming convention.
    channels = {c for c, _ in fake_ws.calls}
    assert channels == {"flash:agv1"}

    # Persisted to disk.
    persisted = read_job(isolated_data_dir / "flash_jobs", final.job_id)
    assert persisted.status == "success"
    assert persisted.line_count == 3


@pytest.mark.asyncio
async def test_trigger_flash_nonzero_rc_marks_error(
    isolated_data_dir: Path,
) -> None:
    projects_root = isolated_data_dir / "projects"
    upload_project(projects_root, "hello", _build_tarball(), max_size_mb=10)

    fake_ws = FakeWS()
    factory = make_subprocess_factory(
        lines=[b"A fatal error occurred: Failed to connect\n"], rc=2
    )
    job_mgr = FlashJobManager()
    job, task = await trigger_flash(
        alias="agv1",
        project="hello",
        target_chip="esp32",
        device_path="/dev/ttyUSB0",
        projects_root=projects_root,
        jobs_dir=isolated_data_dir / "flash_jobs",
        ws_manager=fake_ws,
        monitor_registry=MonitorRegistry(),
        subprocess_factory=factory,
        job_manager=job_mgr,
    )
    final = await task
    assert final.status == "error"
    assert final.return_code == 2
    err_status = [
        p for _c, p in fake_ws.calls
        if p.get("type") == "flash_status" and p.get("state") == "error"
    ]
    assert err_status and err_status[0]["return_code"] == 2


@pytest.mark.asyncio
async def test_trigger_flash_pauses_and_resumes_monitor(
    isolated_data_dir: Path,
) -> None:
    """A running monitor must be stopped before flash and restarted after."""
    projects_root = isolated_data_dir / "projects"
    upload_project(projects_root, "hello", _build_tarball(), max_size_mb=10)

    mon_reg = MonitorRegistry()

    # Spin up a fake monitor for "agv1".
    def _make_pair() -> tuple[Any, Any]:
        reader = asyncio.StreamReader()

        class W:
            def close(self) -> None:
                pass

        return reader, W()

    async def open_serial(self):  # type: ignore[no-untyped-def]
        return _make_pair()

    SerialMonitor._open_serial = open_serial  # type: ignore[method-assign]
    cfg = MonitorConfig(alias="agv1", device_path="/dev/ttyUSB0", baudrate=115200)
    mon, _ = await mon_reg.start(cfg)
    assert mon.is_running()

    fake_ws = FakeWS()
    factory = make_subprocess_factory(lines=[b"OK\n"], rc=0)
    job_mgr = FlashJobManager()
    job, task = await trigger_flash(
        alias="agv1",
        project="hello",
        target_chip="esp32",
        device_path="/dev/ttyUSB0",
        projects_root=projects_root,
        jobs_dir=isolated_data_dir / "flash_jobs",
        ws_manager=fake_ws,
        monitor_registry=mon_reg,
        subprocess_factory=factory,
        job_manager=job_mgr,
    )
    final = await task
    assert final.status == "success"

    states = [p["state"] for _c, p in fake_ws.calls if p.get("type") == "flash_status"]
    assert "monitor_paused" in states
    assert "monitor_resumed" in states
    # Resume happens AFTER the success status broadcast.
    assert states.index("monitor_resumed") > states.index("success")

    # Monitor is back up under same alias.
    resumed = mon_reg.get("agv1")
    assert resumed is not None and resumed.is_running()

    await mon_reg.stop_all()


@pytest.mark.asyncio
async def test_trigger_flash_resumes_monitor_even_on_error(
    isolated_data_dir: Path,
) -> None:
    """Flash failure must not strand the monitor in a stopped state."""
    projects_root = isolated_data_dir / "projects"
    upload_project(projects_root, "hello", _build_tarball(), max_size_mb=10)

    mon_reg = MonitorRegistry()

    def _make_pair() -> tuple[Any, Any]:
        reader = asyncio.StreamReader()

        class W:
            def close(self) -> None:
                pass

        return reader, W()

    async def open_serial(self):  # type: ignore[no-untyped-def]
        return _make_pair()

    SerialMonitor._open_serial = open_serial  # type: ignore[method-assign]
    cfg = MonitorConfig(alias="agv1", device_path="/dev/ttyUSB0", baudrate=115200)
    await mon_reg.start(cfg)

    fake_ws = FakeWS()
    factory = make_subprocess_factory(lines=[b"err\n"], rc=1)
    job_mgr = FlashJobManager()
    _, task = await trigger_flash(
        alias="agv1", project="hello", target_chip="esp32",
        device_path="/dev/ttyUSB0",
        projects_root=projects_root,
        jobs_dir=isolated_data_dir / "flash_jobs",
        ws_manager=fake_ws,
        monitor_registry=mon_reg,
        subprocess_factory=factory,
        job_manager=job_mgr,
    )
    final = await task
    assert final.status == "error"
    states = [p["state"] for _c, p in fake_ws.calls if p.get("type") == "flash_status"]
    assert "monitor_resumed" in states
    await mon_reg.stop_all()


@pytest.mark.asyncio
async def test_trigger_flash_concurrent_blocked(isolated_data_dir: Path) -> None:
    projects_root = isolated_data_dir / "projects"
    upload_project(projects_root, "hello", _build_tarball(), max_size_mb=10)

    fake_ws = FakeWS()

    # Slow factory holds the subprocess open long enough for a 2nd submit attempt.
    held = asyncio.Event()

    class HeldProc(FakeProc):
        async def wait(self) -> int:
            await held.wait()
            self.returncode = 0
            return 0

    async def slow_factory(*argv: str, **_kwargs: Any) -> HeldProc:
        FakeProc.last_argv = list(argv)
        return HeldProc(lines=[b"Working...\n"])

    job_mgr = FlashJobManager()
    _, task1 = await trigger_flash(
        alias="agv1", project="hello", target_chip="esp32",
        device_path="/dev/ttyUSB0",
        projects_root=projects_root,
        jobs_dir=isolated_data_dir / "flash_jobs",
        ws_manager=fake_ws,
        monitor_registry=MonitorRegistry(),
        subprocess_factory=slow_factory,
        job_manager=job_mgr,
    )
    # Yield so the runner enters its stdout loop before we check.
    await asyncio.sleep(0.05)

    with pytest.raises(FlashConflictError):
        await trigger_flash(
            alias="agv1", project="hello", target_chip="esp32",
            device_path="/dev/ttyUSB0",
            projects_root=projects_root,
            jobs_dir=isolated_data_dir / "flash_jobs",
            ws_manager=fake_ws,
            monitor_registry=MonitorRegistry(),
            subprocess_factory=slow_factory,
            job_manager=job_mgr,
        )

    held.set()
    await task1


# ── REST endpoints ───────────────────────────────────────────────────────


def test_api_flash_unknown_alias_404(isolated_data_dir: Path) -> None:
    from teletop_server.main import app

    with TestClient(app) as client:
        r = client.post("/api/flash/ghost", json={"project": "any"})
        assert r.status_code == 404


def test_api_flash_disconnected_alias_409(isolated_data_dir: Path) -> None:
    register_device("offline", usb_port="9-9", target_chip="esp32")
    from teletop_server.main import app

    with TestClient(app) as client:
        r = client.post("/api/flash/offline", json={"project": "any"})
        assert r.status_code == 409


def test_api_flash_missing_project_404(
    isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_device("agv1", usb_port="3-1", target_chip="esp32")

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

    from teletop_server.main import app

    with TestClient(app) as client:
        r = client.post("/api/flash/agv1", json={"project": "ghost"})
        assert r.status_code == 404


def test_api_flash_happy_path_returns_202(
    isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full REST happy path with subprocess swapped for the fake."""
    register_device("agv1", usb_port="3-1", target_chip="esp32")

    from teletop_server import devices as devices_mod
    from teletop_server import flash as flash_mod
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

    # Patch FlashRunner's default subprocess factory so REST path uses fake.
    factory = make_subprocess_factory(lines=[b"OK\n"], rc=0)
    real_init = flash_mod.FlashRunner.__init__

    def patched_init(self, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["subprocess_factory"] = factory
        real_init(self, **kwargs)

    monkeypatch.setattr(flash_mod.FlashRunner, "__init__", patched_init)

    from teletop_server.main import app

    with TestClient(app) as client:
        # First upload a project via REST.
        r0 = client.post(
            "/api/projects/hello/upload",
            files={"archive": ("hello.tar.gz", _build_tarball(), "application/gzip")},
        )
        assert r0.status_code == 200, r0.text

        r = client.post("/api/flash/agv1", json={"project": "hello"})
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["alias"] == "agv1"
        assert body["project"] == "hello"
        assert body["target_chip"] == "esp32"
        job_id = body["job_id"]

        # Wait for the background task to finish.
        async def wait_done() -> None:
            for _ in range(200):
                if not flash_mod.manager.is_active("agv1"):
                    return
                await asyncio.sleep(0.01)
            raise AssertionError("flash never finished")

        asyncio.run(wait_done())

        rj = client.get(f"/api/flash/jobs/{job_id}")
        assert rj.status_code == 200
        assert rj.json()["status"] == "success"

        rl = client.get("/api/flash/jobs")
        assert rl.status_code == 200
        assert any(j["job_id"] == job_id for j in rl.json())


def test_ws_flash_emits_initial_status(isolated_data_dir: Path) -> None:
    from teletop_server.main import app

    with TestClient(app) as client:
        with client.websocket_connect("/ws/flash/ghost") as ws:
            evt = ws.receive_json()
            assert evt["type"] == "flash_status"
            assert evt["alias"] == "ghost"
            assert evt["state"] == "idle"


# ── CLI smoke ────────────────────────────────────────────────────────────


def test_cli_flash_subgroup_help() -> None:
    from teletop_server.main import cli

    result = CliRunner().invoke(cli, ["flash", "--help"])
    assert result.exit_code == 0, result.output
    for sub in ("run", "jobs", "show", "tail"):
        assert sub in result.output


def test_cli_flash_jobs_complains_when_server_down() -> None:
    import os

    from teletop_server.main import cli

    os.environ["TELETOP_PORT"] = "59997"
    try:
        result = CliRunner().invoke(cli, ["flash", "jobs"])
    finally:
        del os.environ["TELETOP_PORT"]
    assert result.exit_code != 0
    assert "could not reach server" in result.output.lower()
