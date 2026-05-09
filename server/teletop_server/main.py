"""FastAPI application entry point + click CLI."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import AsyncIterator, Callable

import click
from fastapi import FastAPI, HTTPException, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from . import __version__ as fallback_version
from .config import Settings, get_settings
from .devices import (
    DEFAULT_TARGET_CHIP,
    TARGET_CHIPS,
    UDEV_RULES_PATH,
    DeviceRegistration,
    DeviceRegistryError,
    DeviceStatus,
    DiscoveredPort,
    DiscoveredPortWithRegistration,
    TargetChip,
    detect_target_chip,
    discover_ports,
    discover_with_registrations,
    generate_udev_rules,
    get_device_status,
    install_udev_rules,
    load_registry,
    match_registration,
    register_device,
    symlink_for,
    uninstall_udev_rules,
    unregister_device,
    update_device,
)
from .ws import WebSocketDisconnect, manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("teletop")


def _resolve_version() -> str:
    try:
        return version("teletop-server")
    except PackageNotFoundError:
        return fallback_version


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.started_at = time.monotonic()
    app.state.settings = settings
    app.state.version = _resolve_version()
    logger.info(
        "teletop starting host=%s port=%d data_dir=%s web_dist=%s",
        settings.host,
        settings.port,
        settings.data_dir,
        settings.web_dist_dir,
    )
    yield
    logger.info("teletop shutting down")


# ── API request body models ──────────────────────────────────────────────


class DeviceCreate(BaseModel):
    alias: str
    target_chip: TargetChip
    serial_number: str | None = None
    usb_port: str | None = None
    notes: str | None = None
    vid: int | None = None
    pid: int | None = None
    usb_chip: str | None = None


class DeviceUpdate(BaseModel):
    """PATCH body — only target_chip and notes are mutable here."""

    target_chip: TargetChip | None = None
    notes: str | None = None


# ── App factory ──────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(title="teletop", version=_resolve_version(), lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Task 12: restrict to tailnet.
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        settings = app.state.settings
        return {
            "status": "ok",
            "service": "teletop",
            "version": app.state.version,
            "uptime_seconds": int(time.monotonic() - app.state.started_at),
            "data_dir": str(settings.data_dir),
            "ws_channels": len(manager.channels),
        }

    @app.get("/api/devices")
    async def api_list_devices() -> list[DeviceStatus]:
        return get_device_status()

    @app.get("/api/devices/discover")
    async def api_discover(
        include_all: bool = False,
    ) -> list[DiscoveredPortWithRegistration]:
        return discover_with_registrations(esp_only=not include_all)

    @app.post("/api/devices", status_code=201)
    async def api_register(body: DeviceCreate) -> DeviceRegistration:
        try:
            return register_device(
                body.alias,
                target_chip=body.target_chip,
                serial_number=body.serial_number,
                usb_port=body.usb_port,
                vid=body.vid,
                pid=body.pid,
                usb_chip=body.usb_chip,
                notes=body.notes,
            )
        except DeviceRegistryError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.patch("/api/devices/{alias}")
    async def api_update(alias: str, body: DeviceUpdate) -> DeviceRegistration:
        try:
            return update_device(
                alias, target_chip=body.target_chip, notes=body.notes
            )
        except KeyError:
            raise HTTPException(status_code=404, detail=f"alias {alias!r} not found")

    @app.delete("/api/devices/{alias}", status_code=204)
    async def api_unregister(alias: str) -> Response:
        try:
            unregister_device(alias)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"alias {alias!r} not found")
        return Response(status_code=204)

    @app.websocket("/ws/test/{channel}")
    async def ws_test(websocket: WebSocket, channel: str) -> None:
        await manager.connect(websocket, channel)
        try:
            while True:
                msg = await websocket.receive_text()
                await manager.broadcast(channel, f"echo[{channel}]: {msg}")
        except WebSocketDisconnect:
            await manager.disconnect(websocket, channel)

    # Static SPA mount must come AFTER all API/WS routes — it's a catch-all.
    settings = get_settings()
    if settings.web_dist_dir.exists():
        app.mount("/", StaticFiles(directory=settings.web_dist_dir, html=True), name="web")
        logger.info("mounted static SPA from %s", settings.web_dist_dir)
    else:
        logger.info("web_dist_dir %s missing — running API-only", settings.web_dist_dir)

    return app


app = create_app()


# ── CLI ──────────────────────────────────────────────────────────────────


def _vid_pid(vid: int | None, pid: int | None) -> str:
    if vid is None or pid is None:
        return "—"
    return f"{vid:04X}:{pid:04X}"


def _identity_str(port: DiscoveredPort) -> str:
    if port.serial_number:
        return f"serial={port.serial_number}"
    if port.usb_port:
        return f"port={port.usb_port}"
    return "—"


@click.group(invoke_without_command=True, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=_resolve_version(), prog_name="teletop-server")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """teletop server — ESP32 flash & monitor over the network."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(serve)


@cli.command()
def serve() -> None:
    """Run the FastAPI server (default action when no subcommand is given)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "teletop_server.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


@cli.command()
@click.option("--all", "show_all", is_flag=True, help="Include non-ESP32 vendors.")
def discover(show_all: bool) -> None:
    """List currently connected serial ports."""
    console = Console()
    ports = discover_ports(esp_only=not show_all)
    registry = load_registry()

    if not ports:
        console.print(
            "[yellow]No serial ports found[/yellow]"
            + ("" if show_all else " — try [bold]--all[/bold] to widen the vendor filter.")
        )
        return

    table = Table(title="Discovered serial ports")
    table.add_column("device")
    table.add_column("VID:PID")
    table.add_column("usb_chip")
    table.add_column("identity")
    table.add_column("status")
    table.add_column("description", overflow="fold")

    for port in ports:
        reg = match_registration(port, registry)
        status = (
            f"[green]{reg.alias} ({reg.target_chip})[/green]"
            if reg
            else "[dim]free[/dim]"
        )
        table.add_row(
            port.device,
            _vid_pid(port.vid, port.pid),
            port.usb_chip or "—",
            _identity_str(port),
            status,
            port.description or "",
        )

    console.print(table)


def _parse_hex_int(ctx: click.Context, param: click.Parameter, value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value, 0)  # accepts "0x1A86", "1A86", "6790"
    except ValueError:
        raise click.BadParameter(f"{value!r} is not a valid integer")


def _resolve_target_chip(
    *,
    target: str | None,
    detect: bool,
    device_path: str | None,
    interactive: bool,
    console: Console,
) -> TargetChip:
    """Decide which TargetChip to record. Exits with 1 on unrecoverable errors."""
    if target:
        return target  # type: ignore[return-value]
    if detect:
        if not device_path:
            console.print(
                "[red]error:[/red] --detect needs a connected device matching the identity"
            )
            sys.exit(1)
        try:
            chip = detect_target_chip(device_path)
        except (subprocess.TimeoutExpired, RuntimeError, FileNotFoundError) as exc:
            console.print(f"[yellow]auto-detect failed:[/yellow] {exc}")
            if not interactive:
                sys.exit(1)
        else:
            console.print(f"[cyan]detected target:[/cyan] {chip}")
            return chip
    if interactive:
        return click.prompt(
            "Target chip",
            type=click.Choice(TARGET_CHIPS),
            default=DEFAULT_TARGET_CHIP,
        )
    console.print(
        "[red]error:[/red] one of --target / --detect is required for non-interactive register"
    )
    sys.exit(1)


@cli.command()
@click.argument("alias")
@click.option(
    "--target",
    type=click.Choice(TARGET_CHIPS),
    default=None,
    help="Target ESP chip family (e.g. esp32, esp8266, esp32s3).",
)
@click.option(
    "--detect",
    is_flag=True,
    help="Auto-detect target via `esptool.py chip_id` (briefly bounces device into bootloader).",
)
@click.option("--serial", "serial_number", default=None, help="Pin to USB serial number.")
@click.option("--port", "usb_port", default=None, help="Pin to USB port path (e.g. 3-1).")
@click.option(
    "--vid", "vid", default=None, callback=_parse_hex_int,
    help="USB vendor ID (hex, e.g. 0x1A86). Required for udev rule generation.",
)
@click.option(
    "--pid", "pid", default=None, callback=_parse_hex_int,
    help="USB product ID (hex, e.g. 0x7523). Required for udev rule generation.",
)
@click.option("--note", "notes", default=None, help="Free-form note.")
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    help="Include non-ESP32 vendors when listing for selection.",
)
def register(
    alias: str,
    target: str | None,
    detect: bool,
    serial_number: str | None,
    usb_port: str | None,
    vid: int | None,
    pid: int | None,
    notes: str | None,
    show_all: bool,
) -> None:
    """Register a device under ALIAS.

    Without --serial/--port, the command lists currently connected ports and
    prompts you to pick one. The chosen port's identity is recorded — serial
    number when available (e.g. CP210x boards), otherwise the USB port path
    (CH340 boards). The target ESP chip family is captured separately:
    pass --target, --detect, or answer the interactive prompt.
    """
    console = Console()

    # ── Non-interactive flag path ────────────────────────────────────────
    if serial_number or usb_port:
        device_path: str | None = None
        if detect:
            ports = discover_ports(esp_only=not show_all)
            match = next(
                (
                    p
                    for p in ports
                    if (serial_number and p.serial_number == serial_number)
                    or (not serial_number and p.usb_port == usb_port)
                ),
                None,
            )
            device_path = match.device if match else None
        target_chip = _resolve_target_chip(
            target=target,
            detect=detect,
            device_path=device_path,
            interactive=False,
            console=console,
        )
        try:
            reg = register_device(
                alias,
                target_chip=target_chip,
                serial_number=serial_number,
                usb_port=usb_port,
                vid=vid,
                pid=pid,
                notes=notes,
            )
        except DeviceRegistryError as exc:
            console.print(f"[red]error:[/red] {exc}")
            sys.exit(1)
        identity = (
            f"serial={reg.serial_number}"
            if reg.serial_number
            else f"usb_port={reg.usb_port or '?'}"
        )
        console.print(
            f"[green]registered[/green] {alias} ({reg.target_chip}, "
            f"usb={reg.usb_chip or 'unknown'}) via {identity}"
        )
        if reg.vid is None or reg.pid is None:
            console.print(
                "[yellow]warning:[/yellow] no VID/PID — udev rule generation "
                "will skip this entry. Pass --vid/--pid or re-register interactively."
            )
        _maybe_offer_udev_reinstall(console)
        return

    # ── Interactive path ─────────────────────────────────────────────────
    ports = discover_ports(esp_only=not show_all)
    if not ports:
        console.print("[red]No ports detected.[/red] Plug in a device or use --all.")
        sys.exit(1)

    console.print("[bold]Connected ports:[/bold]")
    for i, port in enumerate(ports):
        console.print(
            f"  [{i}] {port.device}  {_vid_pid(port.vid, port.pid)}  "
            f"usb_chip={port.usb_chip or '—'}  {_identity_str(port)}"
        )

    idx = click.prompt("Select port", type=click.IntRange(0, len(ports) - 1))
    chosen = ports[idx]

    if chosen.serial_number:
        console.print(
            f"[cyan]→ recording by serial number[/cyan] ({chosen.serial_number}) — "
            "stable across USB ports."
        )
    else:
        console.print(
            f"[yellow]→ recording by USB port path[/yellow] ({chosen.usb_port}) — "
            "this chip exposes no serial; keep the device in this physical socket."
        )

    target_chip = _resolve_target_chip(
        target=target,
        detect=detect,
        device_path=chosen.device,
        interactive=True,
        console=console,
    )

    try:
        reg = register_device(alias, port=chosen, target_chip=target_chip, notes=notes)
    except DeviceRegistryError as exc:
        console.print(f"[red]error:[/red] {exc}")
        sys.exit(1)

    console.print(
        f"[green]registered[/green] {alias} "
        f"(target={reg.target_chip}, usb={reg.usb_chip or 'unknown'})"
    )
    _maybe_offer_udev_reinstall(console)


@cli.command()
@click.argument("alias")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def unregister(alias: str, yes: bool) -> None:
    """Remove ALIAS from the registry."""
    console = Console()
    if not yes and not click.confirm(f"Unregister {alias!r}?", default=False):
        console.print("[yellow]aborted[/yellow]")
        return
    try:
        unregister_device(alias)
    except KeyError:
        console.print(f"[red]error:[/red] alias {alias!r} not registered")
        sys.exit(1)
    console.print(f"[green]unregistered[/green] {alias}")
    _maybe_offer_udev_reinstall(console)


@cli.command(name="list")
def list_cmd() -> None:
    """Show registered devices and their connection state."""
    console = Console()
    statuses = get_device_status()
    if not statuses:
        console.print("[dim]No devices registered yet — run[/dim] [bold]register <alias>[/bold]")
        return

    table = Table(title="Registered devices")
    table.add_column("alias")
    table.add_column("identity")
    table.add_column("target")
    table.add_column("usb_chip")
    table.add_column("connected")
    table.add_column("current device")
    table.add_column("symlink")

    for s in statuses:
        connected = "[green]✓[/green]" if s.connected else "[red]✗[/red]"
        identity = f"{s.identity_type}={s.identity_value}"
        table.add_row(
            s.alias,
            identity,
            s.target_chip,
            s.usb_chip or "—",
            connected,
            s.current_device or "—",
            s.udev_symlink or "—",
        )

    console.print(table)


@cli.command(name="set-target")
@click.argument("alias")
@click.argument("target", type=click.Choice(TARGET_CHIPS))
def set_target_cmd(alias: str, target: str) -> None:
    """Update the target ESP chip for an already-registered ALIAS."""
    console = Console()
    try:
        reg = update_device(alias, target_chip=target)  # type: ignore[arg-type]
    except KeyError:
        console.print(f"[red]error:[/red] alias {alias!r} not registered")
        sys.exit(1)
    console.print(f"[green]updated[/green] {alias} target_chip → {reg.target_chip}")
    _maybe_offer_udev_reinstall(console)


# ── udev commands ────────────────────────────────────────────────────────


def resolve_data_dir(settings: Settings) -> Path:
    """Pick the right data_dir under sudo.

    Settings.data_dir defaults to ``Path.home() / "teletop"`` evaluated at
    class-definition time, so a process started by sudo (HOME=/root) looks
    in /root/teletop and misses the registry that lives in the invoking
    user's home. When TELETOP_DATA_DIR is unset and we're root with a
    populated SUDO_USER, fall back to that user's home.
    """
    if os.environ.get("TELETOP_DATA_DIR"):
        return settings.data_dir
    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user and os.geteuid() == 0:
        try:
            import pwd

            pw = pwd.getpwnam(sudo_user)
            return Path(pw.pw_dir) / "teletop"
        except (KeyError, ImportError):
            pass
    return settings.data_dir


def _apply_data_dir_resolution() -> Callable[[], None]:
    """Apply the SUDO_USER fallback if needed; return a teardown callable.

    The handler should call the returned teardown in a finally block so
    pytest's CliRunner (same process) doesn't leak a TELETOP_DATA_DIR
    override across tests.
    """
    settings = get_settings()
    resolved = resolve_data_dir(settings)
    if resolved == settings.data_dir:
        return lambda: None
    prev = os.environ.get("TELETOP_DATA_DIR")
    os.environ["TELETOP_DATA_DIR"] = str(resolved)
    logger.info(
        "reading registry from %s (resolved via SUDO_USER=%s)",
        resolved / "devices.json",
        os.environ.get("SUDO_USER"),
    )

    def _restore() -> None:
        if prev is None:
            os.environ.pop("TELETOP_DATA_DIR", None)
        else:
            os.environ["TELETOP_DATA_DIR"] = prev

    return _restore


def _maybe_offer_udev_reinstall(console: Console) -> None:
    """If teletop udev rules are already installed, offer to refresh them."""
    if not UDEV_RULES_PATH.exists():
        return
    if not click.confirm("Re-install udev rules now?", default=True):
        console.print(
            "[dim]hint:[/dim] run "
            "[bold]sudo $(which uv) run teletop-server udev-install[/bold] later"
        )
        return
    try:
        install_udev_rules()
    except PermissionError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
    except Exception as exc:  # pragma: no cover — best-effort surface
        console.print(f"[red]udev reload failed:[/red] {exc}")
    else:
        console.print(f"[green]reloaded[/green] {UDEV_RULES_PATH}")


@cli.command(name="udev-rules")
def udev_rules_cmd() -> None:
    """Print the udev rules file derived from the current registry."""
    click.echo(generate_udev_rules(), nl=False)


@cli.command(name="udev-install")
def udev_install_cmd() -> None:
    """Write the rules file to /etc/udev/rules.d and reload udev (root only)."""
    console = Console()
    restore = _apply_data_dir_resolution()
    try:
        try:
            path = install_udev_rules()
        except PermissionError as exc:
            console.print(f"[red]error:[/red] {exc}")
            sys.exit(1)
        except subprocess.CalledProcessError as exc:
            console.print(f"[red]udevadm failed:[/red] {exc}")
            sys.exit(1)

        statuses = get_device_status()
        console.print(f"[green]installed[/green] {path}")
        if not statuses:
            console.print("[dim]no devices registered yet — nothing to symlink[/dim]")
            return
        table = Table(title="Symlinks after reload")
        table.add_column("alias")
        table.add_column("symlink")
        table.add_column("identity")
        for s in statuses:
            sym = symlink_for(s.alias)
            table.add_row(s.alias, str(sym), f"{s.identity_type}={s.identity_value}")
        console.print(table)
    finally:
        restore()


@cli.command(name="udev-uninstall")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def udev_uninstall_cmd(yes: bool) -> None:
    """Remove the rules file and reload udev (root only)."""
    console = Console()
    restore = _apply_data_dir_resolution()
    try:
        if not yes and not click.confirm(
            f"Remove {UDEV_RULES_PATH}?", default=False
        ):
            console.print("[yellow]aborted[/yellow]")
            return
        try:
            removed = uninstall_udev_rules()
        except PermissionError as exc:
            console.print(f"[red]error:[/red] {exc}")
            sys.exit(1)
        except subprocess.CalledProcessError as exc:
            console.print(f"[red]udevadm failed:[/red] {exc}")
            sys.exit(1)
        if removed:
            console.print(f"[green]removed[/green] {UDEV_RULES_PATH}")
        else:
            console.print(f"[dim]no rules file at[/dim] {UDEV_RULES_PATH}")
    finally:
        restore()


if __name__ == "__main__":
    cli()
