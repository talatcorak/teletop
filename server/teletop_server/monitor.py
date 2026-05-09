"""Serial monitor — UART → WebSocket fan-out + rotated file logging.

Each connected device gets one ``SerialMonitor``. The reader is fully async
(via pyserial-asyncio's ``StreamReader``); a bounded queue decouples the
serial side from broadcast/log so a slow client cannot stall the read loop.
The ``MonitorRegistry`` singleton tracks live monitors, dedupes by alias,
and exposes a ``stop_all`` hook the FastAPI lifespan calls on shutdown.

Channel name convention: ``monitor:<alias>``. Every event is a JSON object
shaped as either ``{type: "line", ...}`` or ``{type: "status", ...}`` —
clients can treat the channel as a single union-typed stream.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import serial_asyncio
from serial import SerialException

from .ws import manager

logger = logging.getLogger("teletop.monitor")


# ── Config ───────────────────────────────────────────────────────────────


@dataclass
class MonitorConfig:
    alias: str
    device_path: str
    baudrate: int = 115200
    log_dir: Path | None = None
    queue_size: int = 1000


def channel_for(alias: str) -> str:
    return f"monitor:{alias}"


# ── Helpers ──────────────────────────────────────────────────────────────


def _utc_iso(now: datetime | None = None) -> str:
    """RFC3339-ish UTC timestamp, e.g. 2026-05-09T06:45:12.345Z."""
    dt = now or datetime.now(timezone.utc)
    # millisecond precision is plenty for log-line timestamps.
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _log_filename(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S") + ".log"


# ── SerialMonitor ────────────────────────────────────────────────────────


BroadcastFn = Callable[[str, dict[str, Any]], Awaitable[None]]


class SerialMonitor:
    """Reads one serial port; broadcasts each line + writes to a log file.

    Two coroutines run while ``is_running()``:
      • reader_loop: pulls bytes from pyserial-asyncio, splits on '\\n',
        enqueues decoded text.
      • consumer_loop: drains the queue, broadcasts each line as a JSON
        event, then appends it to the active log file.

    The queue (default 1000) provides bounded backpressure — when full we
    drop the *oldest* line so the most recent firmware output always wins.
    Dropping oldest matches operator intuition for live monitoring: stale
    heartbeats are less interesting than the line that just printed.
    """

    def __init__(
        self,
        cfg: MonitorConfig,
        *,
        broadcast: BroadcastFn | None = None,
    ) -> None:
        self.cfg = cfg
        self._broadcast: BroadcastFn = broadcast or manager.broadcast
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._queue: asyncio.Queue[str] | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._consumer_task: asyncio.Task[None] | None = None
        self._log_file: Any = None
        self._log_path: Path | None = None
        self._running = False
        self._started_at: datetime | None = None
        self._dropped = 0

    # ── public ────────────────────────────────────────────────────────────

    @property
    def started_at(self) -> datetime | None:
        return self._started_at

    @property
    def log_path(self) -> Path | None:
        return self._log_path

    @property
    def dropped(self) -> int:
        return self._dropped

    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        if self._running:
            return
        try:
            self._reader, self._writer = await self._open_serial()
        except (OSError, SerialException) as exc:
            await self._broadcast_status("error", error=str(exc))
            raise

        self._open_log_file()
        self._queue = asyncio.Queue(maxsize=self.cfg.queue_size)
        self._running = True
        self._started_at = datetime.now(timezone.utc)
        self._reader_task = asyncio.create_task(
            self._reader_loop(), name=f"monitor-read-{self.cfg.alias}"
        )
        self._consumer_task = asyncio.create_task(
            self._consumer_loop(), name=f"monitor-consume-{self.cfg.alias}"
        )
        await self._broadcast_status("connected")
        logger.info(
            "monitor[%s] started device=%s baud=%d log=%s",
            self.cfg.alias,
            self.cfg.device_path,
            self.cfg.baudrate,
            self._log_path,
        )

    async def stop(self) -> None:
        if self._reader_task is None and self._consumer_task is None and not self._running:
            return
        self._running = False
        for task in (self._reader_task, self._consumer_task):
            if task is not None and not task.done():
                task.cancel()
        for task in (self._reader_task, self._consumer_task):
            if task is None:
                continue
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning("monitor[%s] task ended with %r", self.cfg.alias, exc)
        self._reader_task = None
        self._consumer_task = None

        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # pragma: no cover — best effort
                pass
            self._writer = None
        self._reader = None

        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:  # pragma: no cover
                pass
            self._log_file = None

        await self._broadcast_status("stopped")
        logger.info("monitor[%s] stopped", self.cfg.alias)

    # ── internals ─────────────────────────────────────────────────────────

    async def _open_serial(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Wrapper kept as a method so tests can monkeypatch one entry point."""
        return await serial_asyncio.open_serial_connection(
            url=self.cfg.device_path, baudrate=self.cfg.baudrate
        )

    def _open_log_file(self) -> None:
        if self.cfg.log_dir is None:
            return
        target_dir = self.cfg.log_dir / self.cfg.alias
        target_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = target_dir / _log_filename()
        # Append mode means a re-start in the same second appends rather
        # than truncates; in practice the timestamp suffix prevents collisions.
        self._log_file = self._log_path.open("a", encoding="utf-8", buffering=1)

    async def _reader_loop(self) -> None:
        assert self._reader is not None
        try:
            while True:
                # readline() returns up to and including the next '\n', or
                # b"" on EOF (device closed / unplugged on most kernels).
                line = await self._reader.readline()
                if not line:
                    await self._broadcast_status(
                        "disconnected", reason="device closed (EOF)"
                    )
                    return
                text = line.decode("utf-8", errors="replace").rstrip("\r\n")
                self._enqueue(text)
        except asyncio.CancelledError:
            raise
        except (OSError, SerialException) as exc:
            # Unplug usually surfaces as OSError(EIO) from the underlying
            # transport. Surface it to clients and let stop()/the registry
            # handle cleanup of file/queue resources.
            await self._broadcast_status("disconnected", reason=str(exc))
        except Exception as exc:  # pragma: no cover — defensive
            logger.exception("monitor[%s] reader crashed", self.cfg.alias)
            await self._broadcast_status("error", error=repr(exc))
        finally:
            self._running = False

    def _enqueue(self, line: str) -> None:
        assert self._queue is not None
        try:
            self._queue.put_nowait(line)
            return
        except asyncio.QueueFull:
            pass
        # Backpressure — drop the oldest line to keep the freshest data.
        try:
            self._queue.get_nowait()
            self._queue.task_done()
        except asyncio.QueueEmpty:  # pragma: no cover — race
            pass
        self._dropped += 1
        # Log on first drop, then every 100, so a sustained flood doesn't
        # spam the log itself while still surfacing the issue.
        if self._dropped == 1 or self._dropped % 100 == 0:
            logger.warning(
                "monitor[%s] queue full — dropped %d lines",
                self.cfg.alias,
                self._dropped,
            )
        try:
            self._queue.put_nowait(line)
        except asyncio.QueueFull:  # pragma: no cover — drained then full again
            pass

    async def _consumer_loop(self) -> None:
        assert self._queue is not None
        channel = channel_for(self.cfg.alias)
        try:
            while True:
                line = await self._queue.get()
                payload = {
                    "type": "line",
                    "alias": self.cfg.alias,
                    "ts": _utc_iso(),
                    "text": line,
                }
                try:
                    await self._broadcast(channel, payload)
                except Exception as exc:  # pragma: no cover — manager swallows internally
                    logger.warning(
                        "monitor[%s] broadcast failed: %r", self.cfg.alias, exc
                    )
                if self._log_file is not None:
                    try:
                        self._log_file.write(line + "\n")
                        # buffering=1 (line buffering) flushes on each newline,
                        # but the explicit flush guarantees the kernel sees it
                        # before the next event so a crash doesn't lose the
                        # most recent line.
                        self._log_file.flush()
                    except Exception as exc:  # pragma: no cover
                        logger.warning(
                            "monitor[%s] log write failed: %r",
                            self.cfg.alias,
                            exc,
                        )
                self._queue.task_done()
        except asyncio.CancelledError:
            raise

    async def _broadcast_status(self, state: str, **extra: Any) -> None:
        payload: dict[str, Any] = {
            "type": "status",
            "alias": self.cfg.alias,
            "state": state,
        }
        payload.update(extra)
        try:
            await self._broadcast(channel_for(self.cfg.alias), payload)
        except Exception as exc:  # pragma: no cover
            logger.debug(
                "monitor[%s] status broadcast failed: %r", self.cfg.alias, exc
            )


# ── Registry ─────────────────────────────────────────────────────────────


class MonitorRegistry:
    """Tracks live monitors by alias. Idempotent start; cooperative stop_all."""

    def __init__(self) -> None:
        self._monitors: dict[str, SerialMonitor] = {}
        self._lock = asyncio.Lock()

    async def start(self, cfg: MonitorConfig) -> tuple[SerialMonitor, bool]:
        """Start (or return) a monitor for ``cfg.alias``.

        Returns ``(monitor, already_running)`` so callers can communicate
        idempotency to the API layer.
        """
        async with self._lock:
            existing = self._monitors.get(cfg.alias)
            if existing is not None and existing.is_running():
                return existing, True
            # Replace stale entries (a monitor whose reader exited on its
            # own would still be in the dict but no longer running).
            monitor = SerialMonitor(cfg)
            self._monitors[cfg.alias] = monitor
        # start() outside the lock — opening the serial port could block.
        try:
            await monitor.start()
        except Exception:
            async with self._lock:
                # Don't leak a half-initialised entry.
                if self._monitors.get(cfg.alias) is monitor:
                    del self._monitors[cfg.alias]
            raise
        return monitor, False

    async def stop(self, alias: str) -> bool:
        """Stop and forget ``alias``. Returns False if there was nothing to stop."""
        async with self._lock:
            mon = self._monitors.pop(alias, None)
        if mon is None:
            return False
        await mon.stop()
        return True

    async def stop_all(self) -> None:
        async with self._lock:
            mons = list(self._monitors.values())
            self._monitors.clear()
        if not mons:
            return
        await asyncio.gather(*(m.stop() for m in mons), return_exceptions=True)

    def list_active(self) -> list[str]:
        return sorted(a for a, m in self._monitors.items() if m.is_running())

    def all(self) -> dict[str, SerialMonitor]:
        return dict(self._monitors)

    def get(self, alias: str) -> SerialMonitor | None:
        return self._monitors.get(alias)


# Module-level singleton, parallel to ws.manager.
registry = MonitorRegistry()


# ── Log file helpers (used by REST log endpoints) ────────────────────────


class LogPathError(ValueError):
    """Raised when a requested log path tries to escape the alias directory."""


def alias_log_dir(log_root: Path, alias: str) -> Path:
    return log_root / alias


def list_log_files(log_root: Path, alias: str) -> list[dict[str, Any]]:
    """Return each .log file in the alias dir with its size + mtime."""
    d = alias_log_dir(log_root, alias)
    if not d.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(d.iterdir()):
        if not p.is_file() or p.suffix != ".log":
            continue
        st = p.stat()
        out.append(
            {
                "filename": p.name,
                "size": st.st_size,
                "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            }
        )
    return out


def resolve_log_file(log_root: Path, alias: str, filename: str) -> Path:
    """Resolve <log_root>/<alias>/<filename> rejecting path traversal.

    Filename must be a bare name (no slashes, no '..'); the resolved path
    must lie inside the alias directory. Symlinks pointing outside are
    rejected by comparing against the resolved alias dir.
    """
    if "/" in filename or "\\" in filename or filename in ("", ".", ".."):
        raise LogPathError(f"invalid filename: {filename!r}")
    base = alias_log_dir(log_root, alias).resolve()
    candidate = (base / filename).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise LogPathError(f"path escapes alias dir: {filename!r}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(filename)
    return candidate


__all__ = [
    "BroadcastFn",
    "LogPathError",
    "MonitorConfig",
    "MonitorRegistry",
    "SerialMonitor",
    "alias_log_dir",
    "channel_for",
    "list_log_files",
    "registry",
    "resolve_log_file",
]
