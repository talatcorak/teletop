"""esptool wrapper — flash an uploaded project to a registered alias.

The flash subsystem owns:

* **FlashJob**: persisted record of one flash attempt, including return code
  and elapsed time. Lives at ``<flash_jobs_dir>/<job_id>.json``.
* **build_esptool_cmd**: pure function that converts a project's
  ``flasher_args.json`` shape + a target chip + a device path into a
  ``python -m esptool ...`` argv list. Pure ⇒ snapshot-testable.
* **FlashRunner**: orchestrates one flash — pauses any running monitor,
  spawns esptool as a subprocess, streams its merged stdout/stderr to the
  ``flash:<alias>`` channel, and resumes the monitor afterwards (success or
  failure — the operator wants their console back either way).
* **FlashJobManager**: process-wide singleton. Tracks active jobs to
  enforce "one flash per alias at a time", and keeps a recent-history
  snapshot for the REST + CLI listing endpoints.

Concurrency: the manager serialises *per alias* (a second flash on the same
alias while one is running → 409). Different aliases flash in parallel —
no hardware conflict, and the queue would otherwise punish multi-device
fleets.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .monitor import MonitorConfig, MonitorRegistry
from .monitor import registry as default_monitor_registry
from .projects import ProjectMeta, build_dir, load_project_meta
from .ws import manager as default_ws_manager

logger = logging.getLogger("teletop.flash")


FlashStatus = Literal["pending", "running", "success", "error"]


# ── Job model ────────────────────────────────────────────────────────────


@dataclass
class FlashJob:
    job_id: str
    alias: str
    project: str
    target_chip: str
    started_at: datetime
    status: FlashStatus = "pending"
    finished_at: datetime | None = None
    return_code: int | None = None
    error: str | None = None
    line_count: int = 0
    cmd: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["started_at"] = self.started_at.isoformat().replace("+00:00", "Z")
        if self.finished_at is not None:
            d["finished_at"] = self.finished_at.isoformat().replace("+00:00", "Z")
        return d

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "FlashJob":
        def _dt(val: Any) -> datetime | None:
            if val is None:
                return None
            return datetime.fromisoformat(val.replace("Z", "+00:00"))

        return cls(
            job_id=data["job_id"],
            alias=data["alias"],
            project=data["project"],
            target_chip=data.get("target_chip", "esp32"),
            started_at=_dt(data["started_at"]) or datetime.now(timezone.utc),
            status=data.get("status", "error"),
            finished_at=_dt(data.get("finished_at")),
            return_code=data.get("return_code"),
            error=data.get("error"),
            line_count=data.get("line_count", 0),
            cmd=data.get("cmd", []),
        )


def channel_for(alias: str) -> str:
    return f"flash:{alias}"


# ── esptool argv builder ─────────────────────────────────────────────────


def build_esptool_cmd(
    *,
    target_chip: str,
    device_path: str,
    project_build_dir: Path,
    flash_files: list[tuple[str, str]],
    extra_esptool_args: dict[str, Any] | None = None,
    write_flash_args: list[str] | None = None,
) -> list[str]:
    """Build the ``python -m esptool ...`` argv list for one write.

    * Module form (``-m esptool``) over the script (``esptool.py``) so we
      pin to the venv's interpreter and avoid PATH resolution surprises in
      sudo / systemd contexts.
    * ``flash_files`` arrives as an ordered list of ``(offset, rel_path)``
      so sort order is the caller's responsibility — preserves the exact
      ordering esptool itself recommends (bootloader, partition table, app).
    """
    cmd: list[str] = [
        sys.executable, "-m", "esptool",
        "--chip", target_chip,
        "--port", device_path,
    ]

    extras = extra_esptool_args or {}
    before = extras.get("before")
    after = extras.get("after")
    if isinstance(before, str) and before:
        cmd += ["--before", before]
    if isinstance(after, str) and after:
        cmd += ["--after", after]

    cmd.append("write_flash")

    write_args = list(write_flash_args or [])
    cmd += write_args

    # ESP8266 needs --flash_size; default to detect when not specified so
    # builds with a minimal flasher_args.json still succeed.
    if target_chip == "esp8266" and "--flash_size" not in write_args:
        cmd += ["--flash_size", "detect"]

    for offset, rel_path in flash_files:
        cmd += [offset, str(project_build_dir / rel_path)]

    return cmd


# ── Job persistence ──────────────────────────────────────────────────────


def _job_path(jobs_dir: Path, job_id: str) -> Path:
    return jobs_dir / f"{job_id}.json"


def write_job(jobs_dir: Path, job: FlashJob) -> None:
    jobs_dir.mkdir(parents=True, exist_ok=True)
    path = _job_path(jobs_dir, job.job_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job.to_json(), indent=2))
    tmp.replace(path)


def read_job(jobs_dir: Path, job_id: str) -> FlashJob:
    path = _job_path(jobs_dir, job_id)
    if not path.is_file():
        raise FileNotFoundError(job_id)
    return FlashJob.from_json(json.loads(path.read_text()))


def list_jobs(jobs_dir: Path, *, limit: int = 20) -> list[FlashJob]:
    if not jobs_dir.is_dir():
        return []
    paths = sorted(
        (p for p in jobs_dir.iterdir() if p.is_file() and p.suffix == ".json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    out: list[FlashJob] = []
    for p in paths[:limit]:
        try:
            out.append(FlashJob.from_json(json.loads(p.read_text())))
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("flash job %s unreadable: %r", p.name, exc)
    return out


# ── Errors raised to API layer ───────────────────────────────────────────


class FlashError(RuntimeError):
    """Validation/precondition error before subprocess starts."""


class FlashConflictError(FlashError):
    """A flash is already in progress for this alias — maps to 409."""


# ── Runner ───────────────────────────────────────────────────────────────


class FlashRunner:
    """Owns one flash invocation; instances are not reused across jobs."""

    def __init__(
        self,
        *,
        job: FlashJob,
        device_path: str,
        project_build: Path,
        meta: ProjectMeta,
        jobs_dir: Path,
        ws_manager: Any = None,
        monitor_registry: MonitorRegistry | None = None,
        monitor_to_resume: MonitorConfig | None = None,
        subprocess_factory: Any = None,
    ) -> None:
        self.job = job
        self.device_path = device_path
        self.project_build = project_build
        self.meta = meta
        self.jobs_dir = jobs_dir
        self._ws = ws_manager or default_ws_manager
        self._monitor_registry = monitor_registry or default_monitor_registry
        self._monitor_to_resume = monitor_to_resume
        # Tests inject a fake subprocess here. Production: asyncio default.
        self._subprocess_factory = subprocess_factory or asyncio.create_subprocess_exec

    async def run(self) -> FlashJob:
        channel = channel_for(self.job.alias)
        flash_files = [(ff.offset, ff.path) for ff in self.meta.flash_files]
        cmd = build_esptool_cmd(
            target_chip=self.job.target_chip,
            device_path=self.device_path,
            project_build_dir=self.project_build,
            flash_files=flash_files,
            extra_esptool_args=self.meta.extra_esptool_args,
            write_flash_args=self.meta.write_flash_args,
        )
        self.job.cmd = cmd
        self.job.status = "running"
        write_job(self.jobs_dir, self.job)
        await self._broadcast(channel, {
            "type": "flash_status", "alias": self.job.alias,
            "job_id": self.job.job_id, "state": "running",
            "cmd": cmd,
        })

        proc = None
        try:
            proc = await self._subprocess_factory(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            assert proc.stdout is not None
            async for raw in proc.stdout:
                text = raw.decode("utf-8", errors="replace").rstrip()
                self.job.line_count += 1
                await self._broadcast(channel, {
                    "type": "flash_line",
                    "alias": self.job.alias,
                    "job_id": self.job.job_id,
                    "ts": _utc_iso(),
                    "text": text,
                })
            rc = await proc.wait()
            self.job.return_code = rc
            self.job.status = "success" if rc == 0 else "error"
            if rc != 0:
                self.job.error = f"esptool exited with return code {rc}"
        except FileNotFoundError as exc:
            # e.g. python interpreter or esptool module not importable.
            self.job.status = "error"
            self.job.error = f"could not start esptool: {exc}"
        except asyncio.CancelledError:
            self.job.status = "error"
            self.job.error = "cancelled"
            if proc is not None and proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:  # pragma: no cover — defensive
                    proc.kill()
            raise
        except Exception as exc:  # pragma: no cover — defensive
            self.job.status = "error"
            self.job.error = repr(exc)
        finally:
            self.job.finished_at = datetime.now(timezone.utc)
            write_job(self.jobs_dir, self.job)
            await self._broadcast(channel, {
                "type": "flash_status",
                "alias": self.job.alias,
                "job_id": self.job.job_id,
                "state": self.job.status,
                "return_code": self.job.return_code,
                "error": self.job.error,
            })
            await self._maybe_resume_monitor(channel)

        return self.job

    async def _maybe_resume_monitor(self, channel: str) -> None:
        if self._monitor_to_resume is None:
            return
        try:
            await self._monitor_registry.start(self._monitor_to_resume)
        except Exception as exc:
            logger.warning(
                "flash[%s] monitor resume failed: %r", self.job.alias, exc
            )
            await self._broadcast(channel, {
                "type": "flash_status",
                "alias": self.job.alias,
                "job_id": self.job.job_id,
                "state": "monitor_resume_failed",
                "error": str(exc),
            })
            return
        await self._broadcast(channel, {
            "type": "flash_status",
            "alias": self.job.alias,
            "job_id": self.job.job_id,
            "state": "monitor_resumed",
        })

    async def _broadcast(self, channel: str, payload: dict[str, Any]) -> None:
        try:
            await self._ws.send_json(channel, payload)
        except Exception as exc:  # pragma: no cover — manager swallows internally
            logger.debug("flash[%s] ws send failed: %r", self.job.alias, exc)


def _utc_iso() -> str:
    dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


# ── Job manager (per-alias serialisation) ────────────────────────────────


class FlashJobManager:
    """Tracks running flash jobs across the process.

    A flash kicks off as a background task — REST returns 202 immediately
    and the WS channel does the heavy lifting. The manager keeps the task
    handle so:

      * shutdown can drain in-flight flashes,
      * a duplicate trigger for the same alias is rejected with 409,
      * tests can `await` completion without poking internals.
    """

    def __init__(self) -> None:
        self._active: dict[str, asyncio.Task[FlashJob]] = {}
        self._lock = asyncio.Lock()

    def is_active(self, alias: str) -> bool:
        task = self._active.get(alias)
        return task is not None and not task.done()

    async def submit(
        self,
        *,
        alias: str,
        project: str,
        target_chip: str,
        device_path: str,
        project_build: Path,
        meta: ProjectMeta,
        jobs_dir: Path,
        ws_manager: Any = None,
        monitor_registry: MonitorRegistry | None = None,
        subprocess_factory: Any = None,
    ) -> tuple[FlashJob, asyncio.Task[FlashJob]]:
        """Validate, snapshot+pause monitor, spawn the runner task.

        Returns the FlashJob (status="running" by the time the task is
        scheduled) plus the task handle for callers that want to await it.
        """
        async with self._lock:
            if self.is_active(alias):
                raise FlashConflictError(
                    f"flash already in progress for alias {alias!r}"
                )
            mon_registry = monitor_registry or default_monitor_registry
            ws = ws_manager or default_ws_manager
            channel = channel_for(alias)

            # Snapshot any currently running monitor so we can resume after.
            monitor_to_resume: MonitorConfig | None = None
            existing_mon = mon_registry.get(alias)
            if existing_mon is not None and existing_mon.is_running():
                monitor_to_resume = existing_mon.cfg
                await mon_registry.stop(alias)
                await ws.send_json(channel, {
                    "type": "flash_status",
                    "alias": alias,
                    "state": "monitor_paused",
                })

            job = FlashJob(
                job_id=uuid.uuid4().hex,
                alias=alias,
                project=project,
                target_chip=target_chip,
                started_at=datetime.now(timezone.utc),
                status="pending",
            )
            write_job(jobs_dir, job)

            runner = FlashRunner(
                job=job,
                device_path=device_path,
                project_build=project_build,
                meta=meta,
                jobs_dir=jobs_dir,
                ws_manager=ws,
                monitor_registry=mon_registry,
                monitor_to_resume=monitor_to_resume,
                subprocess_factory=subprocess_factory,
            )
            task = asyncio.create_task(runner.run(), name=f"flash-{alias}-{job.job_id}")
            self._active[alias] = task

            def _cleanup(_t: asyncio.Task[FlashJob]) -> None:
                # Only clear if we're still the registered task — protects
                # against a fast resubmit replacing us mid-cleanup.
                if self._active.get(alias) is task:
                    self._active.pop(alias, None)

            task.add_done_callback(_cleanup)

        return job, task

    async def stop_all(self) -> None:
        """Cancel every in-flight flash. Used on shutdown."""
        async with self._lock:
            tasks = list(self._active.values())
            self._active.clear()
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    def active_aliases(self) -> list[str]:
        return [a for a, t in self._active.items() if not t.done()]


# Module-level singleton; parallel to monitor.registry / ws.manager.
manager = FlashJobManager()


# ── High-level entrypoint used by API layer ──────────────────────────────


async def trigger_flash(
    *,
    alias: str,
    project: str,
    target_chip: str,
    device_path: str,
    projects_root: Path,
    jobs_dir: Path,
    ws_manager: Any = None,
    monitor_registry: MonitorRegistry | None = None,
    subprocess_factory: Any = None,
    job_manager: FlashJobManager | None = None,
) -> tuple[FlashJob, asyncio.Task[FlashJob]]:
    """Resolve the project on disk and submit a flash job.

    Caller is responsible for the upstream validation (alias exists,
    device connected) — this fn just turns "good inputs" into a running
    task. Raises ProjectNotFoundError / FlashConflictError on its own
    failure modes.
    """
    meta = load_project_meta(projects_root, project)
    pb = build_dir(projects_root, project)
    if not pb.is_dir():
        raise FlashError(f"project {project!r} has no build/ directory")
    jm = job_manager or manager
    return await jm.submit(
        alias=alias,
        project=project,
        target_chip=target_chip,
        device_path=device_path,
        project_build=pb,
        meta=meta,
        jobs_dir=jobs_dir,
        ws_manager=ws_manager,
        monitor_registry=monitor_registry,
        subprocess_factory=subprocess_factory,
    )


__all__ = [
    "FlashConflictError",
    "FlashError",
    "FlashJob",
    "FlashJobManager",
    "FlashRunner",
    "FlashStatus",
    "build_esptool_cmd",
    "channel_for",
    "list_jobs",
    "manager",
    "read_job",
    "trigger_flash",
    "write_job",
]
