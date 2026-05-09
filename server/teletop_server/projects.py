"""Project storage: uploaded firmware bundles + metadata under data_dir.

A "project" is a tar.gz of an ESP-IDF / esptool-friendly ``build/`` tree. We
require ``flasher_args.json`` at the archive root because that file is the
contract between PC-side build artifacts and the on-RPi flash command:
``flash_files`` (offset → relative path) plus optional
``extra_esptool_args`` (chip family, before/after, write_flash_args).

On-disk layout:

    <data_dir>/projects/<name>/
        meta.json
        build/
            flasher_args.json
            <name>.bin
            bootloader/bootloader.bin
            ...

``meta.json`` is the public face — list/show endpoints return it verbatim.
The ``build/`` subtree is opaque to callers; only the flash subsystem reads
it.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger("teletop.projects")


PROJECT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
FLASHER_ARGS_FILENAME = "flasher_args.json"


# ── Errors ───────────────────────────────────────────────────────────────


class ProjectError(ValueError):
    """Generic upload/validation error — handler maps to 400."""


class ProjectNotFoundError(KeyError):
    """Asked for a project name that isn't on disk — handler maps to 404."""


# ── Public models ────────────────────────────────────────────────────────


class FlashFile(BaseModel):
    offset: str            # "0x1000"
    path: str              # relative to build/, POSIX-style
    size: int              # bytes


class ProjectMeta(BaseModel):
    """Persisted as ``<project_dir>/meta.json``. Returned by show()."""

    name: str
    uploaded_at: datetime
    source: str = "pc-build"
    target_chip_hint: str | None = None
    flash_files: list[FlashFile]
    extra_esptool_args: dict[str, Any] = {}
    write_flash_args: list[str] = []


class ProjectListItem(BaseModel):
    name: str
    uploaded_at: datetime
    source: str
    target_chip_hint: str | None
    file_count: int
    total_size: int


# ── Path helpers ─────────────────────────────────────────────────────────


def validate_project_name(name: str) -> None:
    if not PROJECT_NAME_PATTERN.match(name):
        raise ProjectError(
            f"project name {name!r} is invalid — letters, digits, '.', '_', '-' only"
        )


def project_dir(projects_root: Path, name: str) -> Path:
    validate_project_name(name)
    return projects_root / name


def build_dir(projects_root: Path, name: str) -> Path:
    return project_dir(projects_root, name) / "build"


def meta_path(projects_root: Path, name: str) -> Path:
    return project_dir(projects_root, name) / "meta.json"


# ── Tar extraction with traversal protection ─────────────────────────────


def _safe_extract_tar(tf: tarfile.TarFile, dest: Path) -> None:
    """Extract ``tf`` into ``dest`` rejecting any member whose resolved path
    would land outside ``dest`` (Zip Slip / tar slip). Symlinks pointing
    outside dest are also rejected.
    """
    dest_resolved = dest.resolve()
    for member in tf.getmembers():
        # Reject absolute paths and parent-relative names up front so an
        # empty tarball with one ".." entry can't even be staged.
        if member.name.startswith("/") or ".." in Path(member.name).parts:
            raise ProjectError(f"unsafe path in archive: {member.name!r}")
        target = (dest / member.name).resolve()
        try:
            target.relative_to(dest_resolved)
        except ValueError as exc:
            raise ProjectError(
                f"archive entry {member.name!r} escapes extraction dir"
            ) from exc
        # Symlink targets too — could re-introduce escape via link.
        if member.issym() or member.islnk():
            link_target = Path(member.linkname)
            if link_target.is_absolute():
                raise ProjectError(
                    f"absolute symlink in archive: {member.name!r} → {member.linkname}"
                )
            resolved = (target.parent / link_target).resolve()
            try:
                resolved.relative_to(dest_resolved)
            except ValueError as exc:
                raise ProjectError(
                    f"symlink {member.name!r} escapes extraction dir"
                ) from exc
    # filter='data' (Python 3.12+) further hardens extraction; gracefully
    # fall back on older interpreters.
    try:
        tf.extractall(dest, filter="data")  # type: ignore[arg-type]
    except TypeError:  # pragma: no cover — Python < 3.12
        tf.extractall(dest)


def _find_build_root(staged: Path) -> Path:
    """Locate the directory that holds ``flasher_args.json``.

    Some users upload the build directory directly (``flasher_args.json`` at
    the tar root); others wrap it in a top-level folder (``build/...``). We
    accept both shapes so the CLI doesn't need to second-guess what the
    user tar'd.
    """
    if (staged / FLASHER_ARGS_FILENAME).is_file():
        return staged
    # Look one level deep for a single subdir holding flasher_args.json.
    candidates = [
        d for d in staged.iterdir()
        if d.is_dir() and (d / FLASHER_ARGS_FILENAME).is_file()
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise ProjectError(
        f"{FLASHER_ARGS_FILENAME} not found at archive root or single top-level dir"
    )


# ── flasher_args.json validation ─────────────────────────────────────────


def _parse_flasher_args(build_root: Path) -> tuple[
    list[FlashFile], dict[str, Any], list[str], str | None
]:
    """Validate flasher_args.json and return (flash_files, extras, write_args, chip_hint).

    Raises ProjectError on any structural problem so the upload handler can
    surface a clean 400.
    """
    raw_text = (build_root / FLASHER_ARGS_FILENAME).read_text()
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ProjectError(f"{FLASHER_ARGS_FILENAME} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProjectError(f"{FLASHER_ARGS_FILENAME} must be a JSON object")

    flash_files_raw = data.get("flash_files")
    if not isinstance(flash_files_raw, dict) or not flash_files_raw:
        raise ProjectError("flasher_args.json missing non-empty 'flash_files' object")

    flash_files: list[FlashFile] = []
    for offset, rel_path in flash_files_raw.items():
        if not isinstance(offset, str) or not offset.lower().startswith("0x"):
            raise ProjectError(
                f"flash_files offset must be hex string ('0x1000'), got {offset!r}"
            )
        try:
            int(offset, 16)
        except ValueError as exc:
            raise ProjectError(f"flash_files offset {offset!r} not parseable as hex") from exc
        if not isinstance(rel_path, str) or not rel_path:
            raise ProjectError(
                f"flash_files value for offset {offset!r} must be non-empty string"
            )
        if rel_path.startswith("/") or ".." in Path(rel_path).parts:
            raise ProjectError(
                f"flash_files path {rel_path!r} must be relative without '..'"
            )
        full = build_root / rel_path
        if not full.is_file():
            raise ProjectError(
                f"flash_files references missing binary: {rel_path!r}"
            )
        flash_files.append(
            FlashFile(offset=offset, path=rel_path, size=full.stat().st_size)
        )

    extras = data.get("extra_esptool_args") or {}
    if not isinstance(extras, dict):
        raise ProjectError("extra_esptool_args must be an object when present")

    write_flash_args = data.get("write_flash_args") or []
    if not isinstance(write_flash_args, list) or not all(
        isinstance(x, str) for x in write_flash_args
    ):
        raise ProjectError("write_flash_args must be a list of strings when present")

    chip_hint = extras.get("chip") if isinstance(extras.get("chip"), str) else None
    return flash_files, extras, write_flash_args, chip_hint


# ── Public API ───────────────────────────────────────────────────────────


def upload_project(
    projects_root: Path,
    name: str,
    archive_bytes: bytes,
    *,
    max_size_mb: int,
) -> ProjectMeta:
    """Validate + extract + atomically swap into <projects_root>/<name>/.

    The whole pipeline runs in a temp dir first so a malformed upload never
    half-replaces an existing project. The previous version (if any) is
    only deleted once validation succeeds.
    """
    validate_project_name(name)
    if len(archive_bytes) > max_size_mb * 1024 * 1024:
        raise ProjectError(
            f"archive is {len(archive_bytes)} bytes; limit is {max_size_mb} MiB"
        )

    projects_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"teletop-upload-{name}-", dir=projects_root
    ) as tmp_str:
        tmp = Path(tmp_str)
        archive_path = tmp / "upload.tar.gz"
        archive_path.write_bytes(archive_bytes)
        staged = tmp / "staged"
        staged.mkdir()

        try:
            with tarfile.open(archive_path, "r:gz") as tf:
                _safe_extract_tar(tf, staged)
        except tarfile.TarError as exc:
            raise ProjectError(f"could not read tar.gz: {exc}") from exc

        build_root = _find_build_root(staged)
        flash_files, extras, write_args, chip_hint = _parse_flasher_args(build_root)

        meta = ProjectMeta(
            name=name,
            uploaded_at=datetime.now(timezone.utc),
            source="pc-build",
            target_chip_hint=chip_hint,
            flash_files=flash_files,
            extra_esptool_args=extras,
            write_flash_args=write_args,
        )

        # Stage the final layout under tmp before swap so the destination
        # transition is a single rename.
        final_stage = tmp / "final"
        final_stage.mkdir()
        shutil.copytree(build_root, final_stage / "build")
        (final_stage / "meta.json").write_text(
            meta.model_dump_json(indent=2)
        )

        target = projects_root / name
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(final_stage), str(target))

    logger.info(
        "project[%s] uploaded — %d files, target_chip_hint=%s",
        name,
        len(flash_files),
        chip_hint,
    )
    return meta


def load_project_meta(projects_root: Path, name: str) -> ProjectMeta:
    p = meta_path(projects_root, name)
    if not p.is_file():
        raise ProjectNotFoundError(name)
    try:
        return ProjectMeta.model_validate_json(p.read_text())
    except Exception as exc:
        raise ProjectError(f"meta.json for {name!r} is corrupt: {exc}") from exc


def list_projects(projects_root: Path) -> list[ProjectListItem]:
    if not projects_root.is_dir():
        return []
    out: list[ProjectListItem] = []
    for child in sorted(projects_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("teletop-upload-"):
            # Stale temp dir from a crashed upload — skip silently.
            continue
        try:
            meta = load_project_meta(projects_root, child.name)
        except (ProjectNotFoundError, ProjectError):
            continue
        bd = build_dir(projects_root, child.name)
        file_count = 0
        total_size = 0
        if bd.is_dir():
            for path in bd.rglob("*"):
                if path.is_file():
                    file_count += 1
                    total_size += path.stat().st_size
        out.append(
            ProjectListItem(
                name=meta.name,
                uploaded_at=meta.uploaded_at,
                source=meta.source,
                target_chip_hint=meta.target_chip_hint,
                file_count=file_count,
                total_size=total_size,
            )
        )
    return out


def delete_project(projects_root: Path, name: str) -> None:
    p = project_dir(projects_root, name)
    if not p.exists():
        raise ProjectNotFoundError(name)
    shutil.rmtree(p)
    logger.info("project[%s] deleted", name)


__all__ = [
    "FLASHER_ARGS_FILENAME",
    "FlashFile",
    "ProjectError",
    "ProjectListItem",
    "ProjectMeta",
    "ProjectNotFoundError",
    "build_dir",
    "delete_project",
    "list_projects",
    "load_project_meta",
    "meta_path",
    "project_dir",
    "upload_project",
    "validate_project_name",
]
