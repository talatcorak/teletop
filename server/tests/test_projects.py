"""Tests for project upload + listing (validation, traversal, REST, CLI)."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from teletop_server.projects import (
    ProjectError,
    ProjectNotFoundError,
    delete_project,
    list_projects,
    load_project_meta,
    upload_project,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TELETOP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TELETOP_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("TELETOP_FLASH_JOBS_DIR", str(tmp_path / "flash_jobs"))
    monkeypatch.setenv("TELETOP_LOG_DIR", str(tmp_path / "logs"))
    return tmp_path


# ── Helpers ──────────────────────────────────────────────────────────────


def _build_tarball(
    files: dict[str, bytes],
    *,
    inner_root: str | None = None,
    add_traversal: bool = False,
) -> bytes:
    """Pack ``files`` (path → bytes) into a gzipped tar in memory.

    inner_root: if set, prepend this directory under the archive (mimics a
    user who tarred their project root rather than the build/ dir).
    add_traversal: insert a "../escape.bin" member (rejected by extractor).
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel, data in files.items():
            arcname = f"{inner_root}/{rel}" if inner_root else rel
            ti = tarfile.TarInfo(name=arcname)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        if add_traversal:
            evil = tarfile.TarInfo(name="../escape.bin")
            evil.size = 4
            tf.addfile(evil, io.BytesIO(b"evil"))
    return buf.getvalue()


def _flasher_args_v5(chip: str = "esp32") -> dict[str, object]:
    return {
        "write_flash_args": ["--flash_mode", "dio", "--flash_size", "2MB"],
        "flash_settings": {"flash_mode": "dio", "flash_size": "2MB"},
        "flash_files": {
            "0x1000": "bootloader/bootloader.bin",
            "0x8000": "partition_table/partition-table.bin",
            "0x10000": "hello_world.bin",
        },
        "extra_esptool_args": {
            "after": "hard_reset",
            "before": "default_reset",
            "chip": chip,
        },
    }


def _build_files(chip: str = "esp32") -> dict[str, bytes]:
    return {
        "flasher_args.json": json.dumps(_flasher_args_v5(chip)).encode(),
        "bootloader/bootloader.bin": b"\x00" * 100,
        "partition_table/partition-table.bin": b"\x01" * 50,
        "hello_world.bin": b"\xaa" * 200,
    }


# ── Validation tests ─────────────────────────────────────────────────────


def test_upload_rejects_invalid_project_name(tmp_path: Path) -> None:
    with pytest.raises(ProjectError, match="invalid"):
        upload_project(tmp_path, "../bad", b"", max_size_mb=10)


def test_upload_rejects_oversized_archive(tmp_path: Path) -> None:
    big = b"x" * (5 * 1024 * 1024 + 1)
    with pytest.raises(ProjectError, match="limit"):
        upload_project(tmp_path, "ok", big, max_size_mb=5)


def test_upload_rejects_missing_flasher_args(tmp_path: Path) -> None:
    blob = _build_tarball({"hello.bin": b"x"})
    with pytest.raises(ProjectError, match="flasher_args.json"):
        upload_project(tmp_path, "p", blob, max_size_mb=10)


def test_upload_rejects_path_traversal(tmp_path: Path) -> None:
    blob = _build_tarball(_build_files(), add_traversal=True)
    with pytest.raises(ProjectError, match="unsafe path"):
        upload_project(tmp_path, "p", blob, max_size_mb=10)


def test_upload_rejects_missing_bin(tmp_path: Path) -> None:
    files = _build_files()
    # flasher_args references hello_world.bin but we don't include it
    files.pop("hello_world.bin")
    blob = _build_tarball(files)
    with pytest.raises(ProjectError, match="missing binary"):
        upload_project(tmp_path, "p", blob, max_size_mb=10)


def test_upload_rejects_absolute_path_in_flash_files(tmp_path: Path) -> None:
    bad = _flasher_args_v5()
    bad["flash_files"] = {"0x1000": "/etc/passwd"}
    blob = _build_tarball({
        "flasher_args.json": json.dumps(bad).encode(),
    })
    with pytest.raises(ProjectError, match="must be relative"):
        upload_project(tmp_path, "p", blob, max_size_mb=10)


def test_upload_rejects_non_hex_offset(tmp_path: Path) -> None:
    bad = _flasher_args_v5()
    bad["flash_files"] = {"4096": "hello.bin"}
    blob = _build_tarball({
        "flasher_args.json": json.dumps(bad).encode(),
        "hello.bin": b"x",
    })
    with pytest.raises(ProjectError, match="hex string"):
        upload_project(tmp_path, "p", blob, max_size_mb=10)


# ── Happy path ───────────────────────────────────────────────────────────


def test_upload_happy_path_writes_meta_and_files(tmp_path: Path) -> None:
    blob = _build_tarball(_build_files())
    meta = upload_project(tmp_path, "hello_world", blob, max_size_mb=10)

    assert meta.name == "hello_world"
    assert meta.target_chip_hint == "esp32"
    assert {ff.offset for ff in meta.flash_files} == {"0x1000", "0x8000", "0x10000"}
    assert all(ff.size > 0 for ff in meta.flash_files)

    project_root = tmp_path / "hello_world"
    assert (project_root / "meta.json").is_file()
    assert (project_root / "build" / "hello_world.bin").read_bytes() == b"\xaa" * 200

    # round-trip via load_project_meta
    loaded = load_project_meta(tmp_path, "hello_world")
    assert loaded.name == "hello_world"
    assert loaded.extra_esptool_args["after"] == "hard_reset"


def test_upload_accepts_wrapped_root_dir(tmp_path: Path) -> None:
    """Some users tar the parent dir — we accept a single top-level wrapper."""
    blob = _build_tarball(_build_files(), inner_root="build")
    meta = upload_project(tmp_path, "wrapped", blob, max_size_mb=10)
    assert meta.name == "wrapped"
    assert (tmp_path / "wrapped" / "build" / "hello_world.bin").is_file()


def test_upload_overwrites_existing_project(tmp_path: Path) -> None:
    blob1 = _build_tarball(_build_files())
    upload_project(tmp_path, "p", blob1, max_size_mb=10)
    files2 = _build_files()
    files2["hello_world.bin"] = b"\xbb" * 50  # different content + size
    blob2 = _build_tarball(files2)
    meta2 = upload_project(tmp_path, "p", blob2, max_size_mb=10)
    bin_size = next(ff.size for ff in meta2.flash_files if ff.offset == "0x10000")
    assert bin_size == 50
    assert (tmp_path / "p" / "build" / "hello_world.bin").read_bytes() == b"\xbb" * 50


def test_upload_failed_validation_does_not_clobber_existing(tmp_path: Path) -> None:
    blob1 = _build_tarball(_build_files())
    upload_project(tmp_path, "p", blob1, max_size_mb=10)
    bad = _build_tarball({"hello.bin": b"x"})  # no flasher_args.json
    with pytest.raises(ProjectError):
        upload_project(tmp_path, "p", bad, max_size_mb=10)
    # Original survives untouched.
    assert (tmp_path / "p" / "build" / "hello_world.bin").is_file()


def test_list_projects_returns_metadata(tmp_path: Path) -> None:
    upload_project(tmp_path, "p1", _build_tarball(_build_files()), max_size_mb=10)
    upload_project(tmp_path, "p2", _build_tarball(_build_files("esp8266")), max_size_mb=10)
    rows = list_projects(tmp_path)
    by_name = {r.name: r for r in rows}
    assert set(by_name) == {"p1", "p2"}
    assert by_name["p2"].target_chip_hint == "esp8266"
    # 3 .bin files + flasher_args.json all live under build/.
    assert by_name["p1"].file_count == 4
    assert by_name["p1"].total_size > 0


def test_list_projects_skips_temp_upload_dirs(tmp_path: Path) -> None:
    upload_project(tmp_path, "p1", _build_tarball(_build_files()), max_size_mb=10)
    (tmp_path / "teletop-upload-leftover").mkdir()
    rows = list_projects(tmp_path)
    assert {r.name for r in rows} == {"p1"}


def test_delete_project(tmp_path: Path) -> None:
    upload_project(tmp_path, "p", _build_tarball(_build_files()), max_size_mb=10)
    delete_project(tmp_path, "p")
    assert not (tmp_path / "p").exists()
    with pytest.raises(ProjectNotFoundError):
        delete_project(tmp_path, "p")


# ── REST endpoints ───────────────────────────────────────────────────────


def test_api_project_upload_and_list(isolated_data_dir: Path) -> None:
    from teletop_server.main import app

    blob = _build_tarball(_build_files())
    with TestClient(app) as client:
        r = client.post(
            "/api/projects/hello_world/upload",
            files={"archive": ("hello.tar.gz", blob, "application/gzip")},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["name"] == "hello_world"
        assert body["target_chip_hint"] == "esp32"

        r2 = client.get("/api/projects")
        assert r2.status_code == 200
        names = [p["name"] for p in r2.json()]
        assert "hello_world" in names

        r3 = client.get("/api/projects/hello_world")
        assert r3.status_code == 200
        detail = r3.json()
        assert "files" in detail
        assert any(f["offset"] == "0x10000" for f in detail["files"])

        r4 = client.delete("/api/projects/hello_world")
        assert r4.status_code == 204
        r5 = client.get("/api/projects/hello_world")
        assert r5.status_code == 404


def test_api_project_upload_validation_400(isolated_data_dir: Path) -> None:
    from teletop_server.main import app

    bad = _build_tarball({"oops.bin": b"x"})  # no flasher_args.json
    with TestClient(app) as client:
        r = client.post(
            "/api/projects/p/upload",
            files={"archive": ("p.tar.gz", bad, "application/gzip")},
        )
        assert r.status_code == 400
        assert "flasher_args.json" in r.json()["detail"]


def test_api_project_show_404(isolated_data_dir: Path) -> None:
    from teletop_server.main import app

    with TestClient(app) as client:
        r = client.get("/api/projects/ghost")
        assert r.status_code == 404


# ── CLI smoke ────────────────────────────────────────────────────────────


def test_cli_projects_subgroup_help() -> None:
    from teletop_server.main import cli

    result = CliRunner().invoke(cli, ["projects", "--help"])
    assert result.exit_code == 0, result.output
    for sub in ("upload", "list", "show", "delete"):
        assert sub in result.output


def test_cli_projects_list_complains_when_server_down() -> None:
    """CLI should surface a clean error when the server isn't reachable."""
    import os

    from teletop_server.main import cli

    os.environ["TELETOP_PORT"] = "59998"
    try:
        result = CliRunner().invoke(cli, ["projects", "list"])
    finally:
        del os.environ["TELETOP_PORT"]
    assert result.exit_code != 0
    assert "could not reach server" in result.output.lower()
