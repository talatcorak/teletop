"""FastAPI application entry point."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import __version__ as fallback_version
from .config import get_settings
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


def cli() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "teletop_server.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    cli()
