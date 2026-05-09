"""WebSocket connection manager.

Channel-based fan-out. A "channel" is an opaque string — for the monitor it's
the device id; for flash progress it's the flash job id; for project events it
might be "projects". The manager doesn't care about semantics, only routing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

logger = logging.getLogger("teletop.ws")


class ConnectionManager:
    def __init__(self) -> None:
        self.channels: dict[str, set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, channel: str) -> None:
        await websocket.accept()
        async with self._lock:
            self.channels.setdefault(channel, set()).add(websocket)
        logger.info("ws connected channel=%s total=%d", channel, len(self.channels[channel]))

    async def disconnect(self, websocket: WebSocket, channel: str) -> None:
        async with self._lock:
            members = self.channels.get(channel)
            if members is None:
                return
            members.discard(websocket)
            if not members:
                self.channels.pop(channel, None)
        logger.info("ws disconnected channel=%s", channel)

    async def broadcast(self, channel: str, message: str | dict[str, Any]) -> None:
        """Fan a message out to every socket in `channel`.

        Encodes dict → JSON. Removes sockets that fail to send.
        """
        # Snapshot under lock so a concurrent connect/disconnect can't mutate
        # the set while we iterate.
        async with self._lock:
            targets = list(self.channels.get(channel, ()))
        if not targets:
            return

        payload: str = json.dumps(message) if isinstance(message, dict) else message

        # Strategy (B): concurrent send via asyncio.gather. A slow or stuck
        # client must not delay deliveries to its siblings — flash progress
        # and high-baud serial output share this manager.
        results = await asyncio.gather(
            *(ws.send_text(payload) for ws in targets),
            return_exceptions=True,
        )
        dead = [ws for ws, res in zip(targets, results) if isinstance(res, BaseException)]
        for ws, res in zip(targets, results):
            if isinstance(res, BaseException):
                logger.debug("ws send failed channel=%s err=%r", channel, res)
        for ws in dead:
            await self.disconnect(ws, channel)

    async def send_json(self, channel: str, payload: dict[str, Any]) -> None:
        """Structured-event helper. Same as broadcast(channel, dict)."""
        await self.broadcast(channel, payload)


# Module-level singleton — every other module imports this name.
manager = ConnectionManager()


__all__ = ["ConnectionManager", "WebSocketDisconnect", "manager"]
