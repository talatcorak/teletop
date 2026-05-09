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

        # ──────────────────────────────────────────────────────────────────
        # TODO(user): implement the send loop.
        #
        # Required behavior:
        #   1. Send `payload` (str) to every WebSocket in `targets`.
        #   2. Collect any socket whose send raises (WebSocketDisconnect,
        #      RuntimeError from a closed socket, generic Exception — be
        #      permissive; the websocket lib raises a few different types).
        #   3. After the loop, call `await self.disconnect(ws, channel)` for
        #      each dead socket so we don't keep retrying them next tick.
        #
        # Design choice you're making (this is the interesting part):
        #
        #   (A) Sequential `await ws.send_text(payload)` — simple, but a slow
        #       client blocks every later client in the channel. Fine for a
        #       handful of monitors on a LAN; bad if any client is on flaky
        #       wifi while serial output is bursting at 921600 baud.
        #
        #   (B) `await asyncio.gather(*sends, return_exceptions=True)` —
        #       concurrent, no head-of-line blocking. You then walk the
        #       results and pair `targets[i]` with exceptions. Slightly more
        #       code, much better isolation between subscribers.
        #
        # For teletop we expect 1–3 viewers per device on a trusted LAN, but
        # the same manager will also push flash progress (chunky binary-ish
        # ticks) so isolation matters more than absolute simplicity.
        #
        # Pick one and write it. ~5–10 lines.
        # ──────────────────────────────────────────────────────────────────
        raise NotImplementedError("ConnectionManager.broadcast: implement send loop")

    async def send_json(self, channel: str, payload: dict[str, Any]) -> None:
        """Structured-event helper. Same as broadcast(channel, dict)."""
        await self.broadcast(channel, payload)


# Module-level singleton — every other module imports this name.
manager = ConnectionManager()


__all__ = ["ConnectionManager", "WebSocketDisconnect", "manager"]
