"""WebSocket connection manager.

Tracks active client connections grouped by device id so the monitor module
can fan serial output out to every viewer of a given device.
"""

from __future__ import annotations

from collections import defaultdict

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self._rooms: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, device_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._rooms[device_id].add(ws)

    def disconnect(self, device_id: str, ws: WebSocket) -> None:
        self._rooms[device_id].discard(ws)
        if not self._rooms[device_id]:
            self._rooms.pop(device_id, None)

    async def broadcast(self, device_id: str, message: str) -> None:
        # TODO: backpressure / drop-on-slow-client policy is a Task-N decision.
        dead: list[WebSocket] = []
        for ws in self._rooms.get(device_id, set()):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(device_id, ws)


manager = ConnectionManager()
