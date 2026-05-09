"""Smoke + unit tests for the FastAPI app and the WebSocket connection manager.

We deliberately mix two styles:

1. Integration tests using fastapi.testclient.TestClient — they exercise the
   real `/ws/test/{channel}` endpoint and prove that broadcast() actually
   delivers messages end-to-end through the websocket layer.

2. Direct unit tests of `ConnectionManager` with an in-memory fake socket —
   they let us verify branches that are awkward to trigger over a real
   websocket (e.g. send_text raising, channel pruning, dict→JSON encoding).
"""

from __future__ import annotations

import asyncio
import json
import time

from fastapi.testclient import TestClient

from teletop_server.main import app
from teletop_server.ws import ConnectionManager, manager


# ── Integration tests: exercise the real WS endpoint ─────────────────────


def test_health() -> None:
    with TestClient(app) as client:
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "teletop"
        assert "version" in body
        assert body["uptime_seconds"] >= 0
        assert "data_dir" in body
        assert body["ws_channels"] == 0


def test_ws_test_echo() -> None:
    """Single client → broadcast → echo back. Proves broadcast() runs."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/test/dev1") as ws:
            ws.send_text("hello")
            assert ws.receive_text() == "echo[dev1]: hello"


def test_ws_broadcast_two_clients() -> None:
    """Two clients on the same channel — one sends, BOTH must receive.

    This is the core fan-out guarantee of ConnectionManager. If broadcast()
    only sent to the first socket, the second receive_text would block and
    the test would time out.
    """
    with TestClient(app) as client:
        with client.websocket_connect("/ws/test/room") as a:
            with client.websocket_connect("/ws/test/room") as b:
                a.send_text("ping")
                assert a.receive_text() == "echo[room]: ping"
                assert b.receive_text() == "echo[room]: ping"


def test_ws_channels_isolated() -> None:
    """Messages do not leak across channels."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/test/alpha") as a:
            with client.websocket_connect("/ws/test/beta") as b:
                a.send_text("only-alpha")
                assert a.receive_text() == "echo[alpha]: only-alpha"
                b.send_text("only-beta")
                assert b.receive_text() == "echo[beta]: only-beta"


def test_ws_disconnect_prunes_channel() -> None:
    """When the last subscriber leaves, the channel key is removed."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/test/ephemeral"):
            assert "ephemeral" in manager.channels
        # WebSocketDisconnect → endpoint calls manager.disconnect; give it
        # a moment since the cleanup runs on the server task, not ours.
        for _ in range(50):
            if "ephemeral" not in manager.channels:
                break
            time.sleep(0.01)
        assert "ephemeral" not in manager.channels


# ── Direct unit tests of ConnectionManager ───────────────────────────────


class _FakeWS:
    """Minimal stand-in for fastapi.WebSocket — captures send_text calls."""

    def __init__(self, *, fail: bool = False) -> None:
        self.sent: list[str] = []
        self.fail = fail
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, payload: str) -> None:
        if self.fail:
            raise RuntimeError("simulated broken pipe")
        self.sent.append(payload)


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def test_manager_broadcast_str() -> None:
    async def scenario() -> tuple[_FakeWS, _FakeWS]:
        mgr = ConnectionManager()
        a, b = _FakeWS(), _FakeWS()
        await mgr.connect(a, "ch")  # type: ignore[arg-type]
        await mgr.connect(b, "ch")  # type: ignore[arg-type]
        await mgr.broadcast("ch", "hi")
        return a, b

    a, b = _run(scenario())
    assert a.sent == ["hi"]
    assert b.sent == ["hi"]


def test_manager_broadcast_dict_encodes_json() -> None:
    async def scenario() -> _FakeWS:
        mgr = ConnectionManager()
        a = _FakeWS()
        await mgr.connect(a, "ch")  # type: ignore[arg-type]
        await mgr.send_json("ch", {"event": "tick", "n": 1})
        return a

    a = _run(scenario())
    assert len(a.sent) == 1
    assert json.loads(a.sent[0]) == {"event": "tick", "n": 1}


def test_manager_prunes_dead_socket() -> None:
    """A failing send must not block siblings, and the dead socket gets removed."""

    async def scenario() -> tuple[ConnectionManager, _FakeWS, _FakeWS]:
        mgr = ConnectionManager()
        healthy = _FakeWS()
        broken = _FakeWS(fail=True)
        await mgr.connect(healthy, "ch")  # type: ignore[arg-type]
        await mgr.connect(broken, "ch")  # type: ignore[arg-type]
        assert len(mgr.channels["ch"]) == 2
        await mgr.broadcast("ch", "ping")
        return mgr, healthy, broken

    mgr, healthy, broken = _run(scenario())
    # Healthy socket received the message even though sibling failed.
    assert healthy.sent == ["ping"]
    # Broken socket got pruned; healthy still present.
    assert broken not in mgr.channels.get("ch", set())
    assert healthy in mgr.channels["ch"]


def test_manager_broadcast_empty_channel_noop() -> None:
    async def scenario() -> None:
        mgr = ConnectionManager()
        await mgr.broadcast("nobody-here", "lonely")
        await mgr.send_json("nobody-here", {"x": 1})

    _run(scenario())  # must not raise


def test_manager_disconnect_drops_empty_channel() -> None:
    async def scenario() -> ConnectionManager:
        mgr = ConnectionManager()
        a = _FakeWS()
        await mgr.connect(a, "ch")  # type: ignore[arg-type]
        assert "ch" in mgr.channels
        await mgr.disconnect(a, "ch")  # type: ignore[arg-type]
        return mgr

    mgr = _run(scenario())
    assert "ch" not in mgr.channels
