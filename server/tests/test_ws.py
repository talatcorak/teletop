"""Smoke tests for the FastAPI app + WebSocket connection manager."""

from __future__ import annotations

from fastapi.testclient import TestClient

from teletop_server.main import app


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
    with TestClient(app) as client:
        with client.websocket_connect("/ws/test/dev1") as ws:
            ws.send_text("hello")
            received = ws.receive_text()
            assert received == "echo[dev1]: hello"


def test_ws_broadcast_two_clients() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws/test/room") as a:
            with client.websocket_connect("/ws/test/room") as b:
                a.send_text("ping")
                # Both subscribers in the same channel see the echoed message.
                assert a.receive_text() == "echo[room]: ping"
                assert b.receive_text() == "echo[room]: ping"


def test_ws_channels_isolated() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws/test/alpha") as a:
            with client.websocket_connect("/ws/test/beta") as b:
                a.send_text("only-alpha")
                assert a.receive_text() == "echo[alpha]: only-alpha"
                # b should NOT receive — different channel. Send something on
                # b so we can drain its socket and confirm it didn't get the
                # cross-channel message first.
                b.send_text("only-beta")
                assert b.receive_text() == "echo[beta]: only-beta"
