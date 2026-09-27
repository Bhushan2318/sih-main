"""On the serving box a write is refused before its body is read.

FastAPI parses a JSON body before any dependency runs, so the upload router's read-only 409
arrived after the parse had already been paid for. Measured 2026-09-28 on a box-mode mirror
of the live bundle (fresh server, RSS sampled every 20 ms): one 5 MB body to
/api/upload/{id}/confirm-mapping took it from 160 to 234 MB, one 15 MB body from 161 to
424 MB - on a box killed at 512 that idles near 240. Every write on the box is refused
anyway, so app/api/box_guard.py refuses it without reading a byte.

The ASGI calls below are plumbing (no data, no metric): the inner app and the receive
channel are stand-ins that record whether they were reached.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.api import box_guard
from app.config import settings


def _call(scope: dict) -> tuple[list[dict], dict]:
    """Run one request through the guard. Returns (messages sent, what was reached)."""
    reached = {"app": False, "receive": False}
    sent: list[dict] = []

    async def inner(scope, receive, send):
        reached["app"] = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"inner"})

    async def receive():
        reached["receive"] = True
        return {"type": "http.request", "body": b"x" * 1024, "more_body": True}

    async def send(message):
        sent.append(message)

    # A private loop: asyncio.run() would clear this thread's loop, which later tests use.
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(box_guard.BoxRequestGuard(inner)(scope, receive, send))
    finally:
        loop.close()
    return sent, reached


def _http(method: str, path: str, headers: "list[tuple[bytes, bytes]]" = ()) -> dict:
    return {"type": "http", "method": method, "path": path, "headers": list(headers),
            "query_string": b""}


def _status(sent: list[dict]) -> int:
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


def _detail(sent: list[dict]) -> str:
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return json.loads(body)["detail"]


@pytest.mark.parametrize("method,path", [
    ("POST", "/api/upload/x/confirm-mapping"),
    ("POST", "/api/upload"),
    ("POST", "/api/ingest/run-cycle"),
    ("PUT", "/api/regions/IN-MH-NAGPUR"),
    ("DELETE", "/api/model/status"),
    ("PATCH", "/"),
])
def test_a_write_on_the_box_is_refused_without_reading_its_body(monkeypatch, method, path):
    monkeypatch.setattr(settings, "serving_read_only", True)
    sent, reached = _call(_http(method, path, [(b"content-length", b"15000000"),
                                               (b"content-type", b"application/json")]))
    assert _status(sent) == 409
    assert "read-only" in _detail(sent)
    assert reached == {"app": False, "receive": False}


@pytest.mark.parametrize("headers", [
    [(b"content-length", str(box_guard.MAX_BODY_BYTES + 1).encode())],
    [(b"transfer-encoding", b"chunked")],
])
def test_a_read_carrying_a_body_on_the_box_is_refused_unread(monkeypatch, headers):
    """No read endpoint takes a body; one that arrives with a large or unannounced one is
    refused before it is buffered."""
    monkeypatch.setattr(settings, "serving_read_only", True)
    sent, reached = _call(_http("GET", "/api/regions/all", headers))
    assert _status(sent) == 413
    assert reached == {"app": False, "receive": False}


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS"])
def test_reads_on_the_box_pass_through(monkeypatch, method):
    """OPTIONS too: a CORS preflight is answered by the CORS middleware, not refused."""
    monkeypatch.setattr(settings, "serving_read_only", True)
    sent, reached = _call(_http(method, "/api/regions/all", [(b"content-length", b"0")]))
    assert reached["app"] and _status(sent) == 200


def test_off_the_box_writes_reach_the_app(monkeypatch):
    """A dev server uploads and ingests; the guard is the box's alone."""
    monkeypatch.setattr(settings, "serving_read_only", False)
    sent, reached = _call(_http("POST", "/api/upload", [(b"content-length", b"15000000")]))
    assert reached["app"] and _status(sent) == 200


def test_websockets_and_lifespan_pass_through(monkeypatch):
    monkeypatch.setattr(settings, "serving_read_only", True)
    for scope in ({"type": "websocket", "path": "/ws", "headers": []}, {"type": "lifespan"}):
        _, reached = _call(scope)
        assert reached["app"], scope["type"]


def test_the_app_refuses_a_large_write_on_the_box(monkeypatch):
    """Wired into the real app, ahead of routing - the 409 is the guard's, not the upload
    router's, which is only reached after the body is parsed."""
    from app.main import app

    monkeypatch.setattr(settings, "serving_read_only", True)
    body = json.dumps({"mappings": [{}] * 200_000})
    r = TestClient(app).post("/api/upload/x/confirm-mapping", content=body,
                             headers={"Content-Type": "application/json"})
    assert r.status_code == 409
    assert "read-only" in r.json()["detail"]
