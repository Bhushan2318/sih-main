"""On the serving box, refuse a write before its body is read.

WHY
FastAPI reads and parses a request body before any dependency runs, so the upload router's
read-only 409 came after the parse had been paid for. Measured 2026-09-28 on a box-mode
mirror of the live bundle (fresh server, RSS sampled every 20 ms): one 5 MB JSON body to
/api/upload/{id}/confirm-mapping took it from 160 to 234 MB, one 15 MB body from 161 to
424 MB. The box is killed at 512 and idles near 240.

Every write the box could be sent is refused there anyway - uploads, ingest, retrain - so
this refuses them all at the door, as plain ASGI, before routing and before a byte of body
is received. Reads take no body; one that arrives with a large or unannounced (chunked)
body is refused the same way.

Only under SERVING_READ_ONLY, read per request so tests can toggle it. A dev server uploads
and ingests as before.
"""

from __future__ import annotations

import json

from app.config import settings

READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
MAX_BODY_BYTES = 64 * 1024

WRITE_REFUSED = (
    "This deployment is read-only: it serves a model trained elsewhere, and uploads, "
    "ingestion and retraining are refused here. Run the API locally to use them."
)
BODY_REFUSED = "This deployment takes no request bodies."


class BoxRequestGuard:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not settings.serving_read_only:
            await self.app(scope, receive, send)
            return
        if scope["method"] not in READ_METHODS:
            await _refuse(send, 409, WRITE_REFUSED)
            return
        headers = dict(scope.get("headers") or ())
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            length = MAX_BODY_BYTES + 1
        if length > MAX_BODY_BYTES or b"transfer-encoding" in headers:
            await _refuse(send, 413, BODY_REFUSED)
            return
        await self.app(scope, receive, send)


async def _refuse(send, status: int, detail: str) -> None:
    # The same {"detail": ...} shape as an HTTPException, which the dashboard's client reads.
    body = json.dumps({"detail": detail}).encode()
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"connection", b"close")],
    })
    await send({"type": "http.response.body", "body": body})
