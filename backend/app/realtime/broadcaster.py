"""WebSocket connection manager + a broadcast helper safe to call from anywhere.

Retraining runs in a FastAPI BackgroundTask (a worker thread, no running event loop), so
`broadcast_threadsafe()` hands the coroutine to the main loop captured at startup. Handler
code inside a request can await `broadcast()` directly.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import WebSocket

from app.realtime.events import Event, EventType, make_event

log = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        self._active: set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = asyncio.Lock()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called once on startup so background threads know where to schedule sends."""
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._active.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._active.discard(ws)

    @property
    def connection_count(self) -> int:
        return len(self._active)

    async def broadcast(self, event: Event) -> None:
        if not self._active:
            return
        message = event.model_dump(mode="json")
        dead = []
        for ws in list(self._active):
            try:
                await ws.send_json(message)
            except Exception:  # noqa: BLE001 - a dropped client must not break the rest
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._active.discard(ws)

    def broadcast_threadsafe(self, event: Event) -> None:
        """Fire-and-forget from a non-async context (BackgroundTask worker thread)."""
        if self._loop is None or not self._active:
            return
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(event), self._loop)
        except RuntimeError:  # loop already closed (shutdown race)
            log.debug("broadcast skipped: event loop unavailable")


manager = ConnectionManager()


def emit(event_type: EventType, **payload) -> None:
    """Thread-safe convenience used by services."""
    manager.broadcast_threadsafe(make_event(event_type, **payload))
