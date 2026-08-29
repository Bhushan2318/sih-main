"""Background scheduler for live ingestion.

A plain daemon thread on a fixed interval rather than APScheduler: the job is a single
periodic call, the process already runs threads for retraining, and this keeps the
dependency list unchanged.

Off unless ``live_ingest_enabled`` is set. A fresh clone, a test run and CI must never
start reaching out to NOAA on import - live ingestion is switched on deliberately.

The work itself runs off the event loop (this is a thread, not a coroutine), so a GRIB
pull that takes minutes cannot block the API.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.config import settings

log = logging.getLogger("forecastguard.live.scheduler")

# How often to wake and ask "is anything due?". Cheap when nothing is: the orchestrator's
# own idempotency check returns immediately for an already-ingested cycle.
TICK_SECONDS = 15 * 60
# Delay before the first tick, so application startup is never held up by a download.
FIRST_TICK_SECONDS = 60


class LiveScheduler:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_tick: Optional[datetime] = None
        self._next_tick: Optional[datetime] = None
        self._ticks = 0
        self._last_result: Optional[dict] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        return {
            "running": self.running,
            "enabled": settings.live_ingest_enabled,
            "tick_seconds": TICK_SECONDS,
            "ticks_completed": self._ticks,
            "last_tick": self._last_tick.isoformat() if self._last_tick else None,
            "next_tick": self._next_tick.isoformat() if self._next_tick else None,
            "last_result": self._last_result,
        }

    def start(self) -> None:
        if not settings.live_ingest_enabled:
            log.info("live ingestion disabled (set LIVE_INGEST_ENABLED=true to turn it on)")
            return
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="live-scheduler", daemon=True)
        self._thread.start()
        log.info("live scheduler started: cycles=%s every %ds",
                 ",".join(settings.gefs_cycle_list), TICK_SECONDS)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        wait = FIRST_TICK_SECONDS
        while not self._stop.wait(wait):
            wait = TICK_SECONDS
            self._next_tick = datetime.now(timezone.utc) + timedelta(seconds=TICK_SECONDS)
            try:
                # Imported here so the module graph stays light when the scheduler is off.
                from app.live.orchestrator import run_due_work
                self._last_result = run_due_work(trigger="schedule")
                self._ticks += 1
            except Exception:  # noqa: BLE001 - a failed tick must not kill the loop
                log.exception("live ingestion tick failed")
                self._last_result = {"status": "failed"}
            self._last_tick = datetime.now(timezone.utc)


scheduler = LiveScheduler()
