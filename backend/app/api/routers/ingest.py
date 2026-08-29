"""Live-ingestion status and manual triggers.

The status endpoint is what lets the dashboard state feed freshness honestly: it reports
the last cycle actually ingested and when, so a stale feed reads as a stale feed rather
than as data that happens to be current.
"""

from __future__ import annotations

import threading
from datetime import date
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.config import settings
from app.live import orchestrator
from app.live.scheduler import scheduler

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


@router.get("/status")
def ingest_status() -> dict:
    """Feed health: last cycle, last observation refresh, scheduler state."""
    return {**orchestrator.feed_status(), "scheduler": scheduler.status()}


@router.post("/run-cycle")
def run_cycle(
    background: BackgroundTasks,
    init_date: Optional[date] = Query(None, description="UTC cycle date; defaults to the newest published"),
    cycle_hour: Optional[str] = Query(None, pattern="^(00|06|12|18)$"),
    force: bool = Query(False, description="re-ingest even if this cycle is already stored"),
    wait: bool = Query(False, description="run inline and return the result instead of scheduling it"),
) -> dict:
    """Pull one operational GEFS cycle now ("Refresh now")."""
    if cycle_hour is not None and init_date is None:
        raise HTTPException(400, "cycle_hour requires init_date")

    if wait:
        return orchestrator.run_forecast_cycle(init_date, cycle_hour,
                                               trigger="manual", force=force)
    background.add_task(orchestrator.run_forecast_cycle, init_date, cycle_hour,
                        "manual", force)
    return {"status": "started",
            "target": (orchestrator.cycle_target(init_date, cycle_hour)
                       if init_date and cycle_hour else "newest published cycle"),
            "note": "watch /api/ingest/status or the WebSocket for completion"}


@router.post("/refresh-observations")
def refresh_observations(
    background: BackgroundTasks,
    tier: str = Query("final", pattern="^(final|provisional)$"),
    days_back: Optional[int] = Query(None, ge=1, le=92),
    wait: bool = Query(False),
) -> dict:
    """Pull observations for one verification tier now."""
    if wait:
        return orchestrator.run_observation_refresh(tier, days_back, trigger="manual")
    background.add_task(orchestrator.run_observation_refresh, tier, days_back, "manual")
    return {"status": "started", "tier": tier}


@router.post("/backfill")
def backfill(
    background: BackgroundTasks,
    cycles: int = Query(..., ge=1, le=120, description="how many past cycles to pull"),
    stride_hours: int = Query(24, ge=6, le=168),
) -> dict:
    """Pull past cycles, oldest first. Resumable: already-ingested cycles are skipped.

    Deliberately long-running - each cycle is a few minutes - so it always runs in the
    background and reports through /api/ingest/status.
    """
    threading.Thread(
        target=orchestrator.backfill_cycles, args=(cycles, stride_hours),
        name="live-backfill", daemon=True,
    ).start()
    return {"status": "started", "cycles": cycles, "stride_hours": stride_hours,
            "note": "progress appears in /api/ingest/status; re-running resumes where it stopped"}
