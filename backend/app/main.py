"""Sanket - FastAPI application.

    uvicorn app.main:app --reload --port 8000

Startup work is deliberately cheap and safe on a blank machine: create the SQLite tables,
warm the geo index, and note whether a trained model already exists. Nothing here invents
state - with no uploads and no trained model the API answers `model_trained: false` and
the dashboard renders its empty state.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import alerts, ensemble, ingest, model_status, regions, replay, upload, ws
from app.config import settings
from app.db.base import init_db
from app.ml import registry
from app.realtime.broadcaster import manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("forecastguard")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    manager.bind_loop(asyncio.get_running_loop())

    # build the shapely STRtree once, off the request path
    try:
        from app.utils.geo import get_resolver
        get_resolver()
        log.info("geo index ready")
    except Exception:  # noqa: BLE001
        log.exception("geo index failed to build; region resolution will degrade")

    log.info("current model run: %s", registry.current_run_id() or "none (no model trained yet)")

    # Guided replay scores every historical cycle once and memoises the ranking; that is
    # ~1-2 min of XGBoost the first time. Warm it off the request path so the first
    # /api/replay/cycles call is instant. No model -> nothing to warm.
    if registry.current_run_id():
        def _warm_replay() -> None:
            try:
                from app.services import replay_service
                n = len(replay_service.list_cycles())
                log.info("guided-replay cache warm: %d cycles ranked", n)
                # The hero divergence view scores the prior cycle for its delta; warming
                # it here keeps that off the first page load.
                from app.services import ensemble_service
                ensemble_service.get_divergence()
                log.info("ensemble-divergence cache warm")
            except Exception:  # noqa: BLE001
                log.exception("guided-replay warm failed (endpoint still works, just cold)")

        threading.Thread(target=_warm_replay, name="replay-warm", daemon=True).start()

    # Live ingestion is opt-in (LIVE_INGEST_ENABLED); start() is a no-op when it is off,
    # so a fresh clone never reaches out to NOAA just by running the server.
    from app.live.scheduler import scheduler
    scheduler.start()
    try:
        yield
    finally:
        scheduler.stop()


app = FastAPI(
    title="Sanket",
    version="0.6.0",
    lifespan=lifespan,
    description=(
        "Medium-range (Day 1-10) forecast-bust detection for Indian regions. "
        "SIH 2026 PS 26079. Every number served here traces to a real upload and a real "
        "trained model; when neither exists the API says so explicitly."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(regions.router)
app.include_router(alerts.router)
app.include_router(model_status.router)
app.include_router(ingest.router)
app.include_router(replay.router)
app.include_router(ensemble.router)
app.include_router(ws.router)


@app.get("/api/health", tags=["health"])
def health() -> dict:
    return {
        "status": "ok",
        "model_trained": registry.current_run_id() is not None,
        "current_run_id": registry.current_run_id(),
        "websocket_clients": manager.connection_count,
    }
