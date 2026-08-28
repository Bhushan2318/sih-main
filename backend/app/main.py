"""ForecastGuard AI - FastAPI application.

    uvicorn app.main:app --reload --port 8000

Startup work is deliberately cheap and safe on a blank machine: create the SQLite tables,
warm the geo index, and note whether a trained model already exists. Nothing here invents
state - with no uploads and no trained model the API answers `model_trained: false` and
the dashboard renders its empty state.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import alerts, model_status, regions, upload, ws
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
    yield


app = FastAPI(
    title="ForecastGuard AI",
    version="0.4.0",
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
app.include_router(ws.router)


@app.get("/api/health", tags=["health"])
def health() -> dict:
    return {
        "status": "ok",
        "model_trained": registry.current_run_id() is not None,
        "current_run_id": registry.current_run_id(),
        "websocket_clients": manager.connection_count,
    }
