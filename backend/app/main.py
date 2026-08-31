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
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
    if registry.current_run_id() and settings.warm_caches_on_startup:
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
    elif registry.current_run_id():
        log.info("startup cache warm disabled; replay and ensemble will be cold on first call")

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


# The commit this container was built from. Render sets RENDER_GIT_COMMIT on every
# deploy; BUILD_COMMIT is the manual escape hatch for anywhere else. It is reported so a
# deploy can be *observed* rather than guessed at: Render auto-deploys on push, which
# bypasses the refresh workflow entirely, so CI has no other way to know the new process
# is the one answering before it warms the caches. Empty when unset, never absent - a
# caller can then fall back to something else rather than having to handle a missing key.
def _build_commit() -> str:
    return os.getenv("RENDER_GIT_COMMIT") or os.getenv("BUILD_COMMIT") or ""


# GET *and* HEAD. FastAPI's @app.get registers GET alone, so a HEAD probe gets a 405 -
# and HEAD is what uptime monitors send by default. UptimeRobot reported this endpoint
# "down" while the app was serving perfectly, because a rejected method looks identical
# to a broken service from the outside.
@app.api_route("/api/health", methods=["GET", "HEAD"], tags=["health"])
def health() -> dict:
    return {
        "status": "ok",
        "model_trained": registry.current_run_id() is not None,
        "current_run_id": registry.current_run_id(),
        "websocket_clients": manager.connection_count,
        "commit": _build_commit(),
    }



# ---------------------------------------------------------------- static SPA (deployed)
# A deployed image builds the frontend and hands the bundle to FastAPI, so the API and the
# dashboard are one origin and one process - no CORS, no second service, no proxy.
#
# This is mounted only when the directory actually exists, so local development is
# untouched: there `npm run dev` serves the SPA on :5173 and this block is skipped
# entirely. Mounting last also matters - the API routers above are already registered, so
# the catch-all below can never shadow /api or /ws.
_STATIC_DIR = Path(__file__).resolve().parent / "static"

if _STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=_STATIC_DIR / "assets"), name="assets")

    @app.api_route("/{full_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        """Serve the SPA, falling back to index.html for client-side routes.

        A real file wins when one exists (favicon, the topojson the map fetches); anything
        else returns the shell so a deep link or a refresh lands in the app rather than on
        a 404.

        The /api and /ws guard is load-bearing. A registered route matches before this
        catch-all, but an *un*registered one does not - without the guard a typo'd or
        retired endpoint would answer 200 with a page of HTML, and the client's
        res.json() would surface it as a JSON parse error instead of a clean 404.
        """
        if full_path.startswith(("api/", "ws")):
            raise HTTPException(status_code=404, detail=f"No such endpoint: /{full_path}")

        # resolve() collapses any ../ before the check, so a crafted path cannot read
        # outside the bundle even though the router hands us the segment verbatim
        candidate = (_STATIC_DIR / full_path).resolve()
        if full_path and candidate.is_file() and candidate.is_relative_to(_STATIC_DIR):
            return FileResponse(candidate)
        return FileResponse(_STATIC_DIR / "index.html")

    log.info("serving built frontend from %s", _STATIC_DIR)
else:
    log.info("no built frontend at %s; API only (use `npm run dev` for the UI)", _STATIC_DIR)
