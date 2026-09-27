from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, Query, Request

from app.api import schemas
from app.services import replay_service, response_cache

router = APIRouter(prefix="/api/replay", tags=["replay"])


@router.get("/cycles", response_model=list[schemas.ReplayCycleSummary])
def replay_cycles(request: Request):
    return response_cache.respond(request, "replay_cycles", lambda: replay_service.list_cycles())


@router.get("", response_model=schemas.ReplayResponse)
@router.get("/", response_model=schemas.ReplayResponse, include_in_schema=False)
def replay(
    request: Request,
    # A date, not a string: a malformed one is a 422 here, not a 500 out of pandas.
    init_date: Optional[date] = Query(
        None, description="cycle init date, YYYY-MM-DD; omit for the most demo-worthy cycle"
    ),
    focus_region: Optional[str] = Query(
        None, description="region_id to chart; omit for the cycle's peak-bust-risk region"
    ),
):
    # Every cycle Replay lists is built in CI with its default focus (services/response_cache).
    day = init_date.isoformat() if init_date else None
    name = response_cache.replay_name(day) if focus_region is None else None
    return response_cache.respond(
        request, name, lambda: replay_service.get_replay(day, focus_region),
    )
