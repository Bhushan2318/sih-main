"""Guided replay endpoints.

Step through one real historical forecast cycle lead day by lead day and see how a bust
developed. Everything served here is a scored number from a cycle that is actually in
the canonical store, or a sentence generated from those numbers.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from app.api import schemas
from app.services import replay_service

router = APIRouter(prefix="/api/replay", tags=["replay"])


@router.get("/cycles", response_model=list[schemas.ReplayCycleSummary])
def replay_cycles() -> list[schemas.ReplayCycleSummary]:
    """Every scoreable forecast cycle, most demo-worthy (real verified bust) first."""
    return replay_service.list_cycles()


@router.get("", response_model=schemas.ReplayResponse)
@router.get("/", response_model=schemas.ReplayResponse, include_in_schema=False)
def replay(
    init_date: Optional[str] = Query(
        None, description="cycle init date, YYYY-MM-DD; omit for the most demo-worthy cycle"
    ),
    focus_region: Optional[str] = Query(
        None, description="region_id to chart; omit for the cycle's peak-bust-risk region"
    ),
) -> schemas.ReplayResponse:
    return replay_service.get_replay(init_date, focus_region)
