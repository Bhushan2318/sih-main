from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, Query, Request

from app.api import schemas
from app.services import ensemble_service, response_cache

router = APIRouter(prefix="/api/ensemble", tags=["ensemble"])


@router.get("", response_model=schemas.EnsembleDivergenceResponse)
@router.get("/", response_model=schemas.EnsembleDivergenceResponse, include_in_schema=False)
def divergence(
    request: Request,
    init_date: Optional[date] = Query(None, description="Forecast cycle; defaults to the latest scored."),
    region_id: Optional[str] = Query(None, description="Prefer this region if it is chartable."),
):
    # The hero asks with no parameters; that answer is built in CI (services/response_cache).
    name = "ensemble" if init_date is None and region_id is None else None
    return response_cache.respond(
        request, name,
        lambda: ensemble_service.get_divergence(init_date=init_date, region_id=region_id),
    )
