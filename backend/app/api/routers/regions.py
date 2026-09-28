from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from app.api import schemas
from app.config import settings
from app.services import region_service, response_cache
from app.utils import india_districts, india_state_codes

router = APIRouter(prefix="/api/regions", tags=["regions"])


@router.get("", response_model=schemas.RegionsResponse)
@router.get("/", response_model=schemas.RegionsResponse, include_in_schema=False)
def list_regions(
    request: Request,
    lead_time_days: int = Query(1, ge=1, le=10, description="forecast lead day, 1-10"),
):
    if settings.serving_read_only:
        # The dashboard reads /all. On the box one day is cut from that response, which CI
        # built, rather than built again: get_all_regions makes the same per-day call.
        return response_cache.respond(
            request, "regions_all", region_service.get_all_regions,
            patch=lambda body: next(
                d for d in body["days"] if d["lead_time_days"] == lead_time_days))
    return response_cache.respond(
        request, None, lambda: region_service.get_regions(lead_time_days))


@router.get("/all", response_model=schemas.AllRegionsResponse)
def list_regions_all_lead_days(request: Request):
    return response_cache.respond(request, "regions_all", lambda: region_service.get_all_regions())


@router.get("/{region_id}", response_model=schemas.RegionDetailResponse)
def region_detail(request: Request, region_id: str):
    district = india_districts.resolve_by_id(region_id) is not None
    if not district and india_state_codes.resolve_by_region_id(region_id) is None:
        raise HTTPException(
            404,
            f"unknown region_id '{region_id}' (expected a district such as "
            f"IN-MH-NAGPUR, or a state such as IN-MH)",
        )
    if not district and settings.serving_read_only:
        # CI builds district panels only, and the dashboard never opens a state's (a click
        # on a state drills the map into it). Built live, one loaded the model on the box.
        raise HTTPException(
            404, f"'{region_id}' is a state; this deployment serves district panels only.")
    # Every district of the latest cycle is built in CI (services/response_cache).
    return response_cache.respond(
        request, response_cache.region_name(region_id),
        lambda: region_service.get_region_detail(region_id),
    )
