from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query, Request

from app.api import schemas
from app.services import alert_service, response_cache

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("", response_model=schemas.AlertsResponse)
@router.get("/", response_model=schemas.AlertsResponse, include_in_schema=False)
def list_alerts(
    request: Request,
    limit: int = Query(50, ge=1, le=response_cache.ALERTS_MAX_LIMIT),
    risk_band: Optional[str] = Query(None, pattern="^(low|medium|high)$"),
):
    # Built once at the API's ceiling and cut to `limit` here. The list is sorted riskiest
    # first before it is cut, so the first `limit` of it is exactly what a build at `limit`
    # returns - and the dashboard's page size is not a second constant the backend must match.
    def _cut(body: dict) -> dict:
        body["alerts"] = body.get("alerts", [])[:limit]
        return body

    return response_cache.respond(
        request, response_cache.alerts_name(risk_band),
        lambda: alert_service.get_alerts(limit=response_cache.ALERTS_MAX_LIMIT,
                                         risk_band=risk_band),
        patch=_cut,
    )
