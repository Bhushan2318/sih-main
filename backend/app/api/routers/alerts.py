from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query, Request, Response

from app.api import schemas
from app.services import alert_service, cap_service, response_cache

router = APIRouter(prefix="/api/alerts", tags=["alerts"])
# /api/alerts.cap is a sibling of /api/alerts, not a path under it, so it has no prefix.
cap_router = APIRouter(tags=["alerts"])


@cap_router.get("/api/alerts.cap", response_class=Response,
                responses={200: {"content": {cap_service.MEDIA_TYPE: {}}}})
def cap_feed():
    """The high-risk alerts as a CAP 1.2 document (status Test: decision support, not an
    issued warning). Built from the same prebuilt body as /api/alerts?risk_band=high."""
    body = response_cache.decoded(
        response_cache.alerts_name("high"),
        lambda: alert_service.get_alerts(limit=response_cache.ALERTS_MAX_LIMIT,
                                         risk_band="high"))
    return Response(cap_service.to_cap(body), media_type=cap_service.MEDIA_TYPE)


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
