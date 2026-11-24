"""The high-risk alerts as a CAP 1.2 document (OASIS Common Alerting Protocol).

CAP is the format alert aggregators read; NDMA's SACHET is one. This turns the body
/api/alerts already serves for the high band into CAP, so on the box it reads the prebuilt
response and scores nothing.

Two deliberate choices:

- `status` is Test, never Actual. Sanket is decision support for a forecaster, not a
  warning authority, and an aggregator treats Actual as an issued warning.
- Each `<area>` is the district's bounding box from india_districts.json, not its outline.
  The box covers more than the district; `<geocode>` carries the exact district id.
"""
from __future__ import annotations

import datetime as dt
import xml.etree.ElementTree as ET
from typing import Optional

from app.services.ensemble_service import _label
from app.utils import india_districts

CAP_NS = "urn:oasis:names:tc:emergency:cap:1.2"
SITE = "https://sanket-a0dd.onrender.com"
SENDER = "sanket-a0dd.onrender.com"
MEDIA_TYPE = "application/cap+xml"

NOTE = ("Decision support, not an official warning: the probability that the forecast "
        "for this district and day is a bust, from a model, for a forecaster to weigh. "
        "Issued warnings come from IMD.")
NO_MODEL_NOTE = "No model is loaded on this server, so there are no alerts to report."


def _cap_time(value) -> str:
    """CAP's dateTime: seconds, an explicit offset, no fraction and no 'Z'."""
    t = value if isinstance(value, dt.datetime) else dt.datetime.fromisoformat(
        str(value).replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat()


def _day_start(day) -> dt.datetime:
    d = day if isinstance(day, dt.date) else dt.date.fromisoformat(str(day))
    return dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc)


def _certainty(p: float) -> str:
    # CAP 1.2 section 3.2.2: Likely "p > ~50%", Possible "p <= ~50%".
    return "Likely" if p > 0.5 else "Possible"


def _ring(bbox) -> str:
    lo_lon, lo_lat, hi_lon, hi_lat = bbox
    pts = [(lo_lat, lo_lon), (hi_lat, lo_lon), (hi_lat, hi_lon), (lo_lat, hi_lon), (lo_lat, lo_lon)]
    return " ".join(f"{la:.4f},{lo:.4f}" for la, lo in pts)


def _sub(parent: ET.Element, tag: str, text: Optional[str] = None) -> ET.Element:
    el = ET.SubElement(parent, f"{{{CAP_NS}}}{tag}")
    if text is not None:
        el.text = text
    return el


def _info(root: ET.Element, a: dict) -> None:
    p = float(a["bust_probability"])
    pct = round(100 * p)
    name = a.get("region_name") or a["region_id"]
    lead = int(a["lead_time_days"])
    var = _label(a.get("dominant_variable") or "")

    info = _sub(root, "info")
    _sub(info, "language", "en-IN")
    _sub(info, "category", "Met")
    _sub(info, "event", "Forecast bust risk")
    _sub(info, "responseType", "Monitor")
    _sub(info, "urgency", "Future")
    _sub(info, "severity", "Unknown")
    _sub(info, "certainty", _certainty(p))
    if a.get("valid_date"):
        start = _day_start(a["valid_date"])
        _sub(info, "onset", _cap_time(start))
        _sub(info, "expires", _cap_time(start + dt.timedelta(days=1)))
    _sub(info, "senderName", "Sanket")
    _sub(info, "headline", f"{pct}% chance the Day {lead} forecast for {name} is a bust")
    _sub(info, "description", (
        f"Sanket estimates a {pct}% chance that the GEFS forecast for {name} on "
        f"{a.get('valid_date') or 'its valid day'} (Day {lead}) is a bust: an error in the top "
        f"10% of that variable's historical forecast errors."
        + (f" The variable most likely to bust is {var}." if var else "")))
    _sub(info, "web", SITE)
    for key in ("bust_probability", "lead_time_days", "training_run_id"):
        if a.get(key) is not None:
            prm = _sub(info, "parameter")
            _sub(prm, "valueName", key)
            _sub(prm, "value", str(a[key]))

    area = _sub(info, "area")
    _sub(area, "areaDesc", name)
    rec = india_districts.resolve_by_id(a["region_id"])
    if rec is not None and rec.bbox is not None:
        _sub(area, "polygon", _ring(rec.bbox))
    geo = _sub(area, "geocode")
    _sub(geo, "valueName", "SANKET_REGION_ID")
    _sub(geo, "value", str(a["region_id"]))


def to_cap(body: dict) -> bytes:
    """A CAP 1.2 <alert> for an /api/alerts body (a dict, as the cache decodes it)."""
    ET.register_namespace("", CAP_NS)
    alerts = body.get("alerts") or []
    runs = {a.get("training_run_id") for a in alerts if a.get("training_run_id")}
    sent = _cap_time(body["generated_at"])

    root = ET.Element(f"{{{CAP_NS}}}alert")
    stamp = sent.replace("-", "").replace(":", "").replace("+0000", "Z")
    _sub(root, "identifier", "sanket-" + "-".join(sorted(runs) + [stamp]))
    _sub(root, "sender", SENDER)
    _sub(root, "sent", sent)
    _sub(root, "status", "Test")
    _sub(root, "msgType", "Alert")
    _sub(root, "scope", "Public")
    _sub(root, "note", NOTE if body.get("model_trained") else NO_MODEL_NOTE)
    for a in alerts:
        _info(root, a)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)
