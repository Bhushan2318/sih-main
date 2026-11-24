"""The CAP 1.2 feed: /api/alerts.cap.

The document is validated against the OASIS CAP 1.2 schema itself
(tests/data/CAP-v1.2.xsd, fetched from
https://docs.oasis-open.org/emergency/cap/v1.2/CAP-v1.2.xsd on 2026-11-02, 10,098 bytes).

The alert dicts below are plumbing inputs in the shape /api/alerts serves: the region ids
are real districts from data/geo/india_districts.json, the probabilities are made up and
labelled so, and nothing here produces or checks a metric.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

xmlschema = pytest.importorskip("xmlschema")

from app.services import cap_service  # noqa: E402
from app.utils import india_districts  # noqa: E402

XSD = Path(__file__).parent / "data" / "CAP-v1.2.xsd"
NS = {"cap": "urn:oasis:names:tc:emergency:cap:1.2"}


def _real_ids(n: int) -> list:
    return [d.region_id for d in india_districts.load_registry()[:n]]


def _body(alerts: list, run_id: str = "run_20261006T200800Z") -> dict:
    return {
        "generated_at": "2026-11-02T04:12:09.123456Z",
        "model_trained": True,
        "alerts": [
            {"region_id": rid, "region_name": f"name {i}", "lead_time_days": 3,
             "valid_date": "2026-11-04", "bust_probability": p,  # plumbing values
             "risk_band": "high", "dominant_variable": "rainfall_mm",
             "training_run_id": run_id}
            for i, (rid, p) in enumerate(alerts)
        ],
    }


def test_the_document_validates_against_the_oasis_schema():
    ids = _real_ids(3)
    xml = cap_service.to_cap(_body([(ids[0], 0.91), (ids[1], 0.85), (ids[2], 0.42)]))
    xmlschema.XMLSchema(str(XSD)).validate(xml.decode())


def test_an_empty_band_is_still_a_valid_document_with_no_info():
    xml = cap_service.to_cap(_body([]))
    xmlschema.XMLSchema(str(XSD)).validate(xml.decode())
    assert ET.fromstring(xml).findall("cap:info", NS) == []


def test_it_never_claims_to_be_an_actual_warning():
    root = ET.fromstring(cap_service.to_cap(_body([(_real_ids(1)[0], 0.9)])))
    assert root.find("cap:status", NS).text == "Test"
    assert "not an official warning" in root.find("cap:note", NS).text


def test_each_area_is_a_closed_ring_in_lat_lon_order_around_the_district():
    rid = _real_ids(1)[0]
    rec = india_districts.resolve_by_id(rid)
    root = ET.fromstring(cap_service.to_cap(_body([(rid, 0.9)])))
    ring = root.find("cap:info/cap:area/cap:polygon", NS).text.split()
    assert len(ring) >= 4 and ring[0] == ring[-1]
    lats = [float(p.split(",")[0]) for p in ring]
    lons = [float(p.split(",")[1]) for p in ring]
    # lat,lon - not GeoJSON's lon,lat - and the centroid falls inside the box.
    assert min(lats) <= rec.centroid_lat <= max(lats)
    assert min(lons) <= rec.centroid_lon <= max(lons)


def test_the_validity_window_is_the_utc_day_the_lead_verifies_on():
    # valid_date = init + (lead - 1); day k is forecast hours ((k-1)*24, k*24] from 00 UTC.
    root = ET.fromstring(cap_service.to_cap(_body([(_real_ids(1)[0], 0.9)])))
    info = root.find("cap:info", NS)
    assert info.find("cap:onset", NS).text == "2026-11-04T00:00:00+00:00"
    assert info.find("cap:expires", NS).text == "2026-11-05T00:00:00+00:00"


@pytest.mark.parametrize("p, want", [(0.91, "Likely"), (0.51, "Likely"), (0.5, "Possible")])
def test_certainty_follows_the_cap_definitions(p, want):
    # CAP 1.2 3.2.2: Likely is "p > ~50%", Possible is "p <= ~50%".
    root = ET.fromstring(cap_service.to_cap(_body([(_real_ids(1)[0], p)])))
    assert root.find("cap:info/cap:certainty", NS).text == want


def test_an_untrained_server_says_so_instead_of_inventing_alerts():
    xml = cap_service.to_cap({"generated_at": "2026-11-02T04:12:09Z", "model_trained": False,
                              "alerts": []})
    xmlschema.XMLSchema(str(XSD)).validate(xml.decode())
    assert "no model" in ET.fromstring(xml).find("cap:note", NS).text.lower()
