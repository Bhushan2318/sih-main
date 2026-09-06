"""geo.py - resolving a region from an id, a name or a coordinate.

The region unit is the district, so a coordinate resolves to one of 666 districts rather
than to one of 36 states. These run against the real committed boundaries.
"""

from __future__ import annotations

import json

import pytest

from app.utils import india_districts, india_state_codes
from app.utils.geo import DISTRICTS_GEOJSON, GeoResolver


@pytest.fixture(scope="module")
def resolver():
    return GeoResolver()


# ------------------------------------------------------------------- coordinates

@pytest.mark.parametrize("name,lat,lon,expected", [
    ("Mumbai", 19.0760, 72.8777, "IN-MH-MUMBAISUBURBAN"),
    ("Kolkata", 22.5726, 88.3639, "IN-WB-KOLKATA"),
    ("Hyderabad", 17.3850, 78.4867, "IN-TG-HYDERABAD"),
    ("Leh", 34.1526, 77.5771, "IN-LA-LEHLADAKH"),
    ("Nagpur", 21.1458, 79.0882, "IN-MH-NAGPUR"),
    ("Srinagar", 34.0837, 74.7973, "IN-JK-SRINAGAR"),
])
def test_point_resolves_to_a_district(resolver, name, lat, lon, expected):
    m = resolver.resolve_point(lat, lon)
    assert m.region_id == expected, name
    assert m.method == "point_in_polygon"
    assert m.is_district


def test_point_carries_its_state(resolver):
    m = resolver.resolve_point(21.1458, 79.0882)
    assert (m.state_id, m.state_name) == ("IN-MH", "Maharashtra")


def test_coastal_offset_falls_back_to_nearest(resolver):
    """A station recorded just offshore must still resolve; refusing would drop real
    observations."""
    m = resolver.resolve_point(19.5, 86.9)
    assert m.region_id is not None
    assert m.state_id == "IN-OR"
    assert m.method == "nearest_polygon"


def test_open_ocean_is_unresolved(resolver):
    m = resolver.resolve_point(5.0, 65.0)
    assert m.region_id is None
    assert m.method == "unresolved"


# --------------------------------------------------------------- names and ids

def test_canonical_id_is_taken_at_its_word(resolver):
    """An upload carrying a region id is telling us the answer. Re-deriving it from a
    centroid would be a chance to get it wrong - and for a district whose centroid falls
    outside its own polygon, it would get it wrong every time."""
    m = resolver.resolve(name="IN-MH-NAGPUR")
    assert m.region_id == "IN-MH-NAGPUR"
    assert m.method == "region_id"


def test_id_beats_a_conflicting_coordinate(resolver):
    m = resolver.resolve(name="IN-MH-NAGPUR", lat=22.5726, lon=88.3639)
    assert m.region_id == "IN-MH-NAGPUR", "the explicit id must win"


def test_district_name_resolves(resolver):
    m = resolver.resolve(name="Nagpur")
    assert m.region_id == "IN-MH-NAGPUR"
    assert m.method == "name"


def test_ambiguous_district_name_does_not_guess(resolver):
    """Aurangabad is in Bihar and in Maharashtra. Picking one would attribute a
    district's weather to another district."""
    m = resolver.resolve(name="Aurangabad")
    assert m.region_id is None or m.method != "name"
    qualified = resolver.resolve(name="Aurangabad", state="Bihar")
    assert qualified.state_id == "IN-BR"


def test_state_name_resolves_at_state_grain(resolver):
    """A state name cannot produce a district. It resolves, but says what it is, so a
    coarser row is visible rather than passing as district-level."""
    m = resolver.resolve(name="Kerala")
    assert m.region_id == "IN-KL"
    assert m.method == "state_name"
    assert not m.is_district


def test_coordinate_beats_a_state_name(resolver):
    """'Maharashtra' plus a coordinate should give the district the coordinate is in,
    not the whole state."""
    m = resolver.resolve(name="Maharashtra", lat=21.1458, lon=79.0882)
    assert m.region_id == "IN-MH-NAGPUR"
    assert m.method == "point_in_polygon"


def test_unknown_name_without_coordinates_is_unresolved(resolver):
    m = resolver.resolve(name="Nowhere In Particular")
    assert m.region_id is None
    assert m.method == "unresolved"


# ------------------------------------------------------------------ consistency

def test_every_polygon_maps_to_a_known_district():
    from app.config import settings
    from app.db.base import resolve_path

    gj = json.loads((resolve_path(settings.geo_dir) / DISTRICTS_GEOJSON).read_text())
    unmapped = [f["properties"]["region_id"] for f in gj["features"]
                if india_districts.resolve_by_id(f["properties"]["region_id"]) is None]
    assert not unmapped, f"geojson features absent from the registry: {unmapped}"


def test_state_code_table_covers_36():
    assert len(india_state_codes.STATES) == 36
