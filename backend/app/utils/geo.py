"""Resolve a region from a name, an identifier, or a coordinate.

The region unit is the district (666 of them), not the state. A district is what an
administration acts on, and at 0.25 deg the forecast grid resolves it - a state read from
one city point made Maharashtra's weather into Mumbai's.

Resolution order, most trustworthy first:

1. **An identifier that is already canonical.** An upload carrying ``IN-MH-NAGPUR`` is
   telling us the answer; re-deriving it from a centroid would be a chance to get it
   wrong, and for a district whose centroid falls outside its own polygon - a crescent
   around a city, an island group - it would get it wrong reliably.
2. **A district name**, optionally qualified by state. Refused when ambiguous: Aurangabad
   is in both Bihar and Maharashtra, and guessing would attribute one district's weather
   to another.
3. **A state name**, which resolves to the state. This is a coarser grain than the rest of
   the store and is reported as such through ``method``, so a caller can see that a row
   is not district-level rather than infer it.
4. **A coordinate**, by point-in-polygon against the districts, then by nearest district
   within ``NEAREST_MAX_DEG`` - which is what lets a coastal station whose coordinate
   falls just offshore still resolve.

Anything else is ``unresolved``. Nothing is guessed.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import NamedTuple

from shapely import make_valid
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

from app.config import settings
from app.db.base import resolve_path
from app.utils import india_districts, india_state_codes

DISTRICTS_GEOJSON = "india_districts.geojson"

# A coordinate this far outside every district still resolves to the nearest one. Coastal
# stations and buoys are recorded just offshore often enough that refusing them would lose
# real observations.
NEAREST_MAX_DEG = 1.5


class RegionMatch(NamedTuple):
    region_id: str | None
    region_name: str | None
    method: str
    state_id: str | None = None
    state_name: str | None = None

    @property
    def is_district(self) -> bool:
        """False for a state-grain match, so a caller can tell the two apart without
        parsing the identifier."""
        return self.region_id is not None and self.method != "state_name"


UNRESOLVED = RegionMatch(None, None, "unresolved")


def _match(rec: india_districts.DistrictRecord, method: str) -> RegionMatch:
    return RegionMatch(rec.region_id, rec.region_name, method,
                       rec.state_id, rec.state_name)


class GeoResolver:
    def __init__(self, geojson_path: Path | None = None):
        path = Path(geojson_path or (resolve_path(settings.geo_dir) / DISTRICTS_GEOJSON))
        data = json.loads(path.read_text())
        self._geoms = []
        self._ids: list[str] = []
        for feat in data["features"]:
            geom = shape(feat["geometry"])
            if not geom.is_valid:
                geom = make_valid(geom).buffer(0)
            self._geoms.append(geom)
            self._ids.append(feat["properties"]["region_id"])
        self._tree = STRtree(self._geoms)

    # ------------------------------------------------------------------ coordinate

    def resolve_point(self, lat: float, lon: float) -> RegionMatch:
        if lat is None or lon is None:
            return UNRESOLVED
        try:
            pt = Point(float(lon), float(lat))
        except (TypeError, ValueError):
            return UNRESOLVED

        for idx in self._tree.query(pt):
            i = int(idx)
            try:
                if self._geoms[i].covers(pt):
                    rec = india_districts.resolve_by_id(self._ids[i])
                    if rec is not None:
                        return _match(rec, "point_in_polygon")
            except Exception:  # noqa: BLE001 - a still-broken polygon shouldn't kill ingest
                continue

        try:
            nearest = int(self._tree.nearest(pt))
            if self._geoms[nearest].distance(pt) <= NEAREST_MAX_DEG:
                rec = india_districts.resolve_by_id(self._ids[nearest])
                if rec is not None:
                    return _match(rec, "nearest_polygon")
        except Exception:  # noqa: BLE001
            pass
        return UNRESOLVED

    # ----------------------------------------------------------------- name and id

    @staticmethod
    def resolve_identifier(value: str | None) -> RegionMatch:
        """An already-canonical region id, district or state."""
        if not value:
            return UNRESOLVED
        key = str(value).strip()
        rec = india_districts.resolve_by_id(key)
        if rec is not None:
            return _match(rec, "region_id")
        st = india_state_codes.resolve_by_region_id(key)
        if st is not None:
            return RegionMatch(st.region_id, st.region_name, "state_name",
                               st.region_id, st.region_name)
        return UNRESOLVED

    def resolve_name(self, name: str | None, state: str | None = None) -> RegionMatch:
        if not name:
            return UNRESOLVED
        by_id = self.resolve_identifier(name)
        if by_id.region_id is not None:
            return by_id

        rec = india_districts.resolve_by_name(name, state=state)
        if rec is not None:
            return _match(rec, "name")

        st = india_state_codes.resolve_by_name(name)
        if st is not None:
            # Coarser than the rest of the store. Say so through `method` rather than
            # letting it pass as a district-grain row.
            return RegionMatch(st.region_id, st.region_name, "state_name",
                               st.region_id, st.region_name)
        return UNRESOLVED

    # ------------------------------------------------------------------- combined

    def resolve(
        self,
        name: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
        state: str | None = None,
    ) -> RegionMatch:
        if name:
            m = self.resolve_name(name, state=state)
            # A state name is coarser than a coordinate would give. Prefer the coordinate
            # when there is one, and keep the state only as the fallback.
            if m.region_id is not None and m.method != "state_name":
                return m
            if m.method == "state_name" and (lat is None or lon is None):
                return m
        if lat is not None and lon is not None:
            by_point = self.resolve_point(lat, lon)
            if by_point.region_id is not None:
                return by_point
        if name:
            return self.resolve_name(name, state=state)
        return UNRESOLVED


@functools.lru_cache(maxsize=1)
def get_resolver() -> GeoResolver:
    return GeoResolver()
