"""C4 - static per-district descriptors, derived from the district geometry Sanket
already has on disk. No new fetch, no rebuild of the 666 districts themselves.

Why this exists
----------------
region_id is a 666-level categorical, but 2017's bust labels cover only 34 of those
districts (CLAUDE.md known limitations). A categorical can only split on a level it saw
in training, so the other 632 districts get nothing useful from region_id at inference.
Continuous, geometry-based descriptors let an unlabelled district borrow strength from
labelled ones that sit nearby, are a similar size, or are a similar distance from the
country's edge - exactly the generalisation region_id cannot provide.

What this reads and writes
---------------------------
Reads the already-built, already-validated ``india_districts.geojson`` and
``india_districts.json`` (see ``scripts/build_district_geo.py`` - not touched here, and
its corrections - Ladakh, disputed-territory dissolution, Delhi's name - are inherited
unchanged because this script never re-derives the district set itself).

Writes ``district_descriptors.parquet``: one row per region_id, columns
``state_id, centroid_lat, centroid_lon, area_km2, border_distance_km``.

The projection
---------------
``area_km2`` and ``border_distance_km`` use a local equirectangular (tangent-plane)
projection - lon scaled by cos(latitude), lat scaled by a constant km-per-degree - not a
geodesic calculation. That is accurate to a fraction of a percent at district scale and
gets worse the more a shape spans latitude; it is a documented approximation, not a claim
of geodetic precision. ``area_km2`` projects each district about its own centroid
latitude (minimising that district's own distortion); ``border_distance_km`` projects
every district about the whole country's mean centroid, because a distance calculation
needs one shared coordinate system, not one per district.

border_distance_km, precisely
------------------------------
Distance from a district's centroid to the nearest edge of the dissolved union of all 666
district polygons - i.e. the modelled landmass's outer boundary. That boundary is the
coastline *and* every international land border (Pakistan, China, Nepal, Bhutan,
Bangladesh, Myanmar) run together: this repo has no separate coastline reference to tell
them apart, so the feature is named for what it actually measures, not for "coastal
distance" it cannot verify. See docs/known-issues.md.

Each district was simplified independently when the boundaries were built (SIMPLIFY_DEG
in build_district_geo.py), so two neighbours' shared edge does not, in general, line up
after simplification - the plain union of all 666 polygons is riddled with sliver gaps
along nearly every internal border, and its "boundary" is mostly simplification noise, not
the country's edge. Found at real volume: a naive union put Jabalpur - several hundred km
from the nearest coast or border on any atlas - 17 km from "the border". A handful of
districts cannot show this; it only appears once all 666 are unioned together.

Fixed by a morphological closing: buffer every district outward by CLOSE_DEG (larger than
SIMPLIFY_DEG), union, then buffer back inward by the same amount. Gaps narrower than
2*CLOSE_DEG are erased; the true macro-scale boundary is preserved to within about
CLOSE_DEG (~1.1 km) - a documented approximation, not exact geodesy.

    python -m scripts.build_district_descriptors
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from shapely import make_valid
from shapely.geometry import Point, shape
from shapely.ops import transform, unary_union

from app.utils import india_districts as idist
from scripts.fetch_grid_elevation import OUT_FILENAME as ELEVATION_FILENAME

KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_AT_EQUATOR = 111.320

OUT_FILENAME = "district_descriptors.parquet"

# Independently-simplified neighbours leave sliver gaps below this scale (see module
# docstring); larger than build_district_geo.py's own SIMPLIFY_DEG (0.004) so the closing
# step is guaranteed to bridge every gap that simplification itself could have created.
CLOSE_DEG = 0.01


def _project(lon0: float, lat0: float):
    """A local equirectangular projection centred at (lon0, lat0), in km."""
    cos_lat0 = np.cos(np.radians(lat0))

    def fn(lon, lat):
        x = (np.asarray(lon, dtype=float) - lon0) * KM_PER_DEG_LON_AT_EQUATOR * cos_lat0
        y = (np.asarray(lat, dtype=float) - lat0) * KM_PER_DEG_LAT
        return x, y

    return fn


def _polygon_area_km2(geom, lat0: float, lon0: float) -> float:
    return float(transform(_project(lon0, lat0), geom).area)


def _boundary_distance_km(point_lon: float, point_lat: float, boundary_geom,
                          lat0: float, lon0: float) -> float:
    proj = _project(lon0, lat0)
    p = transform(proj, Point(point_lon, point_lat))
    b = transform(proj, boundary_geom)
    return float(p.distance(b))


def _elevation_by_district(elevation: pd.DataFrame, aggregator=None) -> pd.Series:
    """Area-weighted mean elevation per district, via the SAME DistrictGridAggregator
    GEFS/ERA5 aggregation already uses (CLAUDE.md: one weight table) - not a second
    spatial join. `elevation` is scripts/fetch_grid_elevation.py's output: one row per
    weight-table cell, real values fetched from a real elevation source."""
    agg = aggregator or idist.get_aggregator()
    return agg.aggregate(elevation["lat"], elevation["lon"], elevation["elevation_m"])


def _district_geometries() -> dict:
    """region_id -> unified Shapely geometry. A district can be more than one GeoJSON
    feature (disputed-territory dissolution in build_district_geo.py), so features
    sharing a region_id are unioned here rather than assumed to be one-to-one."""
    geo = json.loads((idist.geo_dir() / idist.GEOJSON_FILENAME).read_text())
    geoms: dict = {}
    for feat in geo["features"]:
        rid = feat["properties"]["region_id"]
        g = shape(feat["geometry"])
        if not g.is_valid:
            g = make_valid(g).buffer(0)
        geoms[rid] = g if rid not in geoms else geoms[rid].union(g)
    return geoms


def build() -> pd.DataFrame:
    registry = {d.region_id: d for d in idist.load_registry()}
    geoms = _district_geometries()

    india_lat0 = float(np.mean([d.centroid_lat for d in registry.values()]))
    india_lon0 = float(np.mean([d.centroid_lon for d in registry.values()]))
    # Morphological closing (dilate, union, erode) - see module docstring - so
    # simplification slivers between neighbours do not masquerade as the country's edge.
    closed = unary_union([g.buffer(CLOSE_DEG) for g in geoms.values()]).buffer(-CLOSE_DEG)
    boundary = closed.boundary

    elevation_path = idist.geo_dir() / ELEVATION_FILENAME
    if not elevation_path.exists():
        raise RuntimeError(
            f"{elevation_path} does not exist - run "
            "`python -m scripts.fetch_grid_elevation` first (CLAUDE.md: refuse rather "
            "than fabricate a missing value).")
    elevation_by_district = _elevation_by_district(pd.read_parquet(elevation_path))

    rows = []
    for rid, geom in geoms.items():
        d = registry.get(rid)
        if d is None:
            continue
        rows.append({
            "region_id": rid,
            "state_id": d.state_id,
            "centroid_lat": d.centroid_lat,
            "centroid_lon": d.centroid_lon,
            "area_km2": _polygon_area_km2(geom, d.centroid_lat, d.centroid_lon),
            "border_distance_km": _boundary_distance_km(
                d.centroid_lon, d.centroid_lat, boundary, india_lat0, india_lon0),
            "elevation_mean": elevation_by_district.get(rid, np.nan),
        })
    missing = set(registry) - set(geoms)
    if missing:
        raise RuntimeError(f"{len(missing)} registry districts have no geometry: "
                          f"{sorted(missing)[:5]}...")
    return pd.DataFrame(rows).sort_values("region_id", ignore_index=True)


def main() -> None:
    out = build()
    path = idist.geo_dir() / OUT_FILENAME
    out.to_parquet(path, index=False)
    print(f"{len(out)} districts -> {path} ({path.stat().st_size / 1024:.1f} KB)")
    print(out[["area_km2", "border_distance_km", "elevation_mean"]].describe().to_string())


if __name__ == "__main__":
    main()
