"""Build the "claimed but not administered" overlay for the national map.

Why this exists
----------------
`build_district_geo.py` already draws every Indian district "to the full extent of
India's claim" - but only where GADM has a district polygon to draw at all, disputed or
not. Gilgit-Baltistan, Aksai Chin and the Shaksgam Valley are not Indian districts under
any GADM classification, because no Indian administration governs them - so GADM has no
polygon there, disputed-prefixed or otherwise, and the rendered map's northern edge stops
at the last GADM district (measured: 35.50 deg N) well short of India's actual claim
(~37.05 deg N, the Siachen/Karakoram Pass area). That gap is what a viewer sees as
Kashmir's tip being cut off.

This script does not touch district_grid_weights.parquet, the district registry, or any
model input - none of that has weather data over territory nobody administers, and
Rule 1 (nothing synthetic) rules out inventing any. It produces a second, clearly
separate map layer: a silhouette-only patch drawn *under* the district choropleth so the
country's outline reads correctly without pretending there is district-level forecast
data where none exists or ever will.

Why it is unioned with J&K and Ladakh, not just the claimed part
---------------------------------------------------------------
Natural Earth and GADM are independent datasets digitised at different scales, so their
shared edge along the Line of Control does not coincide. Shipping only the claimed part
put two mismatched edges side by side and left 21 sliver holes between them (measured),
which rendered as white gashes across Kashmir.

So the output is the union of the claimed polygons *and* the J&K/Ladakh districts as the
browser actually draws them - decoded from the same simplified TopoJSON the map loads,
not from the full-resolution registry, because a simplified edge and a full-resolution
one would reintroduce exactly the mismatch this is removing. Interior rings are then
dropped: every hole in a single fused Kashmir outline is an artifact of that mismatch,
not a real enclave. The districts paint on top of this layer, so their own borders and
colours are unaffected - it only supplies the silhouette underneath.

Source
------
Natural Earth 10m Admin 0 - Breakaway, Disputed Areas (public domain,
https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-breakaway-disputed-areas/).
Each disputed feature carries one ADM0_A3_<ISO2> field per claimant country recording
which country's view includes it; this keeps every feature where ADM0_A3_IN == "IND" -
India's own claim, as tracked by a neutral third-party cartographic reference, not this
project's guess. That selects exactly Aksai Chin, the Shaksgam Valley, Gilgit-Baltistan/
Azad Kashmir (Pakistan- and China-administered, India-claimed) and Siachen Glacier.

    python backend/scripts/build_claimed_territory_geo.py

Downloads the source shapefile fresh each run (small, ~200 KB zipped) rather than vendor
it, so a future Natural Earth correction is picked up by re-running this, not by hand-
editing geometry. Output is committed, like the district files.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import requests

BACKEND_DIR = Path(__file__).resolve().parent.parent
GEO_DIR = BACKEND_DIR.parent / "frontend/src/assets/geo"
OUT_PATH = GEO_DIR / "claimed_territory.geojson"
DISTRICT_TOPOJSON = GEO_DIR / "india_districts.topojson"

SOURCE_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_disputed_areas.zip"
SHP_NAME = "ne_10m_admin_0_disputed_areas.shp"

# The states whose rendered districts are fused into the silhouette - see the module
# docstring on why the union is taken against what the browser draws.
KASHMIR_STATES = ("IN-JK", "IN-LA")

# Simplification tolerance in degrees. This is a background silhouette patch, not
# forecast geometry - 0.005 deg (~500 m) keeps the outline recognisable at map-card scale
# while cutting Natural Earth's 10m-scale vertex count by more than half. Applied to the
# claimed polygons only, never to the fused result: simplifying after the union would
# pull the shared edge back off the districts drawn on top of it.
SIMPLIFY_TOLERANCE = 0.005


def _rendered_state_polygons(states: tuple) -> list:
    """The J&K/Ladakh districts exactly as the map draws them, decoded from the rendered
    TopoJSON: quantised delta-encoded arcs, stitched into rings."""
    from shapely.geometry import shape

    topo = json.loads(DISTRICT_TOPOJSON.read_text())
    sx, sy = topo["transform"]["scale"]
    tx, ty = topo["transform"]["translate"]
    arcs = topo["arcs"]

    def arc_points(index: int) -> list:
        # A negative index means that arc traversed backwards; ~i is its real position.
        raw = arcs[index if index >= 0 else ~index]
        x = y = 0
        pts = []
        for dx, dy in raw:
            x += dx
            y += dy
            pts.append((x * sx + tx, y * sy + ty))
        return pts[::-1] if index < 0 else pts

    def ring(indices: list) -> list:
        pts: list = []
        for i in indices:
            part = arc_points(i)
            pts.extend(part if not pts else part[1:])  # arcs share their end vertices
        return pts

    out = []
    for g in topo["objects"]["districts"]["geometries"]:
        if g["properties"].get("state_id") not in states:
            continue
        if g["type"] == "Polygon":
            gj = {"type": "Polygon", "coordinates": [ring(r) for r in g["arcs"]]}
        elif g["type"] == "MultiPolygon":
            gj = {"type": "MultiPolygon",
                  "coordinates": [[ring(r) for r in poly] for poly in g["arcs"]]}
        else:
            continue
        out.append(shape(gj).buffer(0))  # buffer(0) repairs any self-touching ring
    return out


def main() -> int:
    import shapefile  # pyshp
    from shapely.geometry import MultiPolygon, Polygon, mapping, shape
    from shapely.ops import unary_union

    print(f"downloading {SOURCE_URL}")
    resp = requests.get(SOURCE_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        stem = SHP_NAME[:-4]
        members = {ext: f"{stem}.{ext}" for ext in ("shp", "shx", "dbf")}
        bufs = {ext: io.BytesIO(z.read(name)) for ext, name in members.items()}

    sf = shapefile.Reader(shp=bufs["shp"], shx=bufs["shx"], dbf=bufs["dbf"])

    claimed = []
    for i in range(sf.numRecords):
        rec = sf.record(i).as_dict()
        if rec.get("ADM0_A3_IN") == "IND" and rec.get("ADM0_A3") != "IND":
            claimed.append((rec.get("NAME"), shape(sf.shape(i).__geo_interface__)))

    if not claimed:
        raise SystemExit("no ADM0_A3_IN=IND disputed features found - source format changed?")

    names = ", ".join(sorted({n for n, _ in claimed if n}))
    print(f"{len(claimed)} claimed-not-administered feature(s): {names}")

    claimed_only = unary_union([g for _, g in claimed]).simplify(
        SIMPLIFY_TOLERANCE, preserve_topology=True)

    districts = _rendered_state_polygons(KASHMIR_STATES)
    if not districts:
        raise SystemExit(f"no {'/'.join(KASHMIR_STATES)} districts in {DISTRICT_TOPOJSON.name}")
    print(f"fusing with {len(districts)} rendered {'/'.join(KASHMIR_STATES)} district(s)")

    fused = unary_union([claimed_only, *districts])
    parts = list(fused.geoms) if isinstance(fused, MultiPolygon) else [fused]

    # Drop interior rings. A hole inside one fused Kashmir outline is a sliver where the
    # two datasets' Line-of-Control edges disagree, not an enclave - and each one renders
    # as a white gash, which is the whole reason this union exists.
    holes = sum(len(p.interiors) for p in parts)
    filled = [Polygon(p.exterior) for p in parts]
    merged = filled[0] if len(filled) == 1 else MultiPolygon(filled)
    print(f"union: {len(parts)} part(s), {holes} sliver hole(s) filled")

    fc = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "name": "India's claimed extent in Kashmir, as a silhouette",
                "source": "Natural Earth 10m admin-0 disputed areas (public domain), "
                          "features where ADM0_A3_IN == IND, fused with the rendered "
                          f"{'/'.join(KASHMIR_STATES)} districts",
                "note": "Outline only. Carries no forecast or observation value; the "
                        "districts drawn on top of it do.",
            },
            "geometry": mapping(merged),
        }],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(fc))
    print(f"-> {OUT_PATH}  ({OUT_PATH.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
