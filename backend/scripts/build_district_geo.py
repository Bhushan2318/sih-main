"""Build Sanket's district geography from a source admin-2 boundary file.

Why districts
-------------
Until now a "region" here was one of 35 states, and each state's weather was read from a
single city point - so Maharashtra's forecast was Mumbai's forecast. A district is the
unit an administration actually acts on, and at 0.25 deg (~28 km) the GEFS grid has
enough resolution to support it: 676 districts, a median of 14 grid cells each.

Nothing about the download changes. The fetch already pulls whole global GRIB fields and
throws away all but a handful of points; this just says which cells belong to which
district, so the same bytes yield 676 regional averages instead of 36 city readings.

What it writes (all into backend/data/geo/, all tracked in git)
--------------------------------------------------------------
  india_districts.json            the registry: region_id, name, state, area, bbox
  india_districts.geojson         simplified boundaries, for resolving uploaded lat/lon
  district_grid_weights.parquet   (region_id, lat, lon, weight) on the 0.25 deg grid

Area-overlap, not nearest-point
-------------------------------
Each district's value is the area-weighted mean of every 0.25 deg cell its polygon
overlaps. Sampling the nearest grid point instead would leave 10 districts - Kolkata,
Hyderabad, the Puducherry enclaves, Lakshadweep - with no grid centre inside them at all.
Area overlap gives every one of the 676 a defined value, and gives large districts a
genuine spatial mean rather than one arbitrary point.

Source
------
GADM 4.1 India level-2 (https://gadm.org), which is current enough to carry Telangana as
its own state. Two corrections are applied to it, both deliberate:

*Ladakh.* GADM predates the 2019 reorganisation, so Leh and Kargil still sit under Jammu
and Kashmir. They are reassigned - see LADAKH_DISTRICTS.

*Disputed territory.* GADM prefixes a feature's GID_2 with ``Z01``/``Z04``/``Z05``/
``Z07``/``Z09`` where a boundary is internationally disputed, and files it separately
from the ``IND.*`` feature for the same district. All 22 Jammu and Kashmir districts and
both Ladakh districts are ``Z01`` with *no* ``IND.*`` counterpart, and ten Arunachal
Pradesh districts are ``Z07``. Filtering to ``IND.*`` would therefore erase Jammu and
Kashmir, Ladakh and much of Arunachal Pradesh from the map. Every feature is kept
regardless of prefix, and where a district appears under both prefixes the two polygons
are dissolved into one - so each district is one region, drawn to the full extent of
India's claim.

Because every value is aggregated from a grid, the boundary vintage is a presentation
choice and not a provenance one: current districts can be aggregated from 2000-2019
forecasts without any claim that those districts existed then.

    python backend/scripts/build_district_geo.py          # downloads the source
    python backend/scripts/build_district_geo.py --source gadm41_IND_2.json

The generated files are committed, so this only needs re-running when the boundaries
themselves change. It is not on the ingest path.

Building the map's TopoJSON
---------------------------
The dashboard needs a small, topology-aware version. Simplifying each polygon
independently - which is what ``india_districts.geojson`` above does, and is fine for
point lookup - breaks the vertices neighbours share, so a simplifier can no longer tell
that two districts have a common border and draws slivers between them. Hence
``--full-geojson`` and mapshaper:

    python backend/scripts/build_district_geo.py --full-geojson /tmp/districts_full.geojson
    npx mapshaper@0.6 /tmp/districts_full.geojson \
        -simplify percentage=12% keep-shapes planar \
        -clean \
        -rename-layers districts \
        -o format=topojson frontend/src/assets/geo/india_districts.topojson

666 districts land at 374 KB (111 KB gzipped) - smaller than the 36-state file it sits
beside, which was never simplified. ``-clean`` matters: it drops 421 sliver polygons left
between neighbours, and was checked to keep all 666 districts with no empty geometry.
``percentage=25%`` is not better - it leaves *more* unrepaired self-intersections (526
against 288), because those come from the source boundaries rather than from simplifying.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import make_valid
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.utils import india_state_codes  # noqa: E402

GEO_DIR = BACKEND_DIR / "data" / "geo"
SOURCE_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_2.json.zip"

GRID_DEG = 0.25          # GEFS / ERA5 native resolution
SIMPLIFY_DEG = 0.004     # ~440 m; boundaries are stored for point lookup, not for area
MIN_WEIGHT = 1e-9        # drop numerically-zero slivers

# GADM 4.1 predates the 2019 reorganisation, so Ladakh's districts are filed under
# Jammu and Kashmir. Named exactly as GADM spells them.
LADAKH_DISTRICTS = {"Leh(Ladakh)", "Kargil"}

# (normalised state, normalised source name) -> the name to publish.
#
# GADM carries the NCT of Delhi as a single admin-2 unit but labels it "West". The
# geometry is right - it spans the whole territory, 1503 km2 against the true 1484, from
# Narela in the north to Badarpur in the south - so only the label is wrong. Delhi is
# ~1.5 grid cells at 0.25 deg, so one unit is also the honest resolution to report it at;
# splitting it into its eleven real districts would invent detail the data cannot carry.
NAME_CORRECTIONS = {("nctofdelhi", "west"): "Delhi"}


def norm(s: str) -> str:
    """GADM writes state names unspaced ('AndhraPradesh'); the registry writes them
    spaced. Comparing on alphanumerics only lets both spellings meet."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


STATE_BY_NORM = {}
for _s in india_state_codes.STATES:
    STATE_BY_NORM[norm(_s.region_name)] = _s
    for _a in _s.aliases:
        STATE_BY_NORM[norm(_a)] = _s


def clean_district_name(raw: str) -> str:
    """GADM stores names unspaced - 'MumbaiSuburban', 'North24Parganas',
    'NorthandMiddleAndaman', 'Leh(Ladakh)'. Restore the spaces, because these strings are
    displayed to a user on every panel of the dashboard.

    Order matters: the joiner words go in before the camel-case split, or
    'NorthandMiddleAndaman' becomes 'Northand Middle Andaman'.
    """
    name = (raw or "").strip()
    name = re.sub(r"\s*&\s*", " & ", name)
    name = re.sub(r"(?<=[a-z])(and|of)(?=[A-Z])", r" \1 ", name)   # Dadraand -> Dadra and
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)               # MumbaiSuburban
    name = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", name)  # North24Parganas
    name = re.sub(r"\s*\(\s*", " (", name)
    name = re.sub(r"\s*\)", ")", name)
    name = re.sub(r"-([a-z])", lambda m: "-" + m.group(1).upper(), name)  # Saraikela-kharsawan
    return re.sub(r"\s+", " ", name).strip()


def slug(name: str) -> str:
    """Not truncated: cutting at a fixed width produced IN-AS-KAMRUPMETROPOLIT, which
    reads as a bug every time it appears in a log line or an API response."""
    return re.sub(r"[^A-Z0-9]", "", name.upper()) or "UNNAMED"


def fetch_source() -> Path:
    """Pull the GADM archive into a temp dir. Kept out of the repo: it is 4.9 MB of input
    that only this script reads, and what matters downstream is the generated output."""
    import tempfile
    import urllib.request
    import zipfile

    dest = Path(tempfile.gettempdir()) / "gadm41_IND_2.json"
    if dest.exists():
        return dest
    print(f"downloading {SOURCE_URL}")
    zip_path = dest.with_suffix(".zip")
    urllib.request.urlretrieve(SOURCE_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith("_2.json"))
        dest.write_bytes(zf.read(name))
    zip_path.unlink(missing_ok=True)
    return dest


def load_districts(source: Path) -> list[dict]:
    """Source features -> records with a resolved state and a stable region_id.

    region_id is IN-<state>-<district slug>, e.g. IN-MH-NAGPUR. It is derived from names
    rather than from GADM's GID_2 ('IND.16.12_1') because GID_2 is re-numbered between
    GADM releases, and a region_id that changes under a data refresh would silently
    orphan every stored row keyed to it.
    """
    data = json.loads(source.read_text())

    # Dissolve first: one district can arrive as several features (an IND.* polygon plus
    # a Z07.* disputed-area polygon, or three separate Kinnaur pieces). Keyed on
    # (state, district name) because GID_2 is exactly what differs between them.
    merged: dict[tuple[str, str], dict] = {}
    for feat in data["features"]:
        props = feat["properties"]
        key = (norm(props.get("NAME_1", "")), norm(props.get("NAME_2", "")))
        geom = shape(feat["geometry"])
        if not geom.is_valid:
            geom = make_valid(geom).buffer(0)
        if key in merged:
            merged[key]["parts"].append(geom)
        else:
            merged[key] = {"props": props, "parts": [geom]}

    records, seen = [], {}
    for entry in merged.values():
        props = entry["props"]
        raw_state, raw_name = props.get("NAME_1", ""), props.get("NAME_2", "")
        name = NAME_CORRECTIONS.get((norm(raw_state), norm(raw_name))) \
            or clean_district_name(raw_name)

        state = STATE_BY_NORM.get(norm(raw_state))
        if state is None:
            raise SystemExit(f"unresolved state {raw_state!r} for district {name!r}")
        if raw_name in LADAKH_DISTRICTS:
            state = STATE_BY_NORM[norm("Ladakh")]

        region_id = f"{state.region_id}-{slug(name)}"
        if region_id in seen:  # same district name twice in one state
            seen[region_id] += 1
            region_id = f"{region_id}-{seen[region_id]}"
        else:
            seen[region_id] = 1

        parts = entry["parts"]
        geom = parts[0] if len(parts) == 1 else unary_union(parts)
        if not geom.is_valid:
            geom = make_valid(geom).buffer(0)
        records.append(
            {
                "region_id": region_id,
                "region_name": name,
                "state_id": state.region_id,
                "state_name": state.region_name,
                "n_source_features": len(parts),
                "geometry": geom,
            }
        )
    return records


def grid_weights(records: list[dict]) -> pd.DataFrame:
    """Area-overlap weight of every 0.25 deg cell against every district.

    Weights are computed on the FULL-resolution geometry - simplification happens
    afterwards and only to what gets stored. Each district's weights sum to 1, so
    aggregation is a plain weighted mean and a district that straddles the coast is not
    penalised for the cells that fall in the sea.
    """
    geoms = [r["geometry"] for r in records]
    tree = STRtree(geoms)

    minx = min(g.bounds[0] for g in geoms)
    maxx = max(g.bounds[2] for g in geoms)
    miny = min(g.bounds[1] for g in geoms)
    maxy = max(g.bounds[3] for g in geoms)
    half = GRID_DEG / 2
    lons = np.arange(np.floor(minx / GRID_DEG) * GRID_DEG, maxx + GRID_DEG, GRID_DEG)
    lats = np.arange(np.floor(miny / GRID_DEG) * GRID_DEG, maxy + GRID_DEG, GRID_DEG)

    rows = []
    for lon in lons:
        for lat in lats:
            cell = box(lon - half, lat - half, lon + half, lat + half)
            for i in tree.query(cell):
                i = int(i)
                inter = geoms[i].intersection(cell)
                if inter.is_empty or inter.area <= MIN_WEIGHT:
                    continue
                rows.append((records[i]["region_id"], round(float(lat), 4),
                             round(float(lon), 4), float(inter.area)))

    df = pd.DataFrame(rows, columns=["region_id", "lat", "lon", "weight"])
    df["weight"] /= df.groupby("region_id")["weight"].transform("sum")
    return df.sort_values(["region_id", "lat", "lon"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=None,
                    help=f"GADM 4.1 India level-2 GeoJSON. Downloaded from {SOURCE_URL} "
                         "if not given.")
    ap.add_argument("--out-dir", type=Path, default=GEO_DIR)
    ap.add_argument("--full-geojson", type=Path, default=None,
                    help="also write unsimplified boundaries here. Input for building the "
                         "map's TopoJSON: simplifying each polygon on its own breaks the "
                         "vertices neighbours share, so a topology-aware simplifier can no "
                         "longer tell that two districts have a common border and draws "
                         "slivers between them.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = args.source or fetch_source()
    if not source.exists():
        raise SystemExit(f"source not found: {source}")

    print(f"reading {source}")
    records = load_districts(source)
    dissolved = sum(1 for r in records if r["n_source_features"] > 1)
    print(f"  {len(records)} districts across "
          f"{len({r['state_id'] for r in records})} states/UTs"
          f"  ({dissolved} dissolved from multiple source features)")

    print("computing 0.25 deg area-overlap weights (takes a minute)")
    weights = grid_weights(records)
    per_district = weights.groupby("region_id").size()
    orphans = sorted({r["region_id"] for r in records} - set(per_district.index))
    if orphans:
        raise SystemExit(f"districts with no overlapping grid cell: {orphans}")
    print(f"  {len(weights)} (district, cell) pairs over "
          f"{weights[['lat', 'lon']].drop_duplicates().shape[0]} distinct cells")
    print(f"  cells per district: min {per_district.min()} "
          f"median {int(per_district.median())} max {per_district.max()}")

    registry = []
    features = []
    for r in records:
        geom = r["geometry"]
        simple = geom.simplify(SIMPLIFY_DEG, preserve_topology=True)
        if simple.is_empty:
            simple = geom
        b = geom.bounds
        registry.append({
            "region_id": r["region_id"],
            "region_name": r["region_name"],
            "state_id": r["state_id"],
            "state_name": r["state_name"],
            "bbox": [round(v, 4) for v in b],
            "centroid": [round(geom.centroid.y, 4), round(geom.centroid.x, 4)],
        })
        features.append({
            "type": "Feature",
            "properties": {"region_id": r["region_id"], "region_name": r["region_name"],
                           "state_id": r["state_id"]},
            "geometry": mapping(simple),
        })

    reg_path = args.out_dir / "india_districts.json"
    geo_path = args.out_dir / "india_districts.geojson"
    w_path = args.out_dir / "district_grid_weights.parquet"

    if args.full_geojson:
        full = {"type": "FeatureCollection", "features": [
            {"type": "Feature",
             "properties": {"region_id": r["region_id"], "region_name": r["region_name"],
                            "state_id": r["state_id"], "state_name": r["state_name"]},
             "geometry": mapping(r["geometry"])}
            for r in records]}
        args.full_geojson.parent.mkdir(parents=True, exist_ok=True)
        args.full_geojson.write_text(json.dumps(full, ensure_ascii=False))
        print(f"wrote {args.full_geojson}  "
              f"({args.full_geojson.stat().st_size/1e6:.1f} MB, unsimplified)")

    reg_path.write_text(json.dumps(registry, indent=1, ensure_ascii=False))
    geo_path.write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}, ensure_ascii=False))
    weights.to_parquet(w_path, index=False, compression="zstd")

    for p in (reg_path, geo_path, w_path):
        try:
            shown = p.relative_to(BACKEND_DIR)
        except ValueError:  # --out-dir outside the repo
            shown = p
        print(f"wrote {shown}  ({p.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
