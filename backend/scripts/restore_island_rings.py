"""Put back the island rings that simplifying the map's TopoJSON throws away.

Run this after mapshaper, as the last step of building
``frontend/src/assets/geo/india_districts.topojson``. See ``build_district_geo.py`` for
the whole pipeline.

Why it has to exist
-------------------
The map's TopoJSON is built with ``-simplify percentage=12% keep-shapes``. ``keep-shapes``
guarantees one ring per *feature*, not per part, so every district made of several pieces
keeps its largest piece and loses the rest. Measured against the full-resolution
boundaries, 64 districts lost 700 rings that way.

For most of them that is a detail - an offshore islet missing from a district that is
plainly visible anyway. For Lakshadweep it is the whole thing: it is nothing but small
islands, so 23 of its 24 disappeared and the union territory was absent from the map of
India. What survived measured 0.64 x 1.23 units in a 620 x 680 viewBox, under one CSS
pixel on a desktop and a third of one on a phone, and its click target was the same size.

Simplifying more gently does not fix it. Measured on the same input: 12% keeps 1 island of
24 at 383 KB, 50% keeps 1 at 755 KB, 70% keeps 3 at 860 KB, and only no simplification at
all keeps 24, at 1,035 KB. The rings are orders of magnitude smaller than any useful
tolerance, so they will always be the first thing a simplifier discards.

So rather than re-simplify, this takes the simplified output exactly as mapshaper produced
it and appends only the rings that are missing. A ring counts as missing when it
intersects nothing already present for that district. Islands share no borders with
anything, so adding them back cannot reopen the sliver problem that ``-clean`` exists to
solve, and the mainland is not touched at all: measured over every pre-existing vertex,
the largest shift was 0.00041 deg (~45 m), below the 0.0006 deg quantisation step, which
is re-quantisation noise rather than a change of shape.

Cost: 383 KB -> 415 KB, or 114 KB -> 126 KB gzipped.

Usage
-----
    npx mapshaper@0.6 <simplified>.topojson -o format=geojson /tmp/base.geojson
    python backend/scripts/restore_island_rings.py \
        /tmp/base.geojson backend/data/geo/india_districts.geojson /tmp/merged.geojson
    npx mapshaper@0.6 /tmp/merged.geojson -rename-layers districts \
        -o format=topojson frontend/src/assets/geo/india_districts.topojson
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from shapely.geometry import shape
from shapely.strtree import STRtree


def _parts(geometry: dict) -> list:
    """Every polygon of a Polygon or MultiPolygon, as coordinate lists."""
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"]]
    if geometry["type"] == "MultiPolygon":
        return list(geometry["coordinates"])
    raise ValueError(f"not an areal geometry: {geometry['type']}")


def restore(base: dict, source: dict) -> Counter:
    """Append source rings missing from ``base``, in place. Returns what was added."""
    by_id = {f["properties"]["region_id"]: f for f in source["features"]}
    added: Counter = Counter()

    for feat in base["features"]:
        region_id = feat["properties"]["region_id"]
        src = by_id.get(region_id)
        if src is None:
            continue

        present = [shape({"type": "Polygon", "coordinates": c}).buffer(0)
                   for c in _parts(feat["geometry"])]
        tree = STRtree(present)
        keep = _parts(feat["geometry"])

        for coords in _parts(src["geometry"]):
            ring = shape({"type": "Polygon", "coordinates": coords}).buffer(0)
            if ring.is_empty:
                continue
            # buffer(0) above makes the intersects test safe on self-touching rings.
            if any(present[i].intersects(ring) for i in tree.query(ring)):
                continue
            keep.append(coords)
            added[region_id] += 1

        if added[region_id]:
            feat["geometry"] = {"type": "MultiPolygon", "coordinates": keep}

    return added


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", type=Path, help="mapshaper's simplified output, as GeoJSON")
    ap.add_argument("source", type=Path, help="full boundaries (backend/data/geo/india_districts.geojson)")
    ap.add_argument("out", type=Path, help="where to write the merged GeoJSON")
    args = ap.parse_args()

    base = json.loads(args.base.read_text())
    source = json.loads(args.source.read_text())

    before = len(base["features"])
    added = restore(base, source)
    if len(base["features"]) != before:
        raise SystemExit(f"district count changed: {before} -> {len(base['features'])}")

    args.out.write_text(json.dumps(base))
    print(f"restored {sum(added.values())} rings across {len(added)} districts")
    for region_id, n in added.most_common(10):
        print(f"  {region_id:36} +{n}")


if __name__ == "__main__":
    main()
