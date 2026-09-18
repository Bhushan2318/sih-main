"""Elevation for every 0.25 deg grid cell the district weight table uses.

Why this exists
----------------
C4's district descriptors (scripts/build_district_descriptors.py) replace region_id with
geometry-derived features, but had no elevation term: nothing in this repo had ever
fetched one. This does, and only this - it fetches elevation, nothing else.

Source, verified before writing this script, not assumed
----------------------------------------------------------
Open-Elevation's public API (https://api.open-elevation.com), a free, no-auth,
open-source service (github.com/Jorl17/open-elevation). Its own setup docs
(docs/host-your-own.md, fetched 2026-09-18) name its dataset as the CGIAR-CSI SRTM 250m
resampled product (https://srtm.csi.cgiar.org). Checked against three well-known points
before trusting it for 4,902: Everest 27.9881N 86.9250E -> 8771 m (real ~8849 m, close at
250 m resolution), Mumbai 19.0760N 72.8777E -> 6 m (coastal, correct), Delhi 28.6139N
77.2090E -> 214 m (real ~216 m).

Reuses the one weight table
----------------------------
CLAUDE.md: there is exactly one district weight table, and this does not write a second
one. It queries elevation for the same (lat, lon) cells `district_grid_weights.parquet`
already carries - the grid GEFS/ERA5 aggregation already uses - and
`build_district_descriptors.py` aggregates it through the existing
`DistrictGridAggregator`, exactly like a gridded forecast field.

Fetch once
----------
4,902 unique cells, batched 500 at a time (~10 requests, ~25 s measured against the real
API on 2026-09-18). Idempotent: refuses to re-fetch over an existing output file, out of
courtesy to a shared free public service and per CLAUDE.md's fetch discipline.

    python -m scripts.fetch_grid_elevation
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pandas as pd

from app.utils import india_districts as idist

API_URL = "https://api.open-elevation.com/api/v1/lookup"
SOURCE = "Open-Elevation public API (CGIAR-CSI SRTM 250m, https://srtm.csi.cgiar.org)"
BATCH = 500
OUT_FILENAME = "grid_elevation_m.parquet"


def _fetch_batch(cells: list[tuple[float, float]], attempt: int = 0) -> list[float]:
    locs = [{"latitude": lat, "longitude": lon} for lat, lon in cells]
    body = json.dumps({"locations": locs}).encode()
    req = urllib.request.Request(
        API_URL, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            out = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        if attempt >= 3:
            raise RuntimeError(
                f"elevation fetch failed after 3 attempts ({len(cells)} points): {e}. "
                "Refusing to fabricate a value - rerun once the API is reachable.") from e
        time.sleep(2 ** attempt)
        return _fetch_batch(cells, attempt + 1)

    results = out.get("results", [])
    if len(results) != len(cells):
        raise RuntimeError(
            f"asked the API for {len(cells)} points, got {len(results)} back - "
            "refusing to guess which one is missing")
    return [float(r["elevation"]) for r in results]


def fetch() -> pd.DataFrame:
    """One row per unique weight-table cell: lat, lon, elevation_m, source."""
    w = pd.read_parquet(idist.geo_dir() / idist.WEIGHTS_FILENAME)
    cells = w[["lat", "lon"]].drop_duplicates().reset_index(drop=True)

    elevations: list[float] = []
    for i in range(0, len(cells), BATCH):
        chunk = list(zip(cells["lat"].iloc[i:i + BATCH].tolist(),
                        cells["lon"].iloc[i:i + BATCH].tolist()))
        elevations.extend(_fetch_batch(chunk))
        print(f"  elevation {min(i + BATCH, len(cells))}/{len(cells)} cells")

    cells = cells.copy()
    cells["elevation_m"] = elevations
    cells["source"] = SOURCE
    return cells


def main() -> None:
    path = idist.geo_dir() / OUT_FILENAME
    if path.exists():
        print(f"{path} already exists - not re-fetching (delete it to refresh). "
              "CLAUDE.md: fetch once, do not casually re-run.")
        return
    out = fetch()
    out.to_parquet(path, index=False)
    print(f"{len(out)} cells -> {path} ({path.stat().st_size / 1024:.1f} KB)")
    print(out["elevation_m"].describe().to_string())


if __name__ == "__main__":
    main()
