"""Which districts' soil moisture is a sea value in GEFS or ERA5 - data/geo/soil_land_mask.parquet.

Soil moisture only exists over land, and each model fills its sea differently: GEFS writes
~1.0 (100%) into sea cells, ERA5 ~0. A district's value is the area-weighted mean over every
0.25 deg cell it touches, so a coastal district's soil value is partly - for Mumbai City
almost entirely - a sea convention. app.features.engineering drops soil moisture, on both
sides, for districts above SOIL_SEA_FRACTION_MAX sea in either model (feature version 2).

GEFS sea cells: from the gridded fields already on disk (the CNN bundles), a cell whose
Day-1 ensemble-mean soil moisture is >= 0.9 in at least 95% of the cycles that have a value.
Land soil never comes near 0.9; quantisation puts the sea between 0.995 and 1.005, and a few
cycles carry no soil field at all (NaN), which are not counted.

ERA5 water cells: ERA5's own land-sea mask, lsm < 0.5, fetched once by
`fetch_era5_cds_district_observations.py --land-sea-mask`.

    python backend/scripts/build_soil_land_mask.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.features.engineering import SOIL_LAND_MASK_PATH, SOIL_SEA_FRACTION_MAX  # noqa: E402
from app.ingestion import grid_fields as gf  # noqa: E402
from app.utils import india_districts as idist  # noqa: E402

SATURATED = 0.9
ALWAYS = 0.95
GRID_DIRS = [BACKEND / "data" / "samples" / "grids", BACKEND / "data" / "samples" / "grids-2018"]
LSM_PATH = BACKEND / "data" / "samples" / "_era5_cds_v2" / "land_sea_mask.parquet"


def gefs_nonland_cells(fields: np.ndarray) -> np.ndarray:
    """`fields[cycle, lat, lon]` of Day-1 soil moisture (fraction) -> bool[lat, lon]: sea in
    GEFS. NaN cycles do not count either way; a cell with no valid cycle is not sea."""
    valid = np.isfinite(fields)
    sat = valid & (np.nan_to_num(fields, nan=0.0) >= SATURATED)
    n_valid = valid.sum(axis=0)
    return (n_valid > 0) & (sat.sum(axis=0) >= ALWAYS * np.maximum(n_valid, 1))


def district_mask(weights: pd.DataFrame, gefs: pd.DataFrame, lsm: pd.DataFrame,
                  cut: float = SOIL_SEA_FRACTION_MAX) -> pd.DataFrame:
    """Per district: the weight share that is GEFS sea, the share that is ERA5 water, and
    whether soil moisture is excluded - GEFS sea above `cut`.

    Only the forecast side decides it. Estimator-v2 observations
    (district_observations.to_districts_v2) already average ERA5 soil over land cells
    only, so ERA5's water never reaches them; excluding on it too would discard real soil
    in 26 coastal districts (Kerala, Goa, the Odisha coast) where GEFS has no sea at all.
    `era5_water_frac` is kept for reference. This assumes the v2 observations: v1 averaged
    ERA5's water in as ~0 (docs/known-issues.md)."""
    def key(df):
        return df.assign(lat=df["lat"].round(4), lon=df["lon"].round(4))
    m = key(weights).merge(key(gefs), on=["lat", "lon"], how="left").merge(
        key(lsm), on=["lat", "lon"], how="left")
    if m["gefs_nonland"].isna().any() or m["lsm"].isna().any():
        raise ValueError(f"{int(m['gefs_nonland'].isna().sum())} weight cells lack a GEFS "
                         f"value and {int(m['lsm'].isna().sum())} lack an ERA5 mask value")
    m["w_gefs"] = m["weight"] * m["gefs_nonland"].astype(float)
    m["w_era5"] = m["weight"] * (m["lsm"] < 0.5).astype(float)
    g = m.groupby("region_id")[["weight", "w_gefs", "w_era5"]].sum()
    out = pd.DataFrame({"gefs_nonland_frac": g["w_gefs"] / g["weight"],
                        "era5_water_frac": g["w_era5"] / g["weight"]})
    out["soil_excluded"] = out["gefs_nonland_frac"] > cut
    return out.reset_index()


def main() -> int:
    files = sorted(p for d in GRID_DIRS if d.exists() for p in d.glob("*.npz"))
    if not files:
        print(f"no GEFS grid bundles under {GRID_DIRS}", file=sys.stderr)
        return 1
    if not LSM_PATH.exists():
        print(f"no ERA5 land-sea mask at {LSM_PATH}; fetch it with "
              "fetch_era5_cds_district_observations.py --land-sea-mask", file=sys.stderr)
        return 1
    first = gf.load_bundle(files[0])
    fields = np.stack([gf.load_bundle(f).field("soilw_bgrnd", 1, "mean") for f in files])
    nonland = gefs_nonland_cells(fields)
    lat2d, lon2d = np.meshgrid(first.lats, first.lons, indexing="ij")
    gefs = pd.DataFrame({"lat": lat2d.ravel().astype(float), "lon": lon2d.ravel().astype(float),
                         "gefs_nonland": nonland.ravel()})
    weights = pd.read_parquet(idist.geo_dir() / idist.WEIGHTS_FILENAME)
    lsm = pd.read_parquet(LSM_PATH)
    mask = district_mask(weights, gefs, lsm)
    mask.to_parquet(SOIL_LAND_MASK_PATH, index=False)
    ex = mask[mask["soil_excluded"]].sort_values("gefs_nonland_frac", ascending=False)
    print(f"{len(files)} GEFS cycles, {int(nonland.sum())} GEFS sea cells; "
          f"{len(ex)} of {len(mask)} districts excluded at {SOIL_SEA_FRACTION_MAX:.0%} sea")
    print(ex.round(3).to_string(index=False))
    print(f"-> {SOIL_LAND_MASK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
