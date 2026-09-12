"""District rainfall from IMD's gauge-based gridded product, merged into an already-fetched
ERA5/CDS district observation file.

Why this exists
----------------
CLAUDE.md's own stated plan: "IMD gauge-based gridded rainfall at native 0.25 deg for
precipitation, ERA5 ... for everything else." ERA5 precipitation is a reanalysis product
and is measurably weak over India - it is the hardest variable and the driver of most
busts. IMD's product is built from a dense in-situ rain-gauge network, interpolated onto
the same 0.25 deg grid the district weight table already uses.

Why merge rather than write a standalone observation file
-----------------------------------------------------------
IMD only publishes rain (and tmin/tmax, not used here). A precip_mm-only file would be
missing the other eight canonical variables `ingest_upload` expects, so this script reads
the best already-fetched ERA5-family file for the year (via
`ingest_backfill.observation_file`) and replaces only its precip_mm column - everything
else (temperature, humidity, wind, pressure, soil moisture, water vapour) keeps coming
from ERA5/CDS. The two products never blend inside one column: `source` is annotated so
it is always traceable which product produced a row's rainfall.

Access
------
Verified against the real archive 2026-09-13: a plain POST to
https://imdpune.gov.in/cmpg/Griddata/rainfall.php with {"rain": year} returns a real
binary .grd file, no account or key needed. `imdlib` (PyPI) wraps that request and the
binary parsing; this script uses it rather than re-implementing the format.

Grid: 129 lat points (6.5..38.5), 135 lon points (66.5..100.0), both 0.25 deg spacing -
a superset of the district weight table's own grid, so cell matching is a plain filter,
not an interpolation. Missing/sea cells carry the sentinel -999.0.

    pip install imdlib
    python -m scripts.fetch_imd_district_rainfall --years 2018
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.utils.district_observations import grid_cells, to_districts  # noqa: E402
from scripts import ingest_backfill as ib  # noqa: E402

OUT_DIR = BACKEND_DIR / "data" / "samples"
SOURCE_IMD = "IMD gauge-based gridded rainfall 0.25 deg (imdpune.gov.in)"
MISSING = -999.0


def imd_to_long(imd_obj) -> pd.DataFrame:
    """An imdlib IMD object (or anything with the same `.data`/`.lat_array`/`.lon_array`/
    `.start_day` attributes) -> long rows of (lat, lon, date, precip_mm).

    `.data` is (days, lon, lat) - imdlib's own axis order, not this project's (lat, lon).
    The sentinel -999.0 becomes NaN: a sea/no-gauge cell is missing, never a real zero.
    """
    data = imd_obj.data
    n_days = data.shape[0]
    dates = pd.date_range(imd_obj.start_day, periods=n_days)
    lon_arr, lat_arr = imd_obj.lon_array, imd_obj.lat_array

    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr, indexing="ij")
    rows = []
    for i, date in enumerate(dates):
        day = data[i]
        vals = np.where(day == MISSING, np.nan, day)
        rows.append(pd.DataFrame({
            "lat": lat_grid.ravel(),
            "lon": lon_grid.ravel(),
            "date": date,
            "precip_mm": vals.ravel(),
        }))
    return pd.concat(rows, ignore_index=True)


def merge_precip(base: pd.DataFrame, imd_districts: pd.DataFrame) -> pd.DataFrame:
    """Replace `base`'s precip_mm with IMD's, matched on (region_id, date).

    Refuses if IMD covers fewer (region_id, date) pairs than `base` - a real coverage gap,
    not a row to drop silently (CLAUDE.md rule 3: refuse rather than patch). A district-date
    IMD covers but has no valid gauge cell for stays NaN, which is different from IMD never
    having been asked about it at all.
    """
    base = base.copy()
    base["date"] = pd.to_datetime(base["date"]).dt.date
    imd = imd_districts.copy()
    imd["date"] = pd.to_datetime(imd["date"]).dt.date

    base_keys = set(zip(base["region_id"], base["date"]))
    imd_keys = set(zip(imd["region_id"], imd["date"]))
    missing = base_keys - imd_keys
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"IMD is missing {len(missing)} of {len(base_keys)} (region_id, date) pairs "
            f"the base file covers, e.g. {sample} - refusing a silently narrower merge")

    imd_lookup = imd.set_index(["region_id", "date"])["precip_mm"]
    key = list(zip(base["region_id"], base["date"]))
    base["precip_mm"] = [imd_lookup.get(k, np.nan) for k in key]
    base["source"] = base["source"].astype(str) + " | precip_mm: " + SOURCE_IMD
    return base


def build(years: list[int]) -> None:
    import imdlib as imd

    cells = grid_cells()
    wanted = set(zip(np.round(cells.lat, 4), np.round(cells.lon, 4)))

    for year in years:
        base_path = ib.observation_file(year)
        if base_path is None:
            print(f"SKIP {year}: no existing ERA5-family observation file to merge into "
                  f"(run fetch_era5_cds_district_observations.py first)", file=sys.stderr)
            continue

        cache = OUT_DIR / "_imd"
        cache.mkdir(parents=True, exist_ok=True)
        print(f"downloading IMD rain {year} ...")
        obj = imd.get_data("rain", year, year, fn_format="yearwise", file_dir=str(cache))

        long = imd_to_long(obj)
        key = list(zip(np.round(long.lat, 4), np.round(long.lon, 4)))
        long = long[[k in wanted for k in key]]

        imd_districts = to_districts(long, cells, value_columns=["precip_mm"])

        base = (pd.read_parquet(base_path) if base_path.suffix == ".parquet"
                else pd.read_csv(base_path))
        merged = merge_precip(base, imd_districts)

        out = OUT_DIR / f"imd_merged_district_observations_india_{year}.parquet"
        merged.to_parquet(out, index=False)
        delta = (merged["precip_mm"] - pd.read_parquet(base_path)["precip_mm"]).abs()
        print(f"  -> {out.name}  {len(merged):,} rows  "
              f"mean |IMD - ERA5| precip_mm = {delta.mean():.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", required=True, help="2018, or 2016-2019")
    args = ap.parse_args()

    years: set[int] = set()
    for chunk in args.years.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            years.update(range(int(a), int(b) + 1))
        elif chunk:
            years.add(int(chunk))

    build(sorted(years))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
