"""District observations from ERA5, pulled in bulk from the Copernicus CDS.

Why this replaces the Open-Meteo fetch
--------------------------------------
Open-Meteo cannot supply this at district scale. Measured 2026-09-10: 80 of 4,902 cells
in 3.5 hours, which is ~215 hours for a single year, because its cap is on request
*weight* and water vapour alone is ~24x the rest combined. That is why 2017's labels
cover 34 districts of 666.

CDS serves the same reanalysis as gridded fields over a bounding box, so one request
returns every cell for a month instead of twenty cells at a time.

It also removes a limitation rather than working around it. Open-Meteo does not serve
ERA5 on its native grid - a requested point snaps to a finer internal grid (~0.07 deg),
so a "cell value" there was ERA5 sampled *at* a cell centre, not the cell's own value.
CDS returns ERA5 on its native 0.25 deg grid, and the district weight table is built on
exactly that grid (verified: lat 6.75..35.5, lon 68.25..97.5, all on 0.25). The
observation is now ERA5's actual grid-box value, aligned one-to-one with the weight table.

The day convention
------------------
ERA5 stamps an accumulation with the END of the hour it covers: `tp` at 00:00 on the 2nd
is the rain that fell 23:00-24:00 on the 1st. So every variable is grouped by
``valid_time - 1h``, which makes an observation day the half-open window (t-24h, t].

That is deliberately the same convention the forecast side uses for day k,
((k-1)*24, k*24] - see CLAUDE.md rule 4. Both sides of a bust label therefore describe
the identical interval. Grouping by a plain calendar floor instead would shift a day's
rainfall by one hour, on the variable that drives most busts.

A consequence worth stating: a complete day needs stamps 01:00..00:00-next-day, so a
month request alone loses its final day. Each month is fetched together with the first
hour of the next month, and days outside the month are dropped.

Requirements
------------
A free CDS account, its personal access token in ``~/.cdsapirc``, and acceptance of both
the `licence-to-use-copernicus-products` and `cc-by` licences for the dataset. Without the
second one the API returns 403 "required licences not accepted", which looks like a code
fault and is not.

    pip install -r requirements.txt -r requirements-live.txt
    python backend/scripts/fetch_era5_cds_district_observations.py --years 2017
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.utils import india_districts as idist  # noqa: E402
# One weight table, one aggregator - shared with the Open-Meteo fetch so the two cannot
# drift apart. See app/utils/district_observations.py.
from app.utils.district_observations import (  # noqa: E402
    VALUE_COLUMNS, district_metadata, grid_cells, to_districts,
)

OUT_DIR = BACKEND_DIR / "data" / "samples"
DATASET = "reanalysis-era5-single-levels"

# CDS request name -> NetCDF short name. Both verified against a real download on
# 2026-09-10, not inferred: the archive contains t2m d2m msl sp u10 v10 swvl1 tcwv tp.
CDS_VARS = {
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "total_precipitation": "tp",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "volumetric_soil_water_layer_1": "swvl1",
    "total_column_water_vapour": "tcwv",
}

HOURS = [f"{h:02d}:00" for h in range(24)]
SOURCE = "ERA5 hourly single levels via Copernicus CDS (CC-BY 4.0; Copernicus C3S)"

# Bolton (1980), the identical constants used by rh_from_specific_humidity in
# scripts/fetch_gefs_reforecast_sample.py. If the two sides used different saturation
# formulas, the humidity bust label would partly be recording that difference.
_BOLTON_A, _BOLTON_B = 17.67, 243.5


def request_area() -> tuple[float, float, float, float]:
    """(north, west, south, east) covering every weight-table cell, snapped outward.

    Taken from the weight table rather than a hand-written bounding box, so the request
    cannot silently stop covering a district. Snapping outward to the 0.25 deg grid keeps
    ERA5 on its native cells - an off-grid corner would make it interpolate, which is the
    very thing this fetch exists to remove.
    """
    w = pd.read_parquet(idist.geo_dir() / idist.WEIGHTS_FILENAME)
    q = 0.25
    return (float(np.ceil(w.lat.max() / q) * q), float(np.floor(w.lon.min() / q) * q),
            float(np.floor(w.lat.min() / q) * q), float(np.ceil(w.lon.max() / q) * q))


def rh_from_dewpoint(t_k: np.ndarray, td_k: np.ndarray) -> np.ndarray:
    """Relative humidity [%] from temperature and dewpoint, both kelvin.

    RH = 100 * es(Td) / es(T). Written as one exponential of the difference rather than a
    ratio of two, which is algebraically identical and does not overflow at absurd inputs.
    """
    tc = np.asarray(t_k, dtype=float) - 273.15
    tdc = np.asarray(td_k, dtype=float) - 273.15
    expo = (_BOLTON_A * tdc / (tdc + _BOLTON_B)) - (_BOLTON_A * tc / (tc + _BOLTON_B))
    return np.clip(100.0 * np.exp(expo), 0.0, 100.0)


def to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Hourly cell readings -> one canonical row per (lat, lon, day).

    Days without their full 24 stamps are dropped, not summed short: a partial day would
    understate rainfall while looking entirely plausible. Refuse rather than patch.
    """
    df = hourly.copy()
    df["time"] = pd.to_datetime(df["time"])
    # The accumulation is stamped at the END of its hour, so shift before taking the date.
    # This makes a day the half-open window (t-24h, t], matching the forecast side.
    df["date"] = (df["time"] - pd.Timedelta(hours=1)).dt.date

    complete = df.groupby(["lat", "lon", "date"])["time"].transform("size") == 24
    df = df[complete]
    if df.empty:
        return pd.DataFrame(columns=["lat", "lon", "date"] + VALUE_COLUMNS)

    # RH is non-linear in T and Td, so it is computed hourly and then averaged - never
    # derived from the daily mean temperature and dewpoint.
    df["_rh"] = rh_from_dewpoint(df["2m_temperature"].to_numpy(),
                                 df["2m_dewpoint_temperature"].to_numpy())

    g = df.groupby(["lat", "lon", "date"])
    out = pd.DataFrame({
        "t2m_c": g["2m_temperature"].mean() - 273.15,
        "rh2m_pct": g["_rh"].mean(),
        "precip_mm": g["total_precipitation"].sum() * 1000.0,     # m -> mm
        "mslp_hpa": g["mean_sea_level_pressure"].mean() / 100.0,  # Pa -> hPa
        "psfc_hpa": g["surface_pressure"].mean() / 100.0,
        # ERA5 soil moisture is m3/m3 (0..1); canonical soil_moisture_pct is % volumetric
        # (0..100), matching the forecast side's soilw_vol_pct.
        "soil_moisture_pct": g["volumetric_soil_water_layer_1"].mean() * 100.0,
        "pwat_kgm2": g["total_column_water_vapour"].mean(),
        "_u": g["10m_u_component_of_wind"].mean(),
        "_v": g["10m_v_component_of_wind"].mean(),
    }).reset_index()

    # Vector mean, then speed and direction - averaging angles would turn two opposing
    # hours into a strong wind pointing sideways. Same convention as the forecast side.
    out["wspd10m_ms"] = np.sqrt(out["_u"] ** 2 + out["_v"] ** 2)
    out["wdir10m_deg"] = (270.0 - np.degrees(np.arctan2(out["_v"], out["_u"]))) % 360.0
    out = out.drop(columns=["_u", "_v"])
    return out[["lat", "lon", "date"] + VALUE_COLUMNS]


def _requests_for(year: int, month: int) -> list[dict]:
    """The month, plus the single boundary hour that completes its last day."""
    north, west, south, east = request_area()
    common = {
        "product_type": ["reanalysis"],
        "variable": list(CDS_VARS),
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [north, west, south, east],
        "grid": [0.25, 0.25],
    }
    main = {**common, "year": [str(year)], "month": [f"{month:02d}"],
            "day": [f"{d:02d}" for d in range(1, 32)], "time": HOURS}
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    boundary = {**common, "year": [str(ny)], "month": [f"{nm:02d}"],
                "day": ["01"], "time": ["00:00"]}
    return [main, boundary]



def _normalise(ds: "xr.Dataset") -> "xr.Dataset":
    """Drop the scalar coords ERA5 carries, and keep valid_time a dimension.

    Not `squeeze(drop=True)`: that removes *every* length-1 dimension, so a request for a
    single hour - which is exactly what the month-boundary pull is - loses `valid_time`
    itself and the merge downstream fails with KeyError. Only `number` and `expver` are
    ever dropped, and only when they are not dimensions.
    """
    for coord in ("number", "expver"):
        if coord in ds.coords and coord not in ds.dims:
            ds = ds.drop_vars(coord)
    if "valid_time" in ds.coords and "valid_time" not in ds.dims:
        ds = ds.expand_dims("valid_time")
    return ds

def _read_archive(path: Path) -> pd.DataFrame:
    """CDS returns a zip of two NetCDFs split by stepType - accumulations in one file,
    instantaneous fields in the other - regardless of download_format. Verified on a real
    download; reading `path` directly as NetCDF fails."""
    frames = []
    with zipfile.ZipFile(path) as z:
        members = [m for m in z.namelist() if m.endswith(".nc")]
        if not members:
            raise RuntimeError(f"no NetCDF members in {path.name}: {z.namelist()}")
        tmp = path.parent / f"_x_{path.stem}"
        tmp.mkdir(parents=True, exist_ok=True)
        for m in members:
            z.extract(m, tmp)
            ds = xr.open_dataset(tmp / m)
            if "expver" in ds.dims:
                raise RuntimeError(
                    f"{m} spans several expver values (mixed ERA5/ERA5T); refusing rather "
                    "than silently picking one")
            ds = _normalise(ds)
            df = ds.to_dataframe().reset_index()
            ds.close()
            frames.append(df)
        for m in members:
            (tmp / m).unlink(missing_ok=True)
        tmp.rmdir()

    short_to_long = {v: k for k, v in CDS_VARS.items()}
    merged = None
    for df in frames:
        keep = ["valid_time", "latitude", "longitude"] + [
            c for c in df.columns if c in short_to_long]
        df = df[[c for c in keep if c in df.columns]].rename(columns=short_to_long)
        merged = df if merged is None else merged.merge(
            df, on=["valid_time", "latitude", "longitude"], how="outer")

    merged = merged.rename(columns={"valid_time": "time", "latitude": "lat",
                                    "longitude": "lon"})
    missing = [v for v in CDS_VARS if v not in merged.columns]
    if missing:
        raise RuntimeError(f"{path.name} is missing {missing}")
    return merged


def fetch_month(client, year: int, month: int, cache: Path) -> pd.DataFrame:
    """Hourly readings for every weight-table cell in one month, boundary hour included."""
    cells = grid_cells()
    wanted = set(zip(np.round(cells.lat, 4), np.round(cells.lon, 4)))

    parts = []
    for i, req in enumerate(_requests_for(year, month)):
        path = cache / f"era5_{year}{month:02d}_{i}.zip"
        if not path.exists():
            cache.mkdir(parents=True, exist_ok=True)
            client.retrieve(DATASET, req, str(path))
        df = _read_archive(path)
        # Keep only the cells the weight table actually uses: the bounding box is a
        # rectangle, India is not, and ~64% of the box is cells no district touches.
        key = list(zip(np.round(df.lat, 4), np.round(df.lon, 4)))
        df = df[[k in wanted for k in key]]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def build(years: list[int], months: list[int] | None = None,
          keep_downloads: bool = False) -> None:
    import cdsapi

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = grid_cells()
    meta = district_metadata()
    north, west, south, east = request_area()
    print(f"area N{north} W{west} S{south} E{east} on a 0.25 deg grid")
    print(f"{len(cells)} weight-table cells -> {len(meta)} districts")

    client = cdsapi.Client()
    cache = OUT_DIR / "_era5_cds"

    for year in years:
        t0 = time.time()
        monthly = []
        for month in (months or list(range(1, 13))):
            ckpt = cache / f"districts_{year}{month:02d}.parquet"
            if ckpt.exists():
                monthly.append(pd.read_parquet(ckpt))
                print(f"  {year}-{month:02d}  checkpointed")
                continue

            hourly = fetch_month(client, year, month, cache)
            daily = to_daily(hourly)
            # Drop the boundary day that belongs to the next month.
            in_month = pd.to_datetime(daily["date"]).dt.month == month
            daily = daily[in_month]
            districts = to_districts(daily, cells)

            ckpt.parent.mkdir(parents=True, exist_ok=True)
            districts.to_parquet(ckpt, index=False)
            monthly.append(districts)
            print(f"  {year}-{month:02d}  {daily.date.nunique()} days  "
                  f"{len(districts):,} district-days  {time.time()-t0:,.0f}s")

            if not keep_downloads:
                for z in cache.glob(f"era5_{year}{month:02d}_*.zip"):
                    z.unlink()

        out = pd.concat(monthly, ignore_index=True)
        out = out.merge(meta, on="region_id", how="left")
        out["source"] = SOURCE
        cols = (["region_id", "region_name", "state_id", "state_name",
                 "latitude", "longitude", "date"] + VALUE_COLUMNS + ["source"])
        out = out[cols].sort_values(["region_id", "date"])

        suffix = "" if len(monthly) == 12 else f"_m{min(months or [1])}-{max(months or [12])}"
        path = OUT_DIR / f"era5_cds_district_observations_india_{year}{suffix}.parquet"
        out.to_parquet(path, index=False)
        print(f"  -> {path.name}  {len(out):,} rows  {path.stat().st_size/1e6:.1f} MB  "
              f"{time.time()-t0:,.0f}s")
        nulls = out[VALUE_COLUMNS].isna().sum()
        if nulls.any():
            print("  nulls by variable:")
            print(nulls[nulls > 0].to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default="2017", help="2017, 2015-2019, or 2015,2017")
    ap.add_argument("--months", default="1-12",
                    help="1-12, 7, or 1,2,3 - which months of each year to fetch")
    ap.add_argument("--keep-downloads", action="store_true",
                    help="keep the raw CDS zips instead of deleting each month once it "
                         "has been aggregated")
    args = ap.parse_args()

    years: set[int] = set()
    for chunk in args.years.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            years.update(range(int(a), int(b) + 1))
        elif chunk:
            years.add(int(chunk))
    months: set[int] = set()
    for chunk in args.months.split(','):
        chunk = chunk.strip()
        if '-' in chunk:
            a, b = chunk.split('-', 1)
            months.update(range(int(a), int(b) + 1))
        elif chunk:
            months.add(int(chunk))
    build(sorted(years), sorted(months), keep_downloads=args.keep_downloads)


if __name__ == "__main__":
    main()
