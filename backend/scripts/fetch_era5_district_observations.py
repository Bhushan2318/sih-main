"""Build the observation side of Sanket at district resolution, from ERA5.

Why this replaces the city-point version
----------------------------------------
The forecast side is now an area-weighted mean over each district's 0.25 deg grid cells.
If the observation stayed a single point reading, the two sides would be measuring
different things, and the bust label - the difference between them - would be partly
recording that mismatch rather than forecast error. In one Open-Meteo response, the same
day's rainfall across 0.28 deg of latitude read 5.7, 6.9, 13.7 and 21.9 mm. Rainfall
drives most busts, so the sampling difference is not a rounding detail.

So observations are aggregated exactly like forecasts: sampled at the 0.25 deg grid cell
centres and combined through the same weight table, in
``app/utils/india_districts.py``. Both sides become area-weighted means over the same
polygon from the same locations.

Honest limitation
-----------------
Open-Meteo does not serve ERA5 on its native 0.25 deg grid - a requested point snaps to a
finer internal grid (~0.07 deg). So a "cell value" here is ERA5 sampled *at* that cell's
centre, not ERA5's own cell mean. The two sides are therefore consistent in area and in
sampling location, but the observation is not literally the ERA5 grid-box average. That
is a real caveat and is written down rather than smoothed over.

Source
------
ERA5 reanalysis via the Open-Meteo Historical Weather API
(https://open-meteo.com/en/docs/historical-weather-api). CC-BY 4.0; underlying ERA5 is
Copernicus Climate Change Service information. No API key; non-commercial use.

Eight variables come from the daily endpoint, which is ~24x lighter than hourly. Total
column water vapour has no daily form, so it alone is pulled hourly and averaged here -
it is kept because it backs the atmospheric_moisture_kgm2 regressor.

    python backend/scripts/fetch_era5_district_observations.py --years 2019
    python backend/scripts/fetch_era5_district_observations.py --years 2000-2019
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.utils import india_districts as idist  # noqa: E402

OUT_DIR = BACKEND_DIR / "data" / "samples"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Daily variable -> canonical column. Kept in step with the forecast side's canonical
# names so the two join without a translation layer.
DAILY_VARS = {
    "temperature_2m_mean": "t2m_c",
    "relative_humidity_2m_mean": "rh2m_pct",
    "precipitation_sum": "precip_mm",
    "pressure_msl_mean": "mslp_hpa",
    "surface_pressure_mean": "psfc_hpa",
    "wind_speed_10m_mean": "wspd10m_ms",
    "wind_direction_10m_dominant": "wdir10m_deg",
    "soil_moisture_0_to_7cm_mean": "soil_moisture_pct",
}
HOURLY_ONLY = "total_column_integrated_water_vapour"   # -> pwat_kgm2

VALUE_COLUMNS = list(DAILY_VARS.values()) + ["pwat_kgm2"]

# Cells per request. Open-Meteo accepts comma-separated coordinates; the cap is on the
# response, so a batch is bounded by (cells x days x variables), not by cells alone.
BATCH_CELLS = 20
HTTP_RETRIES = 5
HTTP_BACKOFF = 4.0

SOURCE = "ERA5 via Open-Meteo archive-api (CC-BY 4.0; Copernicus C3S)"

_session = requests.Session()


def _get_json(params: dict) -> list | dict:
    last = None
    for attempt in range(HTTP_RETRIES):
        try:
            r = _session.get(ARCHIVE_URL, params=params, timeout=180)
            if r.status_code == 200:
                return r.json()
            # 429 is the documented signal that the daily/hourly quota is spent. Back off
            # hard rather than hammering: an earlier run of the city version tripped this
            # and lost a whole pass.
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            if r.status_code == 429:
                time.sleep(HTTP_BACKOFF * (attempt + 1) * 5)
                continue
        except requests.RequestException as exc:
            last = exc
        time.sleep(HTTP_BACKOFF * (attempt + 1))
    raise RuntimeError(f"Open-Meteo request failed after {HTTP_RETRIES} tries: {last}")


def _as_list(payload) -> list:
    """One coordinate returns an object, several return a list. Normalise."""
    return payload if isinstance(payload, list) else [payload]


def grid_cells() -> pd.DataFrame:
    """The distinct 0.25 deg cells the district weight table draws on."""
    w = pd.read_parquet(idist.geo_dir() / idist.WEIGHTS_FILENAME)
    cells = w[["lat", "lon"]].drop_duplicates().sort_values(["lat", "lon"])
    return cells.reset_index(drop=True)


def fetch_batch(cells: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Daily values plus hourly-averaged water vapour for one batch of cells."""
    lat = ",".join(f"{v:.4f}" for v in cells["lat"])
    lon = ",".join(f"{v:.4f}" for v in cells["lon"])
    common = {"latitude": lat, "longitude": lon, "start_date": start,
              "end_date": end, "timezone": "UTC"}

    daily = _as_list(_get_json({**common, "daily": ",".join(DAILY_VARS)}))
    hourly = _as_list(_get_json({**common, "hourly": HOURLY_ONLY}))
    if len(daily) != len(cells) or len(hourly) != len(cells):
        raise RuntimeError(
            f"asked for {len(cells)} cells, got {len(daily)} daily / {len(hourly)} hourly"
        )

    frames = []
    for (_, cell), d, h in zip(cells.iterrows(), daily, hourly):
        block = d.get("daily")
        if not block:
            raise RuntimeError(f"no daily block for cell {cell.lat},{cell.lon}: "
                               f"{str(d)[:200]}")
        df = pd.DataFrame({canon: block[src] for src, canon in DAILY_VARS.items()})
        df["date"] = pd.to_datetime(block["time"]).date

        # ERA5 soil moisture is m3/m3 (0..1); canonical soil_moisture_pct is % volumetric
        # (0..100), matching the forecast side's soilw_vol_pct.
        df["soil_moisture_pct"] = df["soil_moisture_pct"] * 100.0

        hb = h.get("hourly")
        if hb:
            pw = pd.DataFrame({"t": pd.to_datetime(hb["time"]),
                               "v": hb[HOURLY_ONLY]})
            pw = pw.groupby(pw["t"].dt.date)["v"].mean()
            df["pwat_kgm2"] = df["date"].map(pw)
        else:
            df["pwat_kgm2"] = np.nan

        df["lat"] = cell.lat
        df["lon"] = cell.lon
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def to_districts(cell_rows: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Area-weighted mean over each district's cells, one value per (district, date).

    The same aggregator and the same weights the forecast side uses. NaN cells drop out
    and the remaining weights renormalise, so a coastal district is not voided by the sea
    cells it overlaps; a district with no valid cell for a variable stays NaN.
    """
    agg = idist.get_aggregator()
    prepared = agg.prepare(cells["lat"].to_numpy(), cells["lon"].to_numpy())

    # Cell order must match `cells`, and every cell must be present for every date, or a
    # value would silently line up against the wrong location.
    pivot_index = pd.MultiIndex.from_frame(cells[["lat", "lon"]])
    out = []
    for date, chunk in cell_rows.groupby("date", sort=True):
        chunk = chunk.set_index(["lat", "lon"]).reindex(pivot_index)
        series = {}
        for col in VALUE_COLUMNS:
            if col == "wdir10m_deg":
                # Direction is circular: a plain mean of 350 and 10 degrees is 180, which
                # points the opposite way. Average the unit vectors instead.
                rad = np.radians(chunk[col].to_numpy(dtype=float))
                u = agg.aggregate_prepared(prepared, np.sin(rad))
                v = agg.aggregate_prepared(prepared, np.cos(rad))
                series[col] = (np.degrees(np.arctan2(u, v)) % 360.0)
            else:
                series[col] = agg.aggregate_prepared(prepared,
                                                     chunk[col].to_numpy(dtype=float))
        df = pd.DataFrame(series)
        df.insert(0, "date", date)
        df.insert(0, "region_id", df.index)
        out.append(df.reset_index(drop=True))
    return pd.concat(out, ignore_index=True)


def build(years: list[int], margin_days: int = 14) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = grid_cells()
    meta = pd.DataFrame([
        {"region_id": d.region_id, "region_name": d.region_name,
         "state_id": d.state_id, "state_name": d.state_name,
         "latitude": d.centroid_lat, "longitude": d.centroid_lon}
        for d in idist.load_registry()
    ])
    print(f"{len(cells)} grid cells -> {len(meta)} districts")

    for year in years:
        start = f"{year}-01-01"
        # A Day-10 forecast issued in late December verifies in the next year, so the
        # window runs past the year end.
        end = (pd.Timestamp(f"{year}-12-31") + pd.Timedelta(days=margin_days)).date()
        print(f"\n* {year}  {start} .. {end}")

        t0 = time.time()
        parts = []
        for i in range(0, len(cells), BATCH_CELLS):
            batch = cells.iloc[i:i + BATCH_CELLS]
            parts.append(fetch_batch(batch, start, str(end)))
            done = min(i + BATCH_CELLS, len(cells))
            print(f"    cells {done}/{len(cells)}  {time.time()-t0:,.0f}s", end="\r",
                  flush=True)
        cell_rows = pd.concat(parts, ignore_index=True)
        print(f"\n    fetched {len(cell_rows):,} cell-days in {time.time()-t0:,.0f}s")

        districts = to_districts(cell_rows, cells)
        districts = districts.merge(meta, on="region_id", how="left")
        districts["source"] = SOURCE
        cols = (["region_id", "region_name", "state_id", "state_name",
                 "latitude", "longitude", "date"] + VALUE_COLUMNS + ["source"])
        districts = districts[cols].sort_values(["region_id", "date"])

        path = OUT_DIR / f"era5_district_observations_india_{year}.parquet"
        districts.to_parquet(path, index=False)
        print(f"    -> {path.name}  {len(districts):,} rows  "
              f"{path.stat().st_size/1e6:.1f} MB")
        nulls = districts[VALUE_COLUMNS].isna().sum()
        if nulls.any():
            print("    nulls by variable:")
            print(nulls[nulls > 0].to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default="2019",
                    help="2019, 2010-2019, or 2011,2014,2019")
    ap.add_argument("--max-cells", type=int, default=None,
                    help="use only the first N grid cells (for a quick check)")
    args = ap.parse_args()

    years: set[int] = set()
    for chunk in args.years.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            years.update(range(int(a), int(b) + 1))
        elif chunk:
            years.add(int(chunk))

    if args.max_cells:
        global grid_cells
        _full = grid_cells
        grid_cells = lambda: _full().head(args.max_cells)  # noqa: E731

    build(sorted(years))


if __name__ == "__main__":
    main()
