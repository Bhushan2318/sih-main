from __future__ import annotations

import functools
import logging
import time
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from app.utils.india_districts import (WEIGHTS_FILENAME, geo_dir, get_aggregator,
                                       load_registry)

log = logging.getLogger("forecastguard.live.obs")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

BACKEND_DIR = Path(__file__).resolve().parents[2]

# Open-Meteo takes comma-separated coordinates and answers with one object per location.
# 300 keeps the request URI well inside the limit - 666 coordinates returns HTTP 414
# URI-too-large, which is a URL-length ceiling and not a rate limit. Measured 2026-09-23:
# 300 real weight-table cells, two days, all nine hourly variables, HTTP 200 in 2.0 s and
# 1,139 KB. 4,902 cells is 17 batches, ~33 s plus polite gaps for one day.
CELL_BATCH = 300

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "soil_moisture_0_to_7cm",
    "total_column_integrated_water_vapour",
]
FORECAST_HOURLY_VARS = [v for v in HOURLY_VARS if v != "total_column_integrated_water_vapour"]

HTTP_RETRIES = 5
HTTP_BACKOFF = 3.0
POLITE_GAP_S = 1.2

_session = requests.Session()
_session.headers["User-Agent"] = "Sanket/phase6-live (SIH 2026; NCMRWF PS 26079)"


@dataclass
class ObsReport:
    tier: str
    start: date
    end: date
    cells: int = 0
    rows: int = 0
    failures: list = field(default_factory=list)
    seconds: float = 0.0


def _get_json(url: str, params: dict) -> dict:
    last = None
    for attempt in range(HTTP_RETRIES):
        try:
            r = _session.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as exc:
            last = str(exc)
        time.sleep(HTTP_BACKOFF * (attempt + 1))
    raise RuntimeError(f"observation request failed after {HTTP_RETRIES} tries: {last}")


@functools.lru_cache(maxsize=1)
def _cells() -> tuple:
    """The 0.25 degree cells India overlaps, in one fixed order.

    These are the weight table's own cells - 4,902 of them - not the 666 district
    centroids. A centroid is a point sample, and pairing a point observation against an
    area-mean forecast is the mismatch CLAUDE.md's single-weight-table rule exists to
    prevent: "both sides of the bust label are area means over the same polygon".

    The order is fixed because `_prepared_index` maps these positions into the aggregator
    and the per-cell value arrays are built positionally against it.
    """
    w = pd.read_parquet(geo_dir() / WEIGHTS_FILENAME)
    cells = (w[["lat", "lon"]].drop_duplicates()
             .sort_values(["lat", "lon"], ignore_index=True))
    return (cells["lat"].to_numpy(dtype=float), cells["lon"].to_numpy(dtype=float))


@functools.lru_cache(maxsize=1)
def _prepared_index():
    lats, lons = _cells()
    return get_aggregator().prepare(lats, lons)


@functools.lru_cache(maxsize=1)
def _district_identity() -> pd.DataFrame:
    recs = {r.region_id: r for r in load_registry()}
    ids = get_aggregator().region_ids
    return pd.DataFrame({
        "region_id": ids,
        "region_name": [recs[i].region_name if i in recs else None for i in ids],
        "state_id": [recs[i].state_id if i in recs else None for i in ids],
        "state_name": [recs[i].state_name if i in recs else None for i in ids],
        "latitude": [recs[i].centroid_lat if i in recs else np.nan for i in ids],
        "longitude": [recs[i].centroid_lon if i in recs else np.nan for i in ids],
    })


def _fetch_cell_batch(lats, lons, start: date, end: date, tier: str) -> list:
    """One request for up to CELL_BATCH cells; returns each cell's hourly block in order."""
    params = {
        "latitude": ",".join(f"{v:.4f}" for v in lats),
        "longitude": ",".join(f"{v:.4f}" for v in lons),
        "windspeed_unit": "ms",
        "timezone": "UTC",
        "cell_selection": "nearest",
    }
    if tier == "final":
        params["hourly"] = ",".join(HOURLY_VARS)
        params["start_date"] = start.isoformat()
        params["end_date"] = end.isoformat()
        data = _get_json(ARCHIVE_URL, params)
    else:
        params["hourly"] = ",".join(FORECAST_HOURLY_VARS)
        params["past_days"] = min(92, max(1, (end - start).days + 1))
        params["forecast_days"] = 0
        data = _get_json(FORECAST_URL, params)
    # A single-location request answers with an object; many answer with a list. Asking for
    # one cell is possible when 4,902 is not a multiple of the batch size.
    blocks = data if isinstance(data, list) else [data]
    return [b.get("hourly", {}) for b in blocks]


VALUE_COLS = ["t2m_c", "rh2m_pct", "mslp_hpa", "psfc_hpa", "pwat_kgm2",
              "soil_moisture_pct", "precip_mm", "_u", "_v"]


# Open-Meteo's hourly names -> our daily column, and how a day is reduced.
_HOURLY_TO_DAILY = {
    "temperature_2m": ("t2m_c", "mean", 1.0),
    "relative_humidity_2m": ("rh2m_pct", "mean", 1.0),
    "pressure_msl": ("mslp_hpa", "mean", 1.0),
    "surface_pressure": ("psfc_hpa", "mean", 1.0),
    "total_column_integrated_water_vapour": ("pwat_kgm2", "mean", 1.0),
    "soil_moisture_0_to_7cm": ("soil_moisture_pct", "mean", 100.0),
    "precipitation": ("precip_mm", "sum", 1.0),
}


def _daily_for_batch(blocks: list) -> dict:
    """One request's cells reduced to whole days, as arrays over the batch.

    Every location in an Open-Meteo response shares one time axis, so this is array work:
    stack each variable into (n_cells, n_hours) once and reduce along the hour axis. Doing
    it per cell with a pandas groupby instead cost 4,902 groupbys per day - measured at
    173 s for the seven tests in tests/test_live_district_observations.py, against 2.0 s
    for the request that produced the data.

    Wind is carried as u/v components, never as a bearing. A mean of bearings is wrong
    across the 0/360 wrap - 350 and 10 degrees average to 180, pointing the opposite way -
    and wrong again when cells are combined in `_aggregate_day`. The forecast side
    aggregates components and derives speed and direction last; this matches it, because
    the two sides of the bust label have to be commensurable to be subtracted.

    A day with fewer than 24 hours is dropped, not averaged: a partial mean is a wrong
    number rather than a missing one.

    Returns {date: {daily column: array over the batch's cells}}.
    """
    n = len(blocks)
    ref = next((b.get("time") for b in blocks if b and b.get("time")), None)
    if not ref:
        return {}
    hours = len(ref)
    times = pd.to_datetime(pd.Series(ref))
    days = times.dt.floor("D").dt.date.to_numpy()

    # Which cells in this batch returned a usable, correctly-sized hourly block. A cell
    # whose axis differs from the rest of its own response is dropped rather than aligned:
    # guessing which hours it meant would be fabricating data.
    usable = [i for i, b in enumerate(blocks)
              if b and len(b.get("time", ())) == hours]

    def matrix(name: str):
        """(n_cells, n_hours) of one hourly variable, NaN where a cell has no value.

        Built with one `np.array` call over the raw lists rather than a `pd.Series` per
        cell: at 300 cells x 9 variables x 17 batches that was ~46,000 Series
        constructions per day, and numpy already maps JSON `null` to NaN for a float
        dtype.
        """
        rows = [blocks[i].get(name) for i in usable]
        keep = [i for i, v in zip(usable, rows) if v is not None and len(v) == hours]
        if not keep:
            return None
        m = np.full((n, hours), np.nan, dtype=float)
        m[keep] = np.array([blocks[i][name] for i in keep], dtype=float)
        return m

    mats = {}
    for src, (dest, how, scale) in _HOURLY_TO_DAILY.items():
        m = matrix(src)
        if m is not None:
            mats[dest] = (m * scale, how)

    ws, wd = matrix("wind_speed_10m"), matrix("wind_direction_10m")
    if ws is not None and wd is not None:
        rad = np.deg2rad(wd)
        mats["_u"] = (-ws * np.sin(rad), "mean")
        mats["_v"] = (-ws * np.cos(rad), "mean")

    out: dict = {}
    for day in pd.unique(days):
        cols = days == day
        if int(cols.sum()) < 24:
            continue
        slot = {}
        for dest, (m, how) in mats.items():
            block = m[:, cols]
            with warnings.catch_warnings():
                # An all-NaN cell is a cell with no data, which is the documented case the
                # aggregator renormalises around - not a condition worth a warning per row.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                slot[dest] = (np.nansum(block, axis=1) if how == "sum"
                              else np.nanmean(block, axis=1))
            # nansum turns an all-missing cell into 0.0; missing never becomes zero.
            if how == "sum":
                slot[dest] = np.where(np.isnan(block).all(axis=1), np.nan, slot[dest])
        out[day] = slot
    return out


def _aggregate_day(per_cell: dict) -> pd.DataFrame:
    """Whole-grid values for one date -> one area-weighted row per district.

    `per_cell` maps a value column to an array over all 4,902 cells. Cells that returned
    nothing are NaN and the aggregator drops them, renormalising over what is left, so a
    district covered by a failed batch reports NaN rather than a number built from a
    fraction of its area.
    """
    agg = get_aggregator()
    index = _prepared_index()
    out = _district_identity().copy()
    for col in VALUE_COLS:
        flat = per_cell.get(col)
        if flat is None or np.isnan(flat).all():
            continue
        out[col] = agg.aggregate_prepared(index, flat).to_numpy(dtype=float)

    if {"_u", "_v"}.issubset(out.columns):
        u, v = out["_u"].to_numpy(float), out["_v"].to_numpy(float)
        out["wspd10m_ms"] = np.sqrt(u ** 2 + v ** 2)
        out["wdir10m_deg"] = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
        out = out.drop(columns=["_u", "_v"])
    return out


def fetch_observations(start: date, end: date, tier: str = "final") -> tuple:
    """Area-weighted district observations for every district, from the gridded cells.

    This used to loop over 36 city points, one request each. It now batches the weight
    table's 4,902 cells and aggregates them through the same table the forecasts use, so
    both sides of the bust label are area means over the same polygon.
    """
    if tier not in ("final", "provisional"):
        raise ValueError(f"unknown tier {tier!r}")

    lats, lons = _cells()
    n_cells = len(lats)
    report = ObsReport(tier=tier, start=start, end=end, cells=n_cells)
    t0 = time.time()

    # date -> value column -> array over all 4,902 cells. One day across nine variables
    # is a few hundred thousand floats, so this is held whole rather than streamed; the
    # fetch window is days, never a year.
    by_date: dict = {}

    def slot_for(day, col):
        d = by_date.setdefault(day, {})
        if col not in d:
            d[col] = np.full(n_cells, np.nan, dtype=float)
        return d[col]

    for begin in range(0, n_cells, CELL_BATCH):
        sl = slice(begin, min(begin + CELL_BATCH, n_cells))
        try:
            blocks = _fetch_cell_batch(lats[sl], lons[sl], start, end, tier)
        except Exception as exc:  # noqa: BLE001
            log.warning("observation batch %d-%d failed: %s", sl.start, sl.stop - 1, exc)
            report.failures.append(f"cells {sl.start}-{sl.stop - 1}: {exc}")
            time.sleep(POLITE_GAP_S)
            continue

        for day, cols in _daily_for_batch(blocks).items():
            if tier == "provisional" and not (start <= day <= end):
                continue
            for col, values in cols.items():
                slot_for(day, col)[sl] = values
        time.sleep(POLITE_GAP_S)

    report.seconds = time.time() - t0
    if not by_date:
        return pd.DataFrame(), report

    source = (
        "ERA5 reanalysis via Open-Meteo archive API (CC-BY 4.0; Copernicus C3S), "
        "area-weighted per district"
        if tier == "final" else
        "Near-real-time analysis via Open-Meteo (CC-BY 4.0), area-weighted per district "
        "- PROVISIONAL, subject to revision"
    )
    frames = []
    for day in sorted(by_date):
        block = _aggregate_day(by_date[day])
        block.insert(6, "date", day)
        block["source"] = source
        frames.append(block)

    out = pd.concat(frames, ignore_index=True)
    value_cols = [c for c in out.columns
                  if c not in ("region_id", "region_name", "state_id", "state_name",
                               "latitude", "longitude", "date", "source")]
    out = out.dropna(subset=value_cols, how="all").reset_index(drop=True)
    report.rows = len(out)
    return out, report


def write_observations_csv(frame: pd.DataFrame, tier: str, start: date, end: date,
                           out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"observations_{tier}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    frame.to_csv(path, index=False)
    return path


def default_window(days_back: int, tier: str, today: Optional[date] = None) -> tuple:
    today = today or date.today()
    if tier == "final":
        end = today - timedelta(days=5)
    else:
        end = today - timedelta(days=1)
    return end - timedelta(days=days_back), end
