from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from app.utils import district_observations, humidity
from app.utils.district_observations import CELL_DAILY_COLUMNS
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
    "dew_point_2m",
    "precipitation",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "soil_moisture_0_to_7cm",
    "total_column_integrated_water_vapour",
]
FORECAST_HOURLY_VARS = [v for v in HOURLY_VARS if v != "total_column_integrated_water_vapour"]

# The training side's day (scripts/fetch_era5_cds_district_observations.py `to_cells_daily`):
# the stamps (t-24h, t] - 01 UTC to 00 UTC the next day, because Open-Meteo stamps an
# accumulation at the end of its hour - with instantaneous fields taken at the eight
# 3-hourly instants the forecast's own day is built from.
INSTANT_STEP_HOURS = 3
INSTANTS_PER_DAY = 24 // INSTANT_STEP_HOURS

HTTP_RETRIES = 5
HTTP_BACKOFF = 3.0
POLITE_GAP_S = 1.2
# Open-Meteo's free tier counts a multi-location request as one call per location, and
# answers HTTP 429 once a minute's allowance is spent; a 3-15 s backoff cannot outlast
# that (refresh-data.yml 2026-10-04 06:11Z: 5 of 17 batches lost that way). A 429 for the
# minute waits it out; one for the hour or the day fails the batch at once.
RATE_LIMIT_WAIT_S = 61.0

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
            if r.status_code == 429:
                if "minute" not in r.text.lower():
                    # An hourly or daily allowance: a minute's wait cannot outlast it.
                    raise RuntimeError(f"observation request refused: {last}")
                time.sleep(RATE_LIMIT_WAIT_S)
                continue
        except requests.RequestException as exc:
            last = str(exc)
        time.sleep(HTTP_BACKOFF * (attempt + 1))
    raise RuntimeError(f"observation request failed after {HTTP_RETRIES} tries: {last}")


@functools.lru_cache(maxsize=1)
def _cells() -> tuple:
    """The 0.25 degree cells India overlaps, in one fixed order.

    These are the weight table's own cells - 4,902 of them - not the 666 district
    centroids. A centroid is a point sample, and pairing a point observation against an
    area-mean forecast is the mismatch the single-weight-table rule (docs/engineering-reference.md) exists to
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
    """One request for up to CELL_BATCH cells; returns each cell's hourly block in order.

    `models=era5` makes the archive answer from ERA5's own 0.25 deg cell. Without it the
    answer comes from Open-Meteo's default model mix on a finer grid - measured 2026-10-04,
    19.0 N 72.75 E came back from 19.016 N 72.781 E, 1.9 C colder than ERA5's cell at
    01 UTC on 2026-09-20. ERA5 has no near-real-time feed, so the provisional tier keeps
    the forecast API's default and says so in its source. `elevation=nan` (one per
    coordinate) stops Open-Meteo shifting temperature to a 90 m terrain model, on both
    tiers.
    """
    params = {
        "latitude": ",".join(f"{v:.4f}" for v in lats),
        "longitude": ",".join(f"{v:.4f}" for v in lons),
        "elevation": ",".join("nan" for _ in lats),
        "windspeed_unit": "ms",
        "timezone": "UTC",
        "cell_selection": "nearest",
    }
    if tier == "final":
        params["models"] = "era5"
        params["hourly"] = ",".join(HOURLY_VARS)
        params["start_date"] = start.isoformat()
        # The last day ends at 00 UTC the day after.
        params["end_date"] = (end + timedelta(days=1)).isoformat()
        data = _get_json(ARCHIVE_URL, params)
    else:
        params["hourly"] = ",".join(FORECAST_HOURLY_VARS)
        params["past_days"] = min(92, max(1, (end - start).days + 1))
        # Today's 00 UTC stamp closes yesterday; later hours fall on today, which is outside
        # the window and dropped.
        params["forecast_days"] = 1
        data = _get_json(FORECAST_URL, params)
    # A single-location request answers with an object; many answer with a list. Asking for
    # one cell is possible when 4,902 is not a multiple of the batch size.
    blocks = data if isinstance(data, list) else [data]
    return [b.get("hourly", {}) for b in blocks]


# Open-Meteo's hourly name -> the per-cell daily component, its unit conversion, and
# whether the day sums it (rain) or averages its instants (everything else).
_INSTANT = {
    "temperature_2m": ("t2m_k", lambda x: x + 273.15),
    "surface_pressure": ("sp_pa", lambda x: x * 100.0),
    "pressure_msl": ("msl_pa", lambda x: x * 100.0),
    "total_column_integrated_water_vapour": ("tcwv_kgm2", lambda x: x),
    "soil_moisture_0_to_7cm": ("swvl1_m3m3", lambda x: x),
}


def _daily_for_batch(blocks: list) -> dict:
    """One request's cells reduced to per-cell daily components, as arrays over the batch.

    The components and the reduction are the training side's (`to_cells_daily`), so the
    live and the training observations are the same estimator: q from the dewpoint and
    surface pressure at each instant; T, p, q, u, v, column water vapour and soil averaged
    over the eight 3-hourly instants 03-24 UTC; rain summed over all 24 hourly stamps
    01-24 UTC. RH and wind are not formed here - they are derived after the district mean
    (`district_observations.to_districts_v2`), as the forecast side does.

    Every location in an Open-Meteo response shares one time axis, so this is array work:
    stack each variable into (n_cells, n_hours) once and reduce along the hour axis. Doing
    it per cell with a pandas groupby instead cost 4,902 groupbys per day - measured at
    173 s for the seven tests in tests/test_live_district_observations.py, against 2.0 s
    for the request that produced the data.

    Wind arrives as speed and bearing and is turned into u/v per hour before anything is
    averaged: a mean of bearings is wrong across the 0/360 wrap.

    A day without all 24 stamps is dropped, and a cell missing any instant or hour of a
    variable is NaN for that variable that day: a partial mean is a wrong number rather
    than a missing one.

    Returns {date: {component: array over the batch's cells}}.
    """
    n = len(blocks)
    ref = next((b.get("time") for b in blocks if b and b.get("time")), None)
    if not ref:
        return {}
    hours = len(ref)
    times = pd.to_datetime(pd.Series(ref))
    days = (times - pd.Timedelta(hours=1)).dt.date.to_numpy()
    is_instant = (times.dt.hour % INSTANT_STEP_HOURS == 0).to_numpy()

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

    inst: dict = {}
    for src, (dest, convert) in _INSTANT.items():
        m = matrix(src)
        if m is not None:
            inst[dest] = convert(m)
    td, sp = matrix("dew_point_2m"), matrix("surface_pressure")
    if td is not None and sp is not None:
        inst["q2m_kgkg"] = humidity.specific_humidity_from_dewpoint(td + 273.15, sp * 100.0)
    ws, wd = matrix("wind_speed_10m"), matrix("wind_direction_10m")
    if ws is not None and wd is not None:
        rad = np.deg2rad(wd)
        inst["u10_ms"] = -ws * np.sin(rad)
        inst["v10_ms"] = -ws * np.cos(rad)
    rain = matrix("precipitation")

    out: dict = {}
    for day in pd.unique(days):
        cols = days == day
        if int(cols.sum()) != 24:
            continue
        at = cols & is_instant
        if int(at.sum()) != INSTANTS_PER_DAY:
            continue
        slot = {}
        for dest, m in inst.items():
            block = m[:, at]
            # Not nanmean: a cell missing an instant has no value for the day.
            slot[dest] = np.where(np.isnan(block).any(axis=1), np.nan, block.mean(axis=1))
        if rain is not None:
            block = rain[:, cols]
            slot["tp_mm"] = np.where(np.isnan(block).any(axis=1), np.nan, block.sum(axis=1))
        out[day] = slot
    return out


# District value -> the components it is built from; a district missing any of them in any
# of its cells (soil: its land cells) has no value for it.
_DEPENDS = {
    "t2m_c": ["t2m_k"],
    "rh2m_pct": ["q2m_kgkg", "t2m_k", "sp_pa"],
    "precip_mm": ["tp_mm"],
    "mslp_hpa": ["msl_pa"],
    "wspd10m_ms": ["u10_ms", "v10_ms"],
    "wdir10m_deg": ["u10_ms", "v10_ms"],
    "soil_moisture_pct": ["swvl1_m3m3"],
    "pwat_kgm2": ["tcwv_kgm2"],
}


def _districts_for_day(per_cell: dict, day) -> pd.DataFrame:
    """Whole-grid components for one date -> one row per district, estimator v2.

    `per_cell` maps a component to an array over all 4,902 cells (NaN where a batch failed
    or a cell lacked data). The aggregator renormalises around NaN cells, which is right
    for soil over sea and wrong for a gap: a district covered partly by a failed batch
    would report the mean of the part that came back. So a district with any weight on a
    cell missing a component it needs is set to NaN for that value.
    """
    lats, lons = _cells()
    n = len(lats)
    cells = pd.DataFrame({"lat": lats, "lon": lons})
    frame = cells.copy()
    frame["date"] = day
    for col in CELL_DAILY_COLUMNS:
        frame[col] = per_cell.get(col, np.full(n, np.nan))
    lsm = district_observations.era5_land_sea_mask()
    out = district_observations.to_districts_v2(frame, cells, land_sea_mask=lsm)
    # Open-Meteo's surface_pressure is its MSLP reduced to its own terrain height, not
    # ERA5's sp (measured 2026-10-04: +0.8 hPa at 629 m, +4.3 at 4,835 m). It is not
    # reported. It still turns the dewpoint into q, where 1 hPa moves q by ~0.1%, and RH,
    # which depends on it only through that q.
    out["psfc_hpa"] = np.nan

    land = (lsm.set_index([lsm.lat.round(4), lsm.lon.round(4)])["lsm"]
            .reindex(pd.MultiIndex.from_arrays([np.round(lats, 4), np.round(lons, 4)]))
            .to_numpy(dtype=float) >= district_observations.LAND_FRACTION_MIN)
    agg, index = get_aggregator(), _prepared_index()
    touched = {}
    for col in CELL_DAILY_COLUMNS:
        values = frame[col].to_numpy(dtype=float)
        if np.isnan(values).all():
            continue            # not requested on this tier; the value stays NaN anyway
        gap = np.isnan(values)
        if col == "swvl1_m3m3":
            gap &= land         # sea cells are left out on purpose, not missing
        share = agg.aggregate_prepared(index, gap.astype(float))
        touched[col] = share.reindex(out["region_id"]).to_numpy(dtype=float) > 0
    for value, needs in _DEPENDS.items():
        void = np.zeros(len(out), dtype=bool)
        for c in needs:
            if c in touched:
                void |= touched[c]
        out.loc[void, value] = np.nan
    return out


def fetch_observations(start: date, end: date, tier: str = "final") -> tuple:
    """Area-weighted district observations for every district, from the gridded cells.

    This used to loop over 36 city points, one request each. It now batches the weight
    table's 4,902 cells and aggregates them through the same table the forecasts use, so
    both sides of the bust label are area means over the same polygon - and, since
    2026-10-04, with the training observations' estimator (v2), on ERA5 itself.
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
            # Each request runs a day past `end` (its 00 UTC closes `end`), and the
            # provisional one into today; days outside the window are not this run's.
            if not (start <= day <= end):
                continue
            for col, values in cols.items():
                slot_for(day, col)[sl] = values
        time.sleep(POLITE_GAP_S)

    report.seconds = time.time() - t0
    if not by_date:
        return pd.DataFrame(), report

    estimator = ("estimator v2: daily means of q, T, sp, u, v at the forecast's 3-hourly "
                 "instants, RH and wind derived after the district mean, soil over ERA5 "
                 "land cells")
    source = (
        "ERA5 reanalysis via Open-Meteo archive API (models=era5; CC-BY 4.0; Copernicus "
        f"C3S), area-weighted per district; {estimator}"
        if tier == "final" else
        "Near-real-time analysis via Open-Meteo forecast API (CC-BY 4.0), area-weighted "
        f"per district; {estimator} - PROVISIONAL, not ERA5, subject to revision"
    )
    identity = _district_identity()
    frames = []
    for day in sorted(by_date):
        block = identity.merge(_districts_for_day(by_date[day], day).drop(columns="date"),
                               on="region_id", how="left")
        block.insert(6, "date", day)
        block["source"] = source
        frames.append(block)

    out = pd.concat(frames, ignore_index=True)
    value_cols = [c for c in out.columns
                  if c not in ("region_id", "region_name", "state_id", "state_name",
                               "latitude", "longitude", "date", "source")]
    # A variable this tier does not carry (column water vapour on the forecast API) is
    # left out rather than written as an empty column.
    out = out.drop(columns=[c for c in value_cols if out[c].isna().all()])
    value_cols = [c for c in value_cols if c in out.columns]
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
