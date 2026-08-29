"""Fetch observations for the Indian city points, in two tiers.

Why two tiers
-------------
ERA5 - the reanalysis the regressors were trained against - lands about five days late.
Waiting for it means a forecast issued today shows no verification for most of a week, and
the trajectory charts stay empty. Open-Meteo's forecast endpoint also serves recent *past*
hours from a near-real-time analysis blend, available within hours.

So observations arrive twice:

  * ``provisional`` - near-real-time analysis. Fills the charts immediately, is badged as
    provisional in the UI, and is **excluded from training**.
  * ``final`` - ERA5/ERA5T from the archive endpoint. Overwrites the provisional row for
    the same (region, valid_date, variable) and is what the models learn from.

Keeping training on a single baseline matters: verifying against a different product
shifts the measured error, and therefore the bust label, enough to drift the model. The
same reasoning is why the forecast side pins itself to one grid resolution and one member
count (see live/gefs.py).

Precipitation caveat
--------------------
Near-real-time precipitation from an analysis blend is materially weaker than ERA5's, more
so than temperature or pressure. Provisional rainfall is therefore the least trustworthy
provisional field; it is still shown, still badged, and still replaced by ERA5 later.

Both endpoints serve ERA5/IFS data under CC-BY 4.0 (Copernicus C3S). No API key.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("forecastguard.live.obs")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

BACKEND_DIR = Path(__file__).resolve().parents[2]
CITIES_JSON = BACKEND_DIR / "scripts" / "india_cities.json"

# Identical variable list and daily reductions to scripts/fetch_era5_observations.py, so
# provisional, final and the original 2019 sample are all the same quantities.
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
# The forecast endpoint exposes a slightly smaller set; TCWV is not among them.
FORECAST_HOURLY_VARS = [v for v in HOURLY_VARS if v != "total_column_integrated_water_vapour"]

HTTP_RETRIES = 5
HTTP_BACKOFF = 3.0
POLITE_GAP_S = 1.2

_session = requests.Session()
_session.headers["User-Agent"] = "ForecastGuard-AI/phase6-live (SIH 2026; NCMRWF PS 26079)"


@dataclass
class ObsReport:
    tier: str
    start: date
    end: date
    cities: int = 0
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


def load_cities() -> pd.DataFrame:
    if not CITIES_JSON.exists():
        raise FileNotFoundError(f"missing city table: {CITIES_JSON}")
    return pd.DataFrame(pd.read_json(CITIES_JSON))


def _to_daily(hourly: dict, city: dict, tier: str) -> pd.DataFrame:
    df = pd.DataFrame(hourly)
    if df.empty or "time" not in df:
        return pd.DataFrame()
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.floor("D")

    # Wind is averaged as vectors: a scalar mean of bearings is wrong across the 0/360
    # wrap (350 deg and 10 deg average to 180, not 0).
    if {"wind_speed_10m", "wind_direction_10m"}.issubset(df.columns):
        wd = np.deg2rad(pd.to_numeric(df["wind_direction_10m"], errors="coerce").to_numpy(float))
        ws = pd.to_numeric(df["wind_speed_10m"], errors="coerce").to_numpy(float)
        df["_u"] = -ws * np.sin(wd)
        df["_v"] = -ws * np.cos(wd)

    g = df.groupby("date")
    cols: dict = {}

    def mean_of(src, dest, scale=1.0):
        if src in df.columns:
            cols[dest] = g[src].mean() * scale

    mean_of("temperature_2m", "t2m_c")
    mean_of("relative_humidity_2m", "rh2m_pct")
    mean_of("pressure_msl", "mslp_hpa")
    mean_of("surface_pressure", "psfc_hpa")
    mean_of("total_column_integrated_water_vapour", "pwat_kgm2")
    # Open-Meteo serves soil moisture as m3/m3 (0..1); the canonical variable is percent
    # volumetric (0..100) to match the GEFS soilw column. Dropping this factor once made
    # the soil-moisture regressor score a meaningless R2 of 1.000.
    mean_of("soil_moisture_0_to_7cm", "soil_moisture_pct", 100.0)
    if "precipitation" in df.columns:
        cols["precip_mm"] = g["precipitation"].sum()
    if "_u" in df.columns:
        cols["_u"] = g["_u"].mean()
        cols["_v"] = g["_v"].mean()
    cols["hours_in_day"] = g.size()

    daily = pd.DataFrame(cols).reset_index()
    if "_u" in daily.columns:
        daily["wspd10m_ms"] = np.sqrt(daily["_u"] ** 2 + daily["_v"] ** 2)
        daily["wdir10m_deg"] = (270.0 - np.degrees(np.arctan2(daily["_v"], daily["_u"]))) % 360.0
        daily = daily.drop(columns=["_u", "_v"])

    # A partial day would produce a daily mean over a handful of hours, which is not the
    # same statistic as a full-day mean. Drop rather than publish a misleading value.
    daily = daily[daily["hours_in_day"] >= 24].drop(columns=["hours_in_day"])
    if daily.empty:
        return daily

    daily.insert(0, "city", city["city"])
    daily.insert(1, "state", city["state"])
    daily.insert(2, "region", city["region"])
    daily.insert(3, "latitude", city["lat"])
    daily.insert(4, "longitude", city["lon"])
    # Deliberately avoids the word "forecast": the schema mapper can read a free-text
    # provenance column as a per-row value_type discriminator, and the earlier wording
    # ("...Open-Meteo forecast-api...") relabelled a whole file of observations as
    # forecasts. The live path also excludes this column explicitly; this is belt and
    # braces, and reads more clearly anyway.
    daily["source"] = (
        "ERA5 reanalysis via Open-Meteo archive API (CC-BY 4.0; Copernicus C3S)"
        if tier == "final" else
        "Near-real-time analysis via Open-Meteo (CC-BY 4.0) - PROVISIONAL, subject to revision"
    )
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    return daily


def fetch_observations(start: date, end: date, tier: str = "final") -> tuple:
    """Daily observations for every city between start and end (inclusive).

    tier="final"       -> ERA5 archive endpoint (~5 day latency, training baseline)
    tier="provisional" -> near-real-time analysis (hours, excluded from training)
    """
    if tier not in ("final", "provisional"):
        raise ValueError(f"unknown tier {tier!r}")

    cities = load_cities()
    report = ObsReport(tier=tier, start=start, end=end, cities=len(cities))
    frames = []
    t0 = time.time()

    for _, city in cities.iterrows():
        params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "windspeed_unit": "ms",
            "timezone": "UTC",
            "cell_selection": "nearest",
        }
        try:
            if tier == "final":
                params["hourly"] = ",".join(HOURLY_VARS)
                data = _get_json(ARCHIVE_URL, params)
            else:
                params["hourly"] = ",".join(FORECAST_HOURLY_VARS)
                # past_days pulls recent history from the analysis; forecast_days=0 keeps
                # future hours out entirely - a forecast must never enter as an observation.
                params["past_days"] = min(92, max(1, (end - start).days + 1))
                params["forecast_days"] = 0
                params.pop("start_date"), params.pop("end_date")
                data = _get_json(FORECAST_URL, params)
        except Exception as exc:  # noqa: BLE001
            log.warning("observation fetch failed for %s: %s", city["city"], exc)
            report.failures.append(str(city["city"]))
            continue

        daily = _to_daily(data.get("hourly", {}), city.to_dict(), tier)
        if not daily.empty:
            if tier == "provisional":
                daily = daily[(daily["date"] >= start) & (daily["date"] <= end)]
            frames.append(daily)
        time.sleep(POLITE_GAP_S)

    report.seconds = time.time() - t0
    if not frames:
        return pd.DataFrame(), report

    out = pd.concat(frames, ignore_index=True)
    report.rows = len(out)
    return out, report


def write_observations_csv(frame: pd.DataFrame, tier: str, start: date, end: date,
                           out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"observations_{tier}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    frame.to_csv(path, index=False)
    return path


def default_window(days_back: int, tier: str, today: Optional[date] = None) -> tuple:
    """Date window to refresh for a tier.

    The final tier ends five days ago because ERA5 has not been published for anything
    more recent; asking for it returns nothing rather than an error, which would look like
    a silent gap.
    """
    today = today or date.today()
    if tier == "final":
        end = today - timedelta(days=5)
    else:
        end = today - timedelta(days=1)   # yesterday is the last complete UTC day
    return end - timedelta(days=days_back), end
