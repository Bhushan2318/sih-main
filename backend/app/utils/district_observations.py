"""Turning 0.25 deg cell readings into district values - shared by every observation fetch.

There is exactly one district weight table (docs/engineering-reference.md), and therefore exactly one place
that reads it. Both the Open-Meteo fetch and the CDS fetch import from here rather than
keeping their own copy of the aggregation, because two copies drift and the drift is
invisible: both would keep producing plausible district numbers.

The canonical column set is the forecast side's, so the two halves of a bust label join
without a translation layer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.utils import india_districts as idist

VALUE_COLUMNS = [
    "t2m_c", "rh2m_pct", "precip_mm", "mslp_hpa", "psfc_hpa",
    "wspd10m_ms", "wdir10m_deg", "soil_moisture_pct", "pwat_kgm2",
]

# Direction is circular: a plain mean of 350 and 10 degrees is 180, which points the
# opposite way. Anything listed here is averaged as a unit vector instead.
CIRCULAR_COLUMNS = {"wdir10m_deg"}


def grid_cells() -> pd.DataFrame:
    """The distinct 0.25 deg cells the district weight table draws on."""
    w = pd.read_parquet(idist.geo_dir() / idist.WEIGHTS_FILENAME)
    cells = w[["lat", "lon"]].drop_duplicates().sort_values(["lat", "lon"])
    return cells.reset_index(drop=True)


def to_districts(cell_rows: pd.DataFrame, cells: pd.DataFrame,
                 value_columns: list[str] | None = None) -> pd.DataFrame:
    """Area-weighted mean over each district's cells, one value per (district, date).

    The same aggregator and the same weights the forecast side uses. NaN cells drop out
    and the remaining weights renormalise, so a coastal district is not voided by the sea
    cells it overlaps; a district with no valid cell for a variable stays NaN - never a
    fabricated zero.
    """
    value_columns = list(value_columns or VALUE_COLUMNS)
    agg = idist.get_aggregator()
    prepared = agg.prepare(cells["lat"].to_numpy(), cells["lon"].to_numpy())

    # Cell order must match `cells`, and every cell must be present for every date, or a
    # value would silently line up against the wrong location.
    pivot_index = pd.MultiIndex.from_frame(cells[["lat", "lon"]])
    out = []
    for date, chunk in cell_rows.groupby("date", sort=True):
        chunk = chunk.set_index(["lat", "lon"]).reindex(pivot_index)
        series = {}
        for col in value_columns:
            if col in CIRCULAR_COLUMNS:
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


# A cell counts as land for soil moisture when ERA5's own land-sea mask says it is at
# least half land. ERA5 has no soil over water (swvl1 reads ~0 there), so a coastal
# district's soil value would otherwise be dragged toward zero by its sea cells.
LAND_FRACTION_MIN = 0.5


def to_districts_v2(cells_daily: pd.DataFrame, cells: pd.DataFrame,
                    land_sea_mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Estimator v2: district values from per-cell daily components, one per (district, date).

    The forecast side's order (scripts/fetch_gefs_reforecast_sample.py `_canonicalise`):
    average the components over the district first - q, T, surface pressure, u, v - and
    derive RH, wind speed and direction from those means, with the same functions
    (app/utils/humidity.py). RH and speed are non-linear, so deriving them per cell first,
    as v1 `to_districts` is fed, gives a different number for the same air.

    `cells_daily` has the columns `to_cells_daily` writes. With `land_sea_mask`
    (lat, lon, lsm for every cell), soil uses land cells only; everything else uses every
    cell. Same aggregator and weights as the forecast side; NaN never becomes zero.
    """
    from app.utils import humidity

    soil = cells_daily["swvl1_m3m3"].to_numpy(dtype=float).copy()
    if land_sea_mask is not None:
        lsm = land_sea_mask.set_index([np.round(land_sea_mask.lat, 4),
                                       np.round(land_sea_mask.lon, 4)])["lsm"]
        key = pd.MultiIndex.from_arrays([np.round(cells_daily.lat, 4),
                                         np.round(cells_daily.lon, 4)])
        frac = lsm.reindex(key).to_numpy(dtype=float)
        if np.isnan(frac).any():
            raise ValueError(f"the land-sea mask is missing {int(np.isnan(frac).sum())} "
                             "cell-days' cells")
        soil[frac < LAND_FRACTION_MIN] = np.nan
    comp = cells_daily[["lat", "lon", "date"]].copy()
    for col in ("t2m_k", "q2m_kgkg", "sp_pa", "msl_pa", "u10_ms", "v10_ms",
                "tcwv_kgm2", "tp_mm"):
        comp[col] = cells_daily[col].to_numpy(dtype=float)
    comp["swvl1_land"] = soil

    cols = ["t2m_k", "q2m_kgkg", "sp_pa", "msl_pa", "u10_ms", "v10_ms",
            "tcwv_kgm2", "tp_mm", "swvl1_land"]
    means = to_districts(comp, cells, value_columns=cols)

    spd, drc = humidity.wind_speed_dir(means["u10_ms"].to_numpy(), means["v10_ms"].to_numpy())
    out = pd.DataFrame({
        "region_id": means["region_id"],
        "date": means["date"],
        "t2m_c": means["t2m_k"] - 273.15,
        "rh2m_pct": humidity.rh_from_specific_humidity(
            means["q2m_kgkg"].to_numpy(), means["t2m_k"].to_numpy(),
            means["sp_pa"].to_numpy()),
        "precip_mm": means["tp_mm"],
        "mslp_hpa": means["msl_pa"] / 100.0,
        "psfc_hpa": means["sp_pa"] / 100.0,
        "wspd10m_ms": spd,
        "wdir10m_deg": drc,
        "soil_moisture_pct": means["swvl1_land"] * 100.0,
        "pwat_kgm2": means["tcwv_kgm2"],
    })
    return out[["region_id", "date"] + VALUE_COLUMNS]


def district_metadata() -> pd.DataFrame:
    """region_id -> name, state and centroid, for joining onto fetched values."""
    return pd.DataFrame([
        {"region_id": d.region_id, "region_name": d.region_name,
         "state_id": d.state_id, "state_name": d.state_name,
         "latitude": d.centroid_lat, "longitude": d.centroid_lon}
        for d in idist.load_registry()
    ])
