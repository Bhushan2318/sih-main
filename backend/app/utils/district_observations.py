"""Turning 0.25 deg cell readings into district values - shared by every observation fetch.

There is exactly one district weight table (CLAUDE.md), and therefore exactly one place
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


def district_metadata() -> pd.DataFrame:
    """region_id -> name, state and centroid, for joining onto fetched values."""
    return pd.DataFrame([
        {"region_id": d.region_id, "region_name": d.region_name,
         "state_id": d.state_id, "state_name": d.state_name,
         "latitude": d.centroid_lat, "longitude": d.centroid_lon}
        for d in idist.load_registry()
    ])
