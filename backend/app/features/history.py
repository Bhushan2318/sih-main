"""Earlier forecast cycles, for the jumpiness features (C1).

Shared by the chunked trainer and the one-cycle scorer, so the two cannot disagree about
which cycles count as history. Kept apart from engineering.py, which never touches the
store.
"""
from __future__ import annotations

from datetime import date
from typing import Sequence

import pandas as pd

from app.features import engineering as fe
from app.storage import parquet_store

_READ_COLUMNS = ["region_id", "variable", "valid_date", "init_date", "value", "value_type"]


def forecast_history(first_init, store_inits: Sequence[date],
                     exclude_provisional: bool = False) -> pd.DataFrame:
    """Ensemble-mean trajectories from every cycle issued in the JUMP_LOOKBACK_DAYS before
    `first_init`, restricted to the valid dates `first_init` itself forecasts.

    One cycle is read and reduced to its ensemble means before the next is read, so what is
    held is never more than one cycle of member rows plus trajectories - five members fold
    into one row. `store_inits` comes from the caller (Parquet footers at serve time), so
    nothing here scans the store to find cycles. At the reforecast's sampled density no
    cycle is that close, and this reads nothing at all.
    """
    first = pd.Timestamp(first_init).normalize()
    lo = first - pd.Timedelta(days=fe.JUMP_LOOKBACK_DAYS)
    earlier = sorted({pd.Timestamp(c).normalize() for c in store_inits
                      if lo <= pd.Timestamp(c).normalize() < first})
    parts = []
    for init in earlier:
        rows = parquet_store.read_dataset(
            value_types=["forecast"], columns=_READ_COLUMNS, init_dates=[init.date()],
            valid_date_min=first.date(), exclude_provisional=exclude_provisional,
        )
        rows = fe.drop_beyond_archive_leads(rows)
        if not rows.empty:
            parts.append(fe.forecast_trajectories(rows))
        del rows
    if not parts:
        return pd.DataFrame(columns=fe.TRAJECTORY_COLUMNS)
    return pd.concat(parts, ignore_index=True)
