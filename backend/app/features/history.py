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

_READ_COLUMNS = [
    "region_id", "cycle_hour", "variable", "valid_date", "init_date", "value", "value_type",
]


def forecast_history(first_init, store_inits: Sequence,
                     exclude_provisional: bool = False,
                     first_cycle_hour: int | None = None) -> pd.DataFrame:
    """Read causal ensemble trajectories before one exact forecast cycle.

    ``store_inits`` may contain dates (the legacy date-only caller) or ``(date, hour)``
    cycle identities. When ``first_cycle_hour`` is supplied, cycles issued earlier on the
    same date are included; without it the historical date-only behavior is preserved so
    a multi-cycle training chunk never receives a later same-day cycle as history.
    """
    first = pd.Timestamp(first_init).normalize()
    first_ts = first + pd.Timedelta(hours=int(first_cycle_hour or 0))
    # The lookback is a date window (a cycle on day -9 can still cover the target's
    # day-10 valid date); only the same-day cutoff needs exact hour identity.
    lo = first - pd.Timedelta(days=fe.JUMP_LOOKBACK_DAYS)
    parts = []
    for item in store_inits:
        if isinstance(item, tuple) and len(item) == 2:
            init = pd.Timestamp(item[0]).normalize()
            hour = int(item[1])
        else:
            init = pd.Timestamp(item).normalize()
            hour = None
        cycle_ts = init + pd.Timedelta(hours=hour or 0)
        if cycle_ts < lo or cycle_ts >= first_ts:
            continue
        kwargs = {}
        if hour is not None:
            kwargs["cycle_hours"] = [hour]
        rows = parquet_store.read_dataset(
            value_types=["forecast"], columns=_READ_COLUMNS, init_dates=[init.date()],
            valid_date_min=first.date(), exclude_provisional=exclude_provisional,
            **kwargs,
        )
        if not rows.empty:
            parts.append(fe.forecast_trajectories(rows))
        del rows
    if not parts:
        return pd.DataFrame(columns=fe.TRAJECTORY_COLUMNS)
    return pd.concat(parts, ignore_index=True)
