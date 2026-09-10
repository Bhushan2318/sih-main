"""Shared shapes, frozen so parallel workstreams cannot drift apart.

Three things are consumed across workstream boundaries: the paired-row frame the trainer
eats, the /api/model/status payload the dashboard reads, and the region panel that carries
SHAP. The API halves live in `app/api/schemas.py` as Pydantic models already; what was
missing was the paired frame, which is a pandas contract rather than a response body.

Everything here is descriptive. Nothing in this module transforms data - it says what the
shape must be and refuses anything else, so a change shows up as one loud failure in
tests/test_contracts.py instead of quietly downstream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Measured against the real 2017 store on 2026-09-10 by running
# fe.build_training_frame over one cycle: 10,695 rows, 28 columns. Not transcribed from
# the feature code - printed from the frame it actually produces.
PAIRED_ROW_COLUMNS: tuple[str, ...] = (
    "region_id", "variable", "valid_date", "forecast_value", "value_type",
    "init_date", "lead_time_days", "ensemble_member_id", "observed_value",
    "verification_status", "abs_error", "month", "season", "ensemble_spread",
    "ensemble_member_count", "pressure_rate_of_change", "moisture_rate_of_change",
    "forecast_error_lag", "fc_atmospheric_moisture_kgm2", "fc_humidity_pct",
    "fc_pressure_hpa", "fc_rainfall_mm", "fc_soil_moisture_pct", "fc_temperature_c",
    "fc_wind_direction_deg", "fc_wind_speed_ms",
    "historical_bust_frequency_region_season",
)

# The identity of one row. A duplicate on these keys means a double-counted member.
EVENT_KEYS: tuple[str, ...] = ("region_id", "init_date", "valid_date", "lead_time_days")
MEMBER_KEYS: tuple[str, ...] = EVENT_KEYS + ("ensemble_member_id",)

# The label is NOT here on purpose. y_bust is applied after the chronological split,
# because the bust threshold is computed on training rows only; a label sitting in the
# paired frame would mean the threshold had already seen the held-out data.
LABEL = "y_bust"

# Expected dtype per column, as a numpy kind rather than an exact dtype: 'f' float,
# 'i' integer, 'M' datetime, 'O' object/string, 'C' pandas categorical. Kinds, because
# int32 vs int64 and category vs object are not worth failing a build over, while a float
# column arriving as text is exactly what must fail.
_KINDS: dict[str, str] = {
    "region_id": "C", "variable": "O", "valid_date": "M", "forecast_value": "f",
    "value_type": "O", "init_date": "M", "lead_time_days": "i",
    "ensemble_member_id": "O", "observed_value": "f", "verification_status": "O",
    "abs_error": "f", "month": "i", "season": "C", "ensemble_spread": "f",
    "ensemble_member_count": "i", "pressure_rate_of_change": "f",
    "moisture_rate_of_change": "f", "forecast_error_lag": "f",
    "fc_atmospheric_moisture_kgm2": "f", "fc_humidity_pct": "f", "fc_pressure_hpa": "f",
    "fc_rainfall_mm": "f", "fc_soil_moisture_pct": "f", "fc_temperature_c": "f",
    "fc_wind_direction_deg": "f", "fc_wind_speed_ms": "f",
    "historical_bust_frequency_region_season": "f",
}

# Columns that are legitimately sparse, and why. Nulls here are real signal, never a
# value to fill: soil moisture is masked past day 3 and wind past day 5 in the archive,
# the lag features have no predecessor on the first cycle, and the historical bust
# frequency is absent until a climatology exists.
NULLABLE: frozenset[str] = frozenset({
    "pressure_rate_of_change", "moisture_rate_of_change", "forecast_error_lag",
    "fc_atmospheric_moisture_kgm2", "fc_humidity_pct", "fc_pressure_hpa",
    "fc_rainfall_mm", "fc_soil_moisture_pct", "fc_temperature_c",
    "fc_wind_direction_deg", "fc_wind_speed_ms",
    "historical_bust_frequency_region_season",
})


def _kind(series: pd.Series) -> str:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "C"
    return series.dtype.kind


def validate_paired_frame(df: pd.DataFrame) -> None:
    """Raise ValueError unless `df` matches the paired-row contract.

    Column order is not enforced - a reordered frame is still perfectly usable, and
    failing on it would be brittle for no gain. Presence and dtype are enforced.
    """
    have, want = set(df.columns), set(PAIRED_ROW_COLUMNS)
    missing, extra = sorted(want - have), sorted(have - want)
    if missing:
        raise ValueError(
            f"paired frame is missing {missing}. If a feature was removed, update "
            f"PAIRED_ROW_COLUMNS in the same commit.")
    if extra:
        raise ValueError(
            f"paired frame has unexpected columns {extra}. A new feature must be added "
            f"to PAIRED_ROW_COLUMNS deliberately, so every consumer sees it change.")

    wrong = []
    for col in PAIRED_ROW_COLUMNS:
        want_kind, got_kind = _KINDS[col], _kind(df[col])
        if want_kind == "C" and got_kind == "O":
            continue          # a category that has not been made categorical yet
        if want_kind == "i" and got_kind == "f":
            continue          # an integer column carrying NaN is float in pandas
        if got_kind != want_kind:
            wrong.append(f"{col} (expected {want_kind}, got {got_kind})")
    if wrong:
        raise ValueError(f"paired frame has the wrong dtype for: {', '.join(wrong)}")


def example_column(name: str, n: int) -> pd.Series:
    """A contract-conforming column of arbitrary values, for tests.

    Shape fixture only. These values are meaningless and must never reach anything that
    produces a metric.
    """
    kind = _KINDS[name]
    if kind == "M":
        return pd.Series(pd.date_range("2017-01-01", periods=n, freq="D"))
    if kind == "f":
        return pd.Series(np.zeros(n, dtype="float64"))
    if kind == "i":
        return pd.Series(np.ones(n, dtype="int64"))
    if kind == "C":
        return pd.Series([f"{name}_{i}" for i in range(n)], dtype="category")
    return pd.Series([f"{name}_{i}" for i in range(n)], dtype="object")
