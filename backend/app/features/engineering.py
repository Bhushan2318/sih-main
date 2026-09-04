from __future__ import annotations

import numpy as np
import pandas as pd

FORECAST = "forecast"
OBSERVED = "observed"

_RATE_OF_CHANGE_VARS = {
    "pressure_hpa": "pressure_rate_of_change",
    "atmospheric_moisture_kgm2": "moisture_rate_of_change",
}

_SEASONS = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJAS", 7: "JJAS", 8: "JJAS", 9: "JJAS",
    10: "ON", 11: "ON",
}

EVENT_KEYS = ["region_id", "init_date", "valid_date", "lead_time_days"]
MEMBER_KEYS = EVENT_KEYS + ["ensemble_member_id"]


def _season(month: pd.Series) -> pd.Series:
    return month.map(_SEASONS).astype("category")


def build_training_frame(
    canonical: pd.DataFrame,
    historical_bust_freq: dict | None = None,
    require_observed: bool = True,
) -> pd.DataFrame:
    df = canonical.copy()
    df = df[df["region_id"].notna()]
    fc = df[df["value_type"] == FORECAST].copy()
    ob_cols = ["region_id", "valid_date", "variable", "value"]
    has_vs = "verification_status" in df.columns
    if has_vs:
        ob_cols.append("verification_status")
    ob = df[df["value_type"] == OBSERVED][ob_cols].copy()
    agg = {"value": "mean"}
    if has_vs:
        agg["verification_status"] = (
            lambda s: "provisional" if (s == "provisional").any() else "final"
        )
    ob = (ob.groupby(["region_id", "valid_date", "variable"], as_index=False)
            .agg(agg).rename(columns={"value": "observed_value"}))

    fc = fc.rename(columns={"value": "forecast_value"})
    fc = fc.drop(columns=["verification_status"], errors="ignore")
    paired = fc.merge(
        ob, on=["region_id", "valid_date", "variable"],
        how="inner" if require_observed else "left",
    )
    if paired.empty:
        return paired

    del df, fc, ob

    paired["forecast_value"] = pd.to_numeric(paired["forecast_value"], errors="coerce")
    paired["observed_value"] = pd.to_numeric(paired["observed_value"], errors="coerce")

    sat = (paired["variable"] == "soil_moisture_pct") & (paired["forecast_value"] >= 99.5)
    paired = paired[~sat]

    paired["abs_error"] = (paired["forecast_value"] - paired["observed_value"]).abs()
    required = ["abs_error", "lead_time_days"] if require_observed else ["lead_time_days"]
    paired = paired.dropna(subset=required)
    if paired.empty:
        return paired
    paired["lead_time_days"] = paired["lead_time_days"].astype(int)

    paired["valid_date"] = pd.to_datetime(paired["valid_date"])
    paired["init_date"] = pd.to_datetime(paired["init_date"])
    paired["month"] = paired["valid_date"].dt.month
    paired["season"] = _season(paired["month"])
    paired["region_id"] = paired["region_id"].astype("category")

    grp = paired.groupby(EVENT_KEYS + ["variable"], observed=True)["forecast_value"]
    paired["ensemble_spread"] = grp.transform("std")
    paired["ensemble_member_count"] = grp.transform("count")

    paired = _add_rate_of_change(paired)

    paired = paired.sort_values(MEMBER_KEYS[:-1] + ["variable", "ensemble_member_id", "lead_time_days"])
    paired["forecast_error_lag"] = (
        paired.groupby(["region_id", "init_date", "variable", "ensemble_member_id"],
                       observed=True)["abs_error"].shift(1)
    )

    paired = _add_concurrent_variable_forecasts(paired)

    if historical_bust_freq is not None:
        key = list(zip(paired["region_id"].astype(str), paired["season"].astype(str)))
        paired["historical_bust_frequency_region_season"] = [
            historical_bust_freq.get(k, np.nan) for k in key
        ]
    else:
        paired["historical_bust_frequency_region_season"] = np.nan

    return paired.reset_index(drop=True)


def _add_rate_of_change(paired: pd.DataFrame) -> pd.DataFrame:
    for var, colname in _RATE_OF_CHANGE_VARS.items():
        sub = paired[paired["variable"] == var].sort_values(
            ["region_id", "init_date", "ensemble_member_id", "valid_date"]
        )
        if sub.empty:
            paired[colname] = np.nan
            continue
        grouped = sub.groupby(["region_id", "init_date", "ensemble_member_id"], observed=True)
        days = grouped["valid_date"].diff().dt.days.replace(0, np.nan)
        sub_roc = (grouped["forecast_value"].diff() / days).rename(colname)
        ev = sub.assign(**{colname: sub_roc})[EVENT_KEYS + ["ensemble_member_id", colname]]
        del sub, grouped, days, sub_roc
        paired = paired.merge(ev, on=EVENT_KEYS + ["ensemble_member_id"], how="left")
        del ev
    return paired


def _add_concurrent_variable_forecasts(paired: pd.DataFrame) -> pd.DataFrame:
    wide = (
        paired.pivot_table(
            index=MEMBER_KEYS, columns="variable", values="forecast_value", aggfunc="mean"
        )
        .add_prefix("fc_")
        .reset_index()
    )
    merged = paired.merge(wide, on=MEMBER_KEYS, how="left")
    del wide
    for var in paired["variable"].unique():
        col = f"fc_{var}"
        if col in merged.columns:
            merged.loc[merged["variable"] == var, col] = np.nan
    return merged


def compute_historical_bust_frequency(
    paired_train: pd.DataFrame, large_error_pct: float = 75.0
) -> dict:
    thr = {
        var: np.percentile(g["abs_error"].dropna(), large_error_pct)
        for var, g in paired_train.groupby("variable") if g["abs_error"].notna().any()
    }
    p = paired_train.copy()
    p["is_large"] = [
        row.abs_error > thr.get(row.variable, np.inf) for row in p.itertuples()
    ]
    rate = p.groupby([p["region_id"].astype(str), p["season"].astype(str)])["is_large"].mean()
    return {tuple(k): float(v) for k, v in rate.items()}
