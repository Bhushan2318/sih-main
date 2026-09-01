"""Turn the canonical long table into the per-member training frame for the error
regressors: one row per (region_id, init_date, valid_date, variable, lead_time_days,
ensemble_member_id), with the regressor target and every feature the plan calls for.

Target
------
abs_error = |forecast_value - observed_value|   after joining forecast rows to the single
observed value for that (region_id, valid_date, variable).

Features
--------
lead_time_days, forecast_value, month, season (cat), region_id (cat),
ensemble_spread, ensemble_member_count,
pressure_rate_of_change, moisture_rate_of_change   (vs. the previous available valid_date,
                                                   same region/member/init/variable),
forecast_error_lag                                 (abs_error at lead-1, same
                                                   init/region/variable/member),
historical_bust_frequency_region_season           (train-split "large error" rate, mapped),
fc_<other variable>                                (that event's forecast of every other
                                                   variable, for the same member).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FORECAST = "forecast"
OBSERVED = "observed"

# variables whose short-term tendency is a useful instability proxy
_RATE_OF_CHANGE_VARS = {
    "pressure_hpa": "pressure_rate_of_change",
    "atmospheric_moisture_kgm2": "moisture_rate_of_change",
}

_SEASONS = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJAS", 7: "JJAS", 8: "JJAS", 9: "JJAS",   # Indian monsoon
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
    """Per-member frame with the regressor target + every feature.

    `require_observed=True` (training): inner-join to observations, so every row has a
    real `abs_error` target.
    `require_observed=False` (inference): left-join, keeping forecast rows that have not
    verified yet. `abs_error` is then NaN, and so is `forecast_error_lag` for any lead
    whose earlier lead has not verified either - XGBoost routes those missing values
    natively. Callers should surface that via the per-variable confidence rather than
    pretending the feature was present.

    `historical_bust_freq` is a {(region_id, season): rate} map computed on the training
    split only; pass None on the first pass and re-attach later to avoid leakage."""
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
        # Carried through so the UI can badge a value that ERA5 has not settled yet.
        # "provisional" wins the aggregate: if any contributing row is unsettled, the
        # observation as a whole is not yet final.
        agg["verification_status"] = (
            lambda s: "provisional" if (s == "provisional").any() else "final"
        )
    ob = (ob.groupby(["region_id", "valid_date", "variable"], as_index=False)
            .agg(agg).rename(columns={"value": "observed_value"}))

    fc = fc.rename(columns={"value": "forecast_value"})
    # verification_status describes an observation; forecast rows always carry null. Left
    # in place it collides on the merge and both sides come back suffixed _x/_y, so the
    # column the UI reads would silently not exist.
    fc = fc.drop(columns=["verification_status"], errors="ignore")
    paired = fc.merge(
        ob, on=["region_id", "valid_date", "variable"],
        how="inner" if require_observed else "left",
    )
    if paired.empty:
        return paired

    # Everything below works on `paired` alone. Without these deletes, df/fc/ob stay
    # referenced through all three full-frame merges that follow - about 2.2 GB held for
    # no reason on a ten-year store, on top of whatever each merge needs to build its own
    # copy. Purely a lifetime fix: no value in `paired` changes.
    del df, fc, ob

    paired["forecast_value"] = pd.to_numeric(paired["forecast_value"], errors="coerce")
    paired["observed_value"] = pd.to_numeric(paired["observed_value"], errors="coerce")

    # drop physically implausible saturated soil-moisture (>=99.5% volumetric): GEFS pegs
    # its nearest-land grid point to a fill value for coastal/island regions, which would
    # otherwise inject a huge constant regional bias into the soil-moisture regressor.
    sat = (paired["variable"] == "soil_moisture_pct") & (paired["forecast_value"] >= 99.5)
    paired = paired[~sat]

    # abs_error is the regressor TARGET. At inference the forecast has not verified yet,
    # so it is NaN and only the features matter.
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

    # ---- ensemble spread / count across members for the same event+variable ----------
    grp = paired.groupby(EVENT_KEYS + ["variable"], observed=True)["forecast_value"]
    paired["ensemble_spread"] = grp.transform("std")
    paired["ensemble_member_count"] = grp.transform("count")

    # ---- rate-of-change proxies (previous available valid_date) ----------------------
    paired = _add_rate_of_change(paired)

    # ---- forecast_error_lag (previous lead time) ------------------------------------
    paired = paired.sort_values(MEMBER_KEYS[:-1] + ["variable", "ensemble_member_id", "lead_time_days"])
    paired["forecast_error_lag"] = (
        paired.groupby(["region_id", "init_date", "variable", "ensemble_member_id"],
                       observed=True)["abs_error"].shift(1)
    )

    # ---- concurrent forecast of the other variables (same member/event) -------------
    paired = _add_concurrent_variable_forecasts(paired)

    # ---- historical bust frequency (train-only stat, mapped) -----------------------
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
        # Column-wise diff() rather than groupby.apply(): when every group holds a single
        # row - which happens whenever only one lead day has verified, the normal state
        # for a live cycle in its first days - apply() returns a DataFrame instead of a
        # Series and the downstream Series() construction raises.
        grouped = sub.groupby(["region_id", "init_date", "ensemble_member_id"], observed=True)
        days = grouped["valid_date"].diff().dt.days.replace(0, np.nan)
        sub_roc = (grouped["forecast_value"].diff() / days).rename(colname)
        # map that per-event roc onto every row of the same event (all variables)
        ev = sub.assign(**{colname: sub_roc})[EVENT_KEYS + ["ensemble_member_id", colname]]
        # Drop the per-variable slice and its intermediates before the merge allocates the
        # new frame, rather than after: `sub` is a copy of every row of one variable.
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
    # blank out a row's own variable so the model can't read the answer off itself
    for var in paired["variable"].unique():
        col = f"fc_{var}"
        if col in merged.columns:
            merged.loc[merged["variable"] == var, col] = np.nan
    return merged


def compute_historical_bust_frequency(
    paired_train: pd.DataFrame, large_error_pct: float = 75.0
) -> dict:
    """Leakage-safe: on the training split only, the fraction of (variable, row) pairs in
    each (region_id, season) whose abs_error exceeds that variable's `large_error_pct`
    percentile. A coarse 'how often does this place/season go wrong' prior."""
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
