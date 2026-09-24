from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from app.config import settings
from app.db.base import resolve_path
from app.features.circular import (
    CIRCULAR_VARIABLES as _CIRCULAR_VARIABLES,
    circular_abs_error,
    normalize_degrees,
)
from app.ingestion.canonical_schema import VARIABLE_PLAUSIBLE_RANGE
from app.utils import india_districts as idist

FORECAST = "forecast"
OBSERVED = "observed"

# C4, district descriptors: static, geometry-derived per-district features that replace
# region_id as a raw model feature - see scripts/build_district_descriptors.py.
DISTRICT_DESCRIPTOR_FEATURES = (
    "state_id", "centroid_lat", "centroid_lon", "area_km2", "border_distance_km",
    "elevation_mean",
)

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

EVENT_KEYS = ["region_id", "cycle_hour", "init_date", "valid_date", "lead_time_days"]
MEMBER_KEYS = EVENT_KEYS + ["ensemble_member_id"]

# C1, forecast jumpiness: how far the forecast for one (district, variable, valid date)
# moved between consecutive initialisations. A forecast that keeps changing its mind is
# one the model itself is unsure of, and that is visible at issue time - it uses only
# cycles issued on or before the row's own init_date, never an observation.
JUMP_FEATURES = ("jump_abs_change", "jump_std", "jump_sign_flips", "jump_rel_climatology")
# Cycles in the std / sign-flip window, the row's own cycle included.
JUMP_WINDOW = 5

# C2, time-lagged ensemble (lagged-average forecasting): a GEFS reforecast cycle carries
# only 5 of the operational feed's 31 members. This cycle's own members are pooled with
# up to LAF_WINDOW-1 earlier cycles' ensemble means, each counted as one extra pooled
# value, for the same (district, variable, valid date) - causal, the same discipline as
# C1's jumpiness.
LAF_FEATURES = ("laf_pool_mean", "laf_pool_std", "laf_pool_size", "laf_spread_ratio")
LAF_WINDOW = JUMP_WINDOW
# A cycle issued this many days earlier still reaches the same valid date: Day 10 of
# init-9 is valid on init (valid_date = init + (lead - 1)). Anything older cannot overlap.
MAX_LEAD_DAYS = 10
JUMP_LOOKBACK_DAYS = MAX_LEAD_DAYS - 1
# Degrees: 350 -> 10 is a 20 degree change, not 340.  The variable set itself is
# imported from the canonical schema above so a future circular variable cannot be added
# to one side of the feature pipeline and forgotten on the other.
_TRAJECTORY_KEYS = ["region_id", "variable", "valid_date", "init_date", "cycle_hour"]
TRAJECTORY_COLUMNS = _TRAJECTORY_KEYS + ["fc_mean"]
# Same mask build_training_frame applies to paired rows: the archive saturates soil
# moisture at 100 where it has no value.
_SOIL_SATURATED = 99.5

# C3, MJO (Madden-Julian Oscillation): a global daily index, not per-district, attached
# by an as-of join on init_date - see scripts/fetch_mjo_index.py for why NOAA PSL's OMI
# rather than BOM's RMM, and for the RMM1/RMM2 transform.
MJO_FEATURES = ("mjo_rmm1", "mjo_rmm2", "mjo_amplitude")
# A forecast issued more than this many days after the last known MJO reading treats it
# as unknown, not stale-but-current - the MJO evolves on a ~30-90 day cycle, so a few
# days' lag is a reasonable "still current" window, not an arbitrary one.
MJO_ASOF_TOLERANCE_DAYS = 5


_PLAUSIBLE_RANGES = {
    (variable.value if hasattr(variable, "value") else str(variable)): bounds
    for variable, bounds in VARIABLE_PLAUSIBLE_RANGE.items()
}


def quarantine_invalid_canonical_values(canonical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split canonical rows into usable values and an explicit quarantine.

    Canonical ingestion can contain text coercion failures, NaN/Inf, and values outside
    the physical range for their variable.  Letting any of those into feature engineering
    either poisons an ensemble mean or becomes NaN only after a derived calculation has
    already moved the information around.  The range contract is therefore enforced again
    at the training boundary, on both forecast and observed rows.

    Returns ``(clean, quarantined)`` without mutating the input.  Quarantined rows are kept
    only for a caller that wants to inspect/report them; the normal training path discards
    them and never trains on their derived features.
    """
    required = {"variable", "value"}
    missing = sorted(required - set(canonical.columns))
    if missing:
        raise ValueError(f"canonical frame is missing {missing}")
    if canonical.empty:
        return canonical.copy(), canonical.iloc[0:0].copy()

    clean = canonical.copy()
    values = pd.to_numeric(clean["value"], errors="coerce").to_numpy(dtype=float)
    variables = clean["variable"].map(
        lambda value: value.value if hasattr(value, "value") else str(value))
    lo = variables.map({var: bounds[0] for var, bounds in _PLAUSIBLE_RANGES.items()})
    hi = variables.map({var: bounds[1] for var, bounds in _PLAUSIBLE_RANGES.items()})
    lo = lo.to_numpy(dtype=float)
    hi = hi.to_numpy(dtype=float)
    valid = (np.isfinite(values) & np.isfinite(lo) & np.isfinite(hi)
             & (values >= lo) & (values <= hi))
    clean["value"] = values
    if bool(valid.all()):
        return clean, clean.iloc[0:0].copy()
    return clean.loc[valid].copy(), clean.loc[~valid].copy()


def quarantine_invalid_paired_values(
    paired: pd.DataFrame, *, require_observed: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quarantine unusable canonical values already present in a paired/cache frame.

    A rebuilt frame has already passed :func:`quarantine_invalid_canonical_values`, but
    pooled training can reuse a year cache.  Applying the same boundary check on every
    read keeps an older or externally produced cache from feeding the regressor bad
    targets/features.  Missing derived concurrent forecasts are legitimate sparse data;
    infinity and finite out-of-range canonical values are not.
    """
    if paired.empty:
        return paired, paired.copy()
    if "variable" not in paired.columns:
        raise ValueError("paired frame is missing ['variable']")

    bad = np.zeros(len(paired), dtype=bool)
    variables = paired["variable"].map(
        lambda value: value.value if hasattr(value, "value") else str(value))
    bounds = {var: (lo, hi) for var, (lo, hi) in _PLAUSIBLE_RANGES.items()}
    lower = {var: bounds[var][0] for var in bounds}
    upper = {var: bounds[var][1] for var in bounds}

    required_values = ["forecast_value", "abs_error"]
    if require_observed:
        required_values.append("observed_value")
    for col in required_values:
        if col not in paired.columns:
            continue
        values = pd.to_numeric(paired[col], errors="coerce").to_numpy(dtype=float)
        bad |= ~np.isfinite(values)
        if col != "abs_error":
            lo = variables.map(lower).to_numpy(dtype=float)
            hi = variables.map(upper).to_numpy(dtype=float)
            bad |= ~np.isfinite(lo) | ~np.isfinite(hi) | (values < lo) | (values > hi)
    if "abs_error" in paired.columns:
        errors = pd.to_numeric(paired["abs_error"], errors="coerce").to_numpy(float)
        lo = variables.map(lower).to_numpy(dtype=float)
        hi = variables.map(upper).to_numpy(dtype=float)
        max_error = np.where(variables.isin(_CIRCULAR_VARIABLES), 180.0, hi - lo)
        bad |= errors < 0
        bad |= np.isfinite(errors) & (errors > max_error)

    # Concurrent forecast columns are optional (the archive stops some variables at a
    # shorter lead), hence NaN is allowed there.  Inf is never valid, and a finite value
    # outside the source variable's canonical range is corrupt just as surely.
    for col in (c for c in paired.columns if c.startswith("fc_")):
        source_var = col[3:]
        if source_var not in bounds:
            continue
        values = pd.to_numeric(paired[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        lo, hi = bounds[source_var]
        bad |= np.isinf(values) | (finite & ((values < lo) | (values > hi)))

    if not bad.any():
        return paired, paired.iloc[0:0].copy()
    return paired.loc[~bad], paired.loc[bad].copy()


def absolute_error(forecast, observed, variable):
    """Absolute error for linear variables and smallest angular error for directions."""
    if isinstance(forecast, pd.Series):
        out = (forecast - observed).abs()
        mask = (variable.isin(_CIRCULAR_VARIABLES) if isinstance(variable, pd.Series)
                else np.full(len(out), str(variable) in _CIRCULAR_VARIABLES))
        mask = np.asarray(mask)
        if mask.any():
            out.loc[mask] = circular_abs_error(forecast.loc[mask], observed.loc[mask])
        return out
    out = np.abs(np.asarray(forecast, dtype=float) - np.asarray(observed, dtype=float))
    variables = np.asarray(variable)
    if variables.ndim == 0:
        mask = np.full(out.shape, str(variables.item()) in _CIRCULAR_VARIABLES)
    else:
        mask = np.isin(variables.astype(str), tuple(_CIRCULAR_VARIABLES))
    if mask.any():
        out[mask] = circular_abs_error(forecast, observed)[mask]
    return out


def _circular_means(values: pd.Series, groups: pd.DataFrame) -> pd.Series:
    """Row-aligned circular means, grouped by ``groups``' columns."""
    out = pd.Series(np.nan, index=values.index, dtype=float)
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    positions = np.flatnonzero(np.isfinite(numeric))
    if positions.size == 0:
        return out
    keys = list(groups.columns)
    rows = groups.iloc[positions].reset_index(drop=True).copy()
    rows["_row"] = values.index.to_numpy()[positions]
    radians = np.radians(numeric[positions])
    rows["_sin"] = np.sin(radians)
    rows["_cos"] = np.cos(radians)
    stats = (rows.groupby(keys, sort=False, observed=True)
             .agg(_sin=("_sin", "mean"), _cos=("_cos", "mean")).reset_index())
    resultant = np.hypot(stats["_sin"].to_numpy(float), stats["_cos"].to_numpy(float))
    good = resultant > 1e-12
    stats["_mean"] = np.where(
        good, normalize_degrees(np.degrees(np.arctan2(stats["_sin"], stats["_cos"]))), np.nan)
    merged = rows[keys + ["_row"]].merge(stats[keys + ["_mean"]], on=keys, how="left")
    out.loc[merged["_row"].to_numpy()] = merged["_mean"].to_numpy(float)
    return out


def event_value_means(paired: pd.DataFrame) -> pd.DataFrame:
    """Event-grain forecast/observation means, circular for directional variables."""
    if "cycle_hour" in paired.columns:
        cycle = pd.to_numeric(paired["cycle_hour"], errors="coerce")
        if not pd.api.types.is_numeric_dtype(paired["cycle_hour"]) or cycle.isna().any():
            paired = paired.copy()
            paired["cycle_hour"] = cycle.fillna(0).astype("int16")
    else:
        # Legacy/synthetic frames predate cycle identity. Real cached/serving frames
        # already carry cycle_hour and take the zero-copy path above.
        paired = paired.copy()
        paired["cycle_hour"] = 0
    keys = EVENT_KEYS + ["variable"]
    out = (paired.groupby(keys, observed=True)
           .agg(fc_mean=("forecast_value", "mean"), obs=("observed_value", "mean"))
           .reset_index())
    out["fc_mean"] = out["fc_mean"].astype(float)
    out["obs"] = out["obs"].astype(float)
    for var in sorted(set(paired["variable"].astype(str)) & set(_CIRCULAR_VARIABLES)):
        sub = paired[paired["variable"].astype(str) == var]
        fc = _circular_means(sub["forecast_value"], sub[EVENT_KEYS])
        ob = (_circular_means(sub["observed_value"], sub[EVENT_KEYS])
              if "observed_value" in sub.columns else pd.Series(np.nan, index=sub.index))
        means = sub[EVENT_KEYS].copy()
        means["_fc_mean"] = fc.to_numpy(float)
        means["_obs"] = ob.to_numpy(float)
        means = means.groupby(EVENT_KEYS, sort=False, observed=True, as_index=False).mean()
        target = out["variable"].astype(str) == var
        detail = out.loc[target, EVENT_KEYS].merge(means, on=EVENT_KEYS, how="left",
                                                    validate="one_to_one")
        out.loc[target, "fc_mean"] = detail["_fc_mean"].to_numpy(float)
        out.loc[target, "obs"] = detail["_obs"].to_numpy(float)
    return out


def _season(month: pd.Series) -> pd.Series:
    return month.map(_SEASONS).astype("category")


def build_training_frame(
    canonical: pd.DataFrame,
    historical_bust_freq: dict | None = None,
    require_observed: bool = True,
    forecast_history: pd.DataFrame | None = None,
    jump_climatology: dict | None = None,
) -> pd.DataFrame:
    """`forecast_history` is cycles issued before those in `canonical` - raw forecast rows,
    or trajectories already reduced by forecast_trajectories (app.features.history). They
    feed the jumpiness features and produce no rows of their own: the chunked trainer and
    the one-cycle scorer each see only part of the archive at a time.

    `jump_climatology` is fitted on training rows only (compute_jump_climatology); without
    it `jump_rel_climatology` is NaN, exactly as the historical bust frequency is.
    """
    df, _quarantined = quarantine_invalid_canonical_values(canonical)
    df = df[df["region_id"].notna()]
    # Historical callers do not carry cycle metadata; treat those rows as the original
    # 00Z archive. New canonical rows always provide the field, including 06/12/18Z.
    if "cycle_hour" not in df.columns:
        df["cycle_hour"] = 0
    else:
        df["cycle_hour"] = pd.to_numeric(df["cycle_hour"], errors="coerce").fillna(0).astype("int16")
    fc = df[df["value_type"] == FORECAST].copy()
    trajectories = [forecast_trajectories(fc, validated=True)]
    if forecast_history is not None and not forecast_history.empty:
        trajectories.insert(0, forecast_trajectories(forecast_history))
    all_trajectories = pd.concat(trajectories, ignore_index=True)
    del trajectories
    jumps = compute_jumpiness(all_trajectories)
    laf = compute_time_lagged_ensemble(fc, all_trajectories, validated=True)
    del all_trajectories
    ob_cols = ["region_id", "valid_date", "variable", "value"]
    has_vs = "verification_status" in df.columns
    if has_vs:
        ob_cols.append("verification_status")
    ob = df[df["value_type"] == OBSERVED][ob_cols].copy()
    ob_raw = ob
    agg = {"value": "mean"}
    if has_vs:
        agg["verification_status"] = (
            lambda s: "provisional" if (s == "provisional").any() else "final"
        )
    ob = (ob.groupby(["region_id", "valid_date", "variable"], as_index=False)
            .agg(agg).rename(columns={"value": "observed_value"}))
    ob_keys = ["region_id", "valid_date", "variable"]
    for var in sorted(set(ob_raw["variable"].astype(str)) & set(_CIRCULAR_VARIABLES)):
        sub = ob_raw[ob_raw["variable"].astype(str) == var]
        circular = _circular_means(sub["value"], sub[ob_keys[:-1]])
        detail = sub[ob_keys].copy()
        detail["_circular_observed"] = circular.to_numpy(float)
        detail = detail.groupby(ob_keys, sort=False, observed=True,
                                as_index=False).mean()
        target = ob["variable"].astype(str) == var
        got = ob.loc[target, ob_keys].merge(detail, on=ob_keys, how="left",
                                             validate="one_to_one")
        ob.loc[target, "observed_value"] = got["_circular_observed"].to_numpy(float)

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

    paired["abs_error"] = absolute_error(
        paired["forecast_value"], paired["observed_value"], paired["variable"])
    required = ["abs_error", "lead_time_days"] if require_observed else ["lead_time_days"]
    paired = paired.dropna(subset=required)
    if paired.empty:
        return paired
    paired["lead_time_days"] = paired["lead_time_days"].astype(int)

    paired["valid_date"] = pd.to_datetime(paired["valid_date"])
    paired["init_date"] = pd.to_datetime(paired["init_date"])
    paired = _merge_jumpiness(paired, jumps)
    del jumps
    paired = attach_jump_climatology(paired, jump_climatology)
    paired = _merge_laf(paired, laf)
    del laf
    paired["month"] = paired["valid_date"].dt.month
    paired["season"] = _season(paired["month"])
    paired = attach_district_descriptors(paired)
    paired = attach_mjo_index(paired)
    paired["region_id"] = paired["region_id"].astype("category")

    grp = paired.groupby(EVENT_KEYS + ["variable"], observed=True)["forecast_value"]
    paired["ensemble_member_count"] = grp.transform("count")
    paired = _add_ensemble_spread(paired)

    paired = _add_rate_of_change(paired)
    paired = _add_concurrent_variable_forecasts(paired)

    if historical_bust_freq is not None:
        key = list(zip(paired["region_id"].astype(str), paired["season"].astype(str)))
        paired["historical_bust_frequency_region_season"] = [
            historical_bust_freq.get(k, np.nan) for k in key
        ]
    else:
        paired["historical_bust_frequency_region_season"] = np.nan

    return paired.reset_index(drop=True)


def _add_ensemble_spread(paired: pd.DataFrame) -> pd.DataFrame:
    """Linear standard deviation, or RMS angular deviation for wind direction."""
    keys = EVENT_KEYS + ["variable"]
    paired["ensemble_spread"] = (paired.groupby(keys, observed=True)["forecast_value"]
                                 .transform("std"))
    for var in sorted(set(paired["variable"].astype(str)) & set(_CIRCULAR_VARIABLES)):
        mask = paired["variable"].astype(str) == var
        sub = paired.loc[mask]
        if sub.empty:
            continue
        mean = _circular_means(sub["forecast_value"], sub[EVENT_KEYS])
        deviation = circular_abs_error(sub["forecast_value"], mean)
        work = sub[keys].copy()
        work["_sq"] = deviation ** 2
        sum_sq = work.groupby(keys, observed=True)["_sq"].transform("sum")
        count = work.groupby(keys, observed=True)["_sq"].transform("count")
        spread = pd.Series(np.sqrt(sum_sq / count), index=sub.index).where(count >= 2)
        paired.loc[mask, "ensemble_spread"] = spread
    return paired


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
    for var in sorted(set(paired["variable"].astype(str)) & set(_CIRCULAR_VARIABLES)):
        sub = paired[paired["variable"].astype(str) == var]
        detail = sub[MEMBER_KEYS].copy()
        detail["_circular_mean"] = _circular_means(
            sub["forecast_value"], sub[MEMBER_KEYS]).to_numpy(float)
        detail = detail.groupby(MEMBER_KEYS, sort=False, observed=True,
                                as_index=False).mean()
        key = wide[MEMBER_KEYS].merge(detail, on=MEMBER_KEYS, how="left")
        wide[f"fc_{var}"] = key["_circular_mean"].to_numpy(float)
    merged = paired.merge(wide, on=MEMBER_KEYS, how="left")
    del wide
    for var in paired["variable"].unique():
        col = f"fc_{var}"
        if col in merged.columns:
            merged.loc[merged["variable"] == var, col] = np.nan
    return merged


def forecast_trajectories(fc: pd.DataFrame, *, validated: bool = False) -> pd.DataFrame:
    """The ensemble-mean forecast per (region, variable, valid_date, init_date).

    The ensemble mean, not a member: perturbed member p01 of one cycle is not the same
    trajectory as p01 of the next, so a member-by-member change would be noise between
    unrelated perturbations. Wind direction is averaged as a unit vector. ``validated``
    is an internal fast path for build_training_frame after its boundary quarantine.
    """
    out_cols = TRAJECTORY_COLUMNS
    if fc.empty:
        return pd.DataFrame(columns=out_cols)
    if "value" not in fc.columns and "fc_mean" in fc.columns:
        # Already reduced by forecast_history.  Still reject an impossible cached mean.
        reduced = fc.copy()
        if "cycle_hour" not in reduced.columns:
            reduced["cycle_hour"] = 0
        else:
            reduced["cycle_hour"] = pd.to_numeric(
                reduced["cycle_hour"], errors="coerce").fillna(0).astype("int16")
        out = reduced[out_cols].copy()
        values = pd.to_numeric(out["fc_mean"], errors="coerce").to_numpy(float)
        limits = out["variable"].astype(str).map(
            {v: bounds for v, bounds in _PLAUSIBLE_RANGES.items()})
        lo = limits.map(lambda x: x[0] if isinstance(x, tuple) else np.nan).to_numpy(float)
        hi = limits.map(lambda x: x[1] if isinstance(x, tuple) else np.nan).to_numpy(float)
        valid = np.isfinite(values) & (values >= lo) & (values <= hi)
        return out.loc[valid].reset_index(drop=True)
    if not validated:
        fc, _quarantined = quarantine_invalid_canonical_values(fc)
    if "cycle_hour" not in fc.columns:
        cycle_hour = pd.Series(0, index=fc.index, dtype="int16")
    else:
        cycle_hour = pd.to_numeric(fc["cycle_hour"], errors="coerce").fillna(0).astype("int16")
    t = pd.DataFrame({
        "region_id": fc["region_id"].to_numpy(),
        "cycle_hour": cycle_hour.to_numpy(),
        "variable": fc["variable"].to_numpy(),
        "valid_date": pd.to_datetime(fc["valid_date"]).to_numpy(),
        "init_date": pd.to_datetime(fc["init_date"]).to_numpy(),
        "value": pd.to_numeric(fc["value"], errors="coerce").to_numpy(dtype=float),
    })
    t = t[t["value"].notna() & t["init_date"].notna() & t["region_id"].notna()]
    t = t[~((t["variable"] == "soil_moisture_pct") & (t["value"] >= _SOIL_SATURATED))]
    if t.empty:
        return pd.DataFrame(columns=out_cols)

    circ = t["variable"].isin(_CIRCULAR_VARIABLES).to_numpy()
    rad = np.radians(t["value"].to_numpy())
    t["_sin"] = np.where(circ, np.sin(rad), np.nan)
    t["_cos"] = np.where(circ, np.cos(rad), np.nan)
    g = (t.groupby(_TRAJECTORY_KEYS, sort=False, observed=True)
          .agg(fc_mean=("value", "mean"), _sin=("_sin", "mean"), _cos=("_cos", "mean"))
          .reset_index())
    del t
    circ_g = g["variable"].isin(_CIRCULAR_VARIABLES).to_numpy()
    if circ_g.any():
        sin_mean = g.loc[circ_g, "_sin"].to_numpy(float)
        cos_mean = g.loc[circ_g, "_cos"].to_numpy(float)
        resultant = np.hypot(sin_mean, cos_mean)
        circular_mean = np.full(resultant.shape, np.nan, dtype=float)
        defined = resultant > 1e-12
        circular_mean[defined] = (
            normalize_degrees(np.degrees(np.arctan2(sin_mean[defined], cos_mean[defined]))))
        g.loc[circ_g, "fc_mean"] = circular_mean
    g["region_id"] = g["region_id"].astype(str)
    g["variable"] = g["variable"].astype(str)
    return g[out_cols]


def compute_jumpiness(trajectories: pd.DataFrame, window: int = JUMP_WINDOW) -> pd.DataFrame:
    """jump_abs_change, jump_std and jump_sign_flips per trajectory row.

    For each (region, variable, valid_date), cycles are ordered by init_date and each row
    looks back only at cycles issued before it:

      jump_abs_change  |this cycle - the previous cycle covering the same valid date|
      jump_std         sample std (ddof=1) of the last `window` cycles, this one included;
                       needs at least 3 - with 2 it is only |change| / sqrt(2) again
      jump_sign_flips  how often the direction of change reversed within that window;
                       an unchanged step carries no direction and is skipped over, so
                       up / flat / down still counts as one reversal. Needs 3 cycles.

    Too few cycles is NaN, never 0: "no earlier forecast" is not "it did not move".
    Wind direction differences are wrapped to (-180, 180] before anything is computed.
    """
    feats = ["jump_abs_change", "jump_std", "jump_sign_flips"]
    if trajectories.empty:
        return pd.DataFrame(columns=_TRAJECTORY_KEYS + feats)
    if "cycle_hour" not in trajectories.columns:
        trajectories = trajectories.copy()
        trajectories["cycle_hour"] = 0
    else:
        trajectories = trajectories.copy()
        trajectories["cycle_hour"] = pd.to_numeric(
            trajectories["cycle_hour"], errors="coerce").fillna(0).astype("int16")
    k = max(int(window), 3)
    out = trajectories.sort_values(_TRAJECTORY_KEYS, ignore_index=True)
    grp = out.groupby(["region_id", "variable", "valid_date"], sort=False,
                      observed=True)["fc_mean"]
    # column j is the value j cycles back; NaN once a trajectory has no earlier cycle
    vals = np.column_stack([grp.shift(j).to_numpy(dtype=float) for j in range(k)])
    diffs = vals[:, :-1] - vals[:, 1:]          # column j: change into cycle t-j
    del vals
    circ = out["variable"].isin(_CIRCULAR_VARIABLES).to_numpy()
    if circ.any():
        diffs[circ] = (diffs[circ] + 180.0) % 360.0 - 180.0

    # Values relative to this cycle, rebuilt from the (wrapped) changes, so the spread of
    # 350 -> 10 -> 350 is taken over 350, 370, 350. NaNs only ever trail, so they stay put.
    rel = np.column_stack([np.zeros(len(out)), -np.cumsum(diffs, axis=1)])
    n = np.sum(~np.isnan(rel), axis=1)
    mean = np.nansum(rel, axis=1) / n
    ss = np.nansum((rel - mean[:, None]) ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        std = np.where(n >= 3, np.sqrt(ss / (n - 1)), np.nan)

    sign = np.sign(diffs)
    flips = np.zeros(len(out))
    last = np.zeros(len(out))
    for j in range(sign.shape[1]):
        s = sign[:, j]
        moved = ~np.isnan(s) & (s != 0)
        flips += moved & (last != 0) & (s != last)
        last = np.where(moved, s, last)

    out["jump_abs_change"] = np.abs(diffs[:, 0])
    out["jump_std"] = std
    out["jump_sign_flips"] = np.where(n >= 3, flips, np.nan)
    return out[_TRAJECTORY_KEYS + feats]


def _merge_jumpiness(paired: pd.DataFrame, jumps: pd.DataFrame) -> pd.DataFrame:
    feats = ["jump_abs_change", "jump_std", "jump_sign_flips"]
    if jumps.empty:
        for col in feats:
            paired[col] = np.nan
        return paired
    return paired.merge(jumps, on=_TRAJECTORY_KEYS, how="left")


def compute_time_lagged_ensemble(fc: pd.DataFrame, trajectories: pd.DataFrame,
                                 window: int = LAF_WINDOW, *,
                                 validated: bool = False) -> pd.DataFrame:
    """laf_pool_mean, laf_pool_std, laf_pool_size and laf_spread_ratio per trajectory row
    (C2, time-lagged ensemble / lagged-average forecasting).

    A GEFS reforecast cycle carries only 5 of the operational feed's 31 members
    (CLAUDE.md known limitations). This widens it cheaply: this cycle's own real members
    are pooled with up to `window - 1` EARLIER cycles that are also valid for the same
    date, each contributing its ensemble MEAN as one extra pooled value - not its
    individual members, which forecast_trajectories/forecast_history already discard for
    memory. Causal: only cycles issued on or before this row's own init_date are pooled.

    Linear variables retain the exact grouped pooled-variance identity. For wind
    direction, both the mean and every squared deviation are circular: the mean is a
    resultant unit vector and a deviation is the smallest angular separation in
    ``[0, 180]``. Treating 350 and 10 as the linear numbers 350 and 10 would manufacture
    a 170-degree spread where the physical disagreement is 20 degrees.

    No earlier cycle makes the pool identical to this cycle's members, so the spread
    ratio is exactly 1. Otherwise zero own spread with non-zero pooled spread is
    undefined (NaN), never infinite. Too few members has no spread and is also NaN.
    """
    feats = ["laf_pool_mean", "laf_pool_std", "laf_pool_size"]
    all_feats = feats + ["laf_spread_ratio"]
    if trajectories.empty:
        return pd.DataFrame(columns=_TRAJECTORY_KEYS + all_feats)
    if "cycle_hour" not in trajectories.columns:
        trajectories = trajectories.copy()
        trajectories["cycle_hour"] = 0
    else:
        trajectories = trajectories.copy()
        trajectories["cycle_hour"] = pd.to_numeric(
            trajectories["cycle_hour"], errors="coerce").fillna(0).astype("int16")

    own = fc.copy()
    if not validated:
        own, _quarantined = quarantine_invalid_canonical_values(own)
    if "cycle_hour" not in own.columns:
        own["cycle_hour"] = 0
    else:
        own["cycle_hour"] = pd.to_numeric(own["cycle_hour"], errors="coerce").fillna(0).astype("int16")
    own["value"] = pd.to_numeric(own["value"], errors="coerce")
    # Match forecast_trajectories' own explicit conversion - own_stats' merge key must be
    # the same dtype as trajectories', or the merge below raises rather than silently
    # coercing (real failure: fc's raw valid_date/init_date arrived as object dtype).
    own["valid_date"] = pd.to_datetime(own["valid_date"])
    own["init_date"] = pd.to_datetime(own["init_date"])
    own = own[own["value"].notna() & own["init_date"].notna() & own["region_id"].notna()]
    own = own[~((own["variable"] == "soil_moisture_pct") & (own["value"] >= _SOIL_SATURATED))]

    circular_own = own["variable"].isin(_CIRCULAR_VARIABLES)
    stats = []
    linear = own[~circular_own]
    if not linear.empty:
        stats.append((linear.groupby(_TRAJECTORY_KEYS, sort=False, observed=True)["value"]
                      .agg(mean0="mean", var0="var", n0="count").reset_index()))

    circular_values = own[circular_own].copy()
    if not circular_values.empty:
        keys = _TRAJECTORY_KEYS
        radians = np.radians(circular_values["value"].to_numpy(float))
        circular_values["_sin"] = np.sin(radians)
        circular_values["_cos"] = np.cos(radians)
        vector_mean = (circular_values.groupby(keys, sort=False, observed=True)
                       .agg(_sin=("_sin", "mean"), _cos=("_cos", "mean"),
                            n0=("value", "count")).reset_index())
        resultant = np.hypot(vector_mean["_sin"].to_numpy(float),
                              vector_mean["_cos"].to_numpy(float))
        mean0 = np.full(len(vector_mean), np.nan, dtype=float)
        defined = resultant > 1e-12
        mean0[defined] = normalize_degrees(np.degrees(np.arctan2(
            vector_mean.loc[defined, "_sin"], vector_mean.loc[defined, "_cos"])))
        vector_mean["mean0"] = mean0
        row_values = circular_values[keys + ["value"]].copy()
        row_values = row_values.merge(vector_mean[keys + ["mean0"]], on=keys, how="left")
        row_values["_sq"] = circular_abs_error(
            row_values["value"], row_values["mean0"]) ** 2
        own_spread = (row_values.groupby(keys, sort=False, observed=True)["_sq"]
                      .agg(_own_ss="sum", _n="count").reset_index())
        circular_stats = vector_mean.merge(own_spread, on=keys, how="left")
        circular_stats["var0"] = circular_stats["_own_ss"] / circular_stats["_n"]
        stats.append(circular_stats[keys + ["mean0", "var0", "n0"]])

    own_stats = (pd.concat(stats, ignore_index=True) if stats else
                 pd.DataFrame(columns=_TRAJECTORY_KEYS + ["mean0", "var0", "n0"]))

    out = trajectories.sort_values(_TRAJECTORY_KEYS, ignore_index=True)
    out = out.merge(own_stats, on=_TRAJECTORY_KEYS, how="left")
    circular_out = out["variable"].isin(_CIRCULAR_VARIABLES).to_numpy()

    k = max(int(window), 1)
    grp = out.groupby(["region_id", "variable", "valid_date"], sort=False,
                      observed=True)["fc_mean"]
    # column j (1-indexed) is the ensemble mean j cycles back; NaN once a trajectory has
    # no earlier cycle - never treated as zero.
    prior = (np.column_stack([grp.shift(j).to_numpy(dtype=float) for j in range(1, k)])
            if k > 1 else np.empty((len(out), 0)))

    if (~circular_out).any():
        linear_out = out.loc[~circular_out]
        n0 = linear_out["n0"].to_numpy(dtype=float)
        mean0 = linear_out["mean0"].to_numpy(dtype=float)
        var0 = linear_out["var0"].to_numpy(dtype=float)
        linear_prior = prior[~circular_out]
        n_prior = (np.sum(np.isfinite(linear_prior), axis=1) if linear_prior.size
                   else np.zeros(len(linear_out)))
        sum_prior = (np.nansum(linear_prior, axis=1) if linear_prior.size
                     else np.zeros(len(linear_out)))
        pool_size = n0 + n_prior

        with np.errstate(invalid="ignore", divide="ignore"):
            pool_mean = (n0 * mean0 + sum_prior) / pool_size
            within = np.where(n0 >= 2, (n0 - 1) * var0, 0.0)
            own_between = n0 * (mean0 - pool_mean) ** 2
            prior_between = (np.nansum((linear_prior - pool_mean[:, None]) ** 2, axis=1)
                             if linear_prior.size else np.zeros(len(linear_out)))
            ss = within + own_between + prior_between
            pool_var = np.where(pool_size >= 2, ss / (pool_size - 1), np.nan)
            pool_std = np.sqrt(pool_var)
            own_std = np.where(n0 >= 2, np.sqrt(var0), np.nan)
            ratio = np.where(n_prior == 0, 1.0, pool_std / own_std)
            ratio = np.where(n0 < 2, np.nan, ratio)
            ratio = np.where((own_std == 0) & (n_prior > 0), np.nan, ratio)
        out.loc[~circular_out, "laf_pool_mean"] = pool_mean
        out.loc[~circular_out, "laf_pool_std"] = pool_std
        out.loc[~circular_out, "laf_pool_size"] = pool_size
        out.loc[~circular_out, "laf_spread_ratio"] = ratio
    if circular_out.any():
        circular_frame = out.loc[circular_out].copy()
        circular_prior = prior[circular_out]
        n0 = circular_frame["n0"].to_numpy(dtype=float)
        mean0 = circular_frame["mean0"].to_numpy(dtype=float)
        var0 = circular_frame["var0"].to_numpy(dtype=float)
        n_prior = (np.sum(np.isfinite(circular_prior), axis=1) if circular_prior.size
                   else np.zeros(len(circular_frame)))
        pool_size = n0 + n_prior

        with np.errstate(invalid="ignore", divide="ignore"):
            prior_sin = (np.nansum(np.sin(np.radians(circular_prior)), axis=1)
                         if circular_prior.size else np.zeros(len(circular_frame)))
            prior_cos = (np.nansum(np.cos(np.radians(circular_prior)), axis=1)
                         if circular_prior.size else np.zeros(len(circular_frame)))
            sin_sum = n0 * np.sin(np.radians(mean0)) + prior_sin
            cos_sum = n0 * np.cos(np.radians(mean0)) + prior_cos
            resultant_fraction = np.hypot(sin_sum, cos_sum) / pool_size
            pool_mean = np.full(len(circular_frame), np.nan, dtype=float)
            defined = np.isfinite(resultant_fraction) & (resultant_fraction > 1e-12)
            pool_mean[defined] = normalize_degrees(np.degrees(np.arctan2(
                sin_sum[defined], cos_sum[defined])))

            values_with_pool = circular_values[_TRAJECTORY_KEYS + ["value"]].merge(
                circular_frame[_TRAJECTORY_KEYS].assign(laf_pool_mean=pool_mean),
                on=_TRAJECTORY_KEYS, how="left", validate="many_to_one")
            values_with_pool["_sq"] = circular_abs_error(
                values_with_pool["value"], values_with_pool["laf_pool_mean"]) ** 2
            own_ss_by_key = (values_with_pool.groupby(_TRAJECTORY_KEYS, observed=True)["_sq"]
                             .sum())
            key_index = pd.MultiIndex.from_frame(circular_frame[_TRAJECTORY_KEYS])
            own_ss = own_ss_by_key.reindex(key_index).to_numpy(float)
            prior_ss = (np.nansum(circular_abs_error(
                circular_prior, pool_mean[:, None]) ** 2, axis=1)
                if circular_prior.size else np.zeros(len(circular_frame)))
            ss = own_ss + prior_ss
            # Circular spread is RMS angular deviation, matching ensemble_spread and
            # remaining bounded by 180 degrees (an n-1 denominator can exceed that for
            # two opposing members).
            pool_var = np.where(pool_size >= 2, ss / pool_size, np.nan)
            pool_std = np.sqrt(pool_var)
            own_std = np.where(n0 >= 2, np.sqrt(var0), np.nan)
            ratio = np.where(n_prior == 0, 1.0, pool_std / own_std)
            ratio = np.where(n0 < 2, np.nan, ratio)
            ratio = np.where((own_std == 0) & (n_prior > 0), np.nan, ratio)

        circular_frame["laf_pool_mean"] = pool_mean
        circular_frame["laf_pool_std"] = pool_std
        circular_frame["laf_pool_size"] = pool_size
        circular_frame["laf_spread_ratio"] = ratio
        out.loc[circular_out, all_feats] = circular_frame[all_feats].to_numpy()

    return out[_TRAJECTORY_KEYS + all_feats]


def _merge_laf(paired: pd.DataFrame, laf: pd.DataFrame) -> pd.DataFrame:
    feats = ["laf_pool_mean", "laf_pool_std", "laf_pool_size", "laf_spread_ratio"]
    if laf.empty:
        for col in feats:
            paired[col] = np.nan
        return paired
    return paired.merge(laf, on=_TRAJECTORY_KEYS, how="left")


def compute_jump_climatology(frame: pd.DataFrame) -> dict:
    """(region_id, variable) -> mean jump_abs_change. Fit on TRAINING rows only.

    Member-grain rows repeat one trajectory value per member; every member row carries the
    same value, so the mean is unchanged by that repetition when member counts agree.
    """
    ok = frame["jump_abs_change"].notna()
    if not ok.any():
        return {}
    sub = frame.loc[ok, ["region_id", "variable", "jump_abs_change"]]
    m = sub.groupby(["region_id", "variable"], observed=True)["jump_abs_change"].mean()
    return {(str(r), str(v)): float(x) for (r, v), x in m.items()}


def attach_jump_climatology(frame: pd.DataFrame, climatology: dict | None) -> pd.DataFrame:
    """jump_rel_climatology = jump_abs_change / that district-variable's climatology.

    NaN where the climatology is unknown or zero: a district whose forecast never moved in
    training has no scale to be relative to, and inventing one would be a fabricated value.
    A merge on a thin key frame, not a per-row Python lookup - see attach_hbf_column.
    """
    col = "jump_rel_climatology"
    if not climatology or frame.empty or "jump_abs_change" not in frame.columns:
        frame[col] = np.nan
        return frame
    ref = pd.DataFrame([(r, v, x) for (r, v), x in climatology.items()],
                       columns=["region_id", "variable", "_clim"])
    key = pd.DataFrame({"region_id": frame["region_id"].astype(str).to_numpy(),
                        "variable": frame["variable"].astype(str).to_numpy()})
    clim = key.merge(ref, on=["region_id", "variable"], how="left")["_clim"].to_numpy(float)
    del key
    clim = np.where(clim > 0, clim, np.nan)
    frame[col] = pd.to_numeric(frame["jump_abs_change"], errors="coerce").to_numpy(float) / clim
    return frame


def attach_district_descriptors(frame: pd.DataFrame) -> pd.DataFrame:
    """Merge the static per-district descriptors (C4) onto `frame` by region_id.

    A left merge, not a lookup dict: the descriptor table is ~666 rows regardless of how
    many rows `frame` has, so this is a thin-table merge like attach_jump_climatology, not
    a per-row Python call. A region_id absent from the descriptor table (should not
    happen - the descriptors are built from the same registry every region_id comes from)
    gets NaN, never a fabricated value.
    """
    desc = idist.load_district_descriptors()
    key = pd.DataFrame({"region_id": frame["region_id"].astype(str).to_numpy()})
    merged = key.merge(desc, on="region_id", how="left")
    for col in DISTRICT_DESCRIPTOR_FEATURES:
        frame[col] = merged[col].to_numpy()
    return frame


@functools.lru_cache(maxsize=1)
def load_mjo_index() -> pd.DataFrame:
    """(date, mjo_rmm1, mjo_rmm2, mjo_amplitude) - one row per calendar day, built once
    by scripts/fetch_mjo_index.py. Not regenerated here.

    A missing file degrades to an empty index - MJO becomes NaN everywhere, not a hard
    failure - the same discipline `_merge_jumpiness`/`_merge_laf` already apply to a
    missing climatology: "the code path is complete; the data is not there" is a real,
    named state (docs/known-issues.md), not something training should crash over.
    """
    path = resolve_path(settings.data_dir) / "mjo_omi_index.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["date", *MJO_FEATURES])
    df = pd.read_parquet(path, columns=["date", *MJO_FEATURES])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date", ignore_index=True)


def attach_mjo_index(frame: pd.DataFrame, mjo: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge the global daily MJO index (C3) onto `frame` by an as-of join on init_date.

    Backward-only (`direction="backward"`): the MJO reading for a day after a forecast
    was issued could not have been known at issue time, so only readings on or before
    init_date are ever eligible - the same causality discipline as C1's jumpiness and
    C2's time-lagged ensemble. `tolerance` refuses a match older than
    MJO_ASOF_TOLERANCE_DAYS: a forecast issued long after the last known reading gets an
    unknown MJO state, not a stale one presented as current.
    """
    if mjo is None:
        mjo = load_mjo_index()
    if mjo.empty:
        for col in MJO_FEATURES:
            frame[col] = np.nan
        return frame
    # A member-grain frame can be millions of rows over a handful of distinct cycles;
    # as-of join only the unique init_dates, then map back with a plain equality merge -
    # the same "thin-table merge, not a per-row lookup" discipline as
    # attach_district_descriptors.
    dates = pd.DataFrame({
        "init_date": pd.to_datetime(frame["init_date"]).drop_duplicates().sort_values(),
    })
    asof = pd.merge_asof(
        dates, mjo.sort_values("date"), left_on="init_date", right_on="date",
        direction="backward", tolerance=pd.Timedelta(days=MJO_ASOF_TOLERANCE_DAYS),
    )[["init_date", *MJO_FEATURES]]
    key = pd.DataFrame({"init_date": pd.to_datetime(frame["init_date"]).to_numpy()})
    merged = key.merge(asof, on="init_date", how="left")
    for col in MJO_FEATURES:
        frame[col] = merged[col].to_numpy()
    return frame


def compute_historical_bust_frequency(
    paired_train: pd.DataFrame, large_error_pct: float = 75.0
) -> dict:
    thresholds = {}
    for var, g in paired_train.groupby("variable", observed=True):
        vals = pd.to_numeric(g["abs_error"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            thresholds[var] = float(np.percentile(vals, large_error_pct))
    thr = thresholds
    p = paired_train.copy()
    p["is_large"] = [
        row.abs_error > thr.get(row.variable, np.inf) for row in p.itertuples()
    ]
    rate = p.groupby([p["region_id"].astype(str), p["season"].astype(str)])["is_large"].mean()
    return {tuple(k): float(v) for k, v in rate.items()}
