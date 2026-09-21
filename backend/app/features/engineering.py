from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from app.config import settings
from app.db.base import resolve_path
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

EVENT_KEYS = ["region_id", "init_date", "valid_date", "lead_time_days"]
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
# Degrees: 350 -> 10 is a 20 degree change, not 340.
_CIRCULAR_VARIABLES = {"wind_direction_deg"}
_TRAJECTORY_KEYS = ["region_id", "variable", "valid_date", "init_date"]
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
    df = canonical.copy()
    df = df[df["region_id"].notna()]
    fc = df[df["value_type"] == FORECAST].copy()
    trajectories = [forecast_trajectories(fc)]
    if forecast_history is not None and not forecast_history.empty:
        trajectories.insert(0, forecast_trajectories(forecast_history))
    all_trajectories = pd.concat(trajectories, ignore_index=True)
    del trajectories
    jumps = compute_jumpiness(all_trajectories)
    laf = compute_time_lagged_ensemble(fc, all_trajectories)
    del all_trajectories
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


def forecast_trajectories(fc: pd.DataFrame) -> pd.DataFrame:
    """The ensemble-mean forecast per (region, variable, valid_date, init_date).

    The ensemble mean, not a member: perturbed member p01 of one cycle is not the same
    trajectory as p01 of the next, so a member-by-member change would be noise between
    unrelated perturbations. Wind direction is averaged as a unit vector.
    """
    out_cols = TRAJECTORY_COLUMNS
    if fc.empty:
        return pd.DataFrame(columns=out_cols)
    if "value" not in fc.columns and "fc_mean" in fc.columns:
        return fc[out_cols]                      # already reduced
    t = pd.DataFrame({
        "region_id": fc["region_id"].to_numpy(),
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
        g.loc[circ_g, "fc_mean"] = np.degrees(
            np.arctan2(g.loc[circ_g, "_sin"], g.loc[circ_g, "_cos"])) % 360.0
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
                                 window: int = LAF_WINDOW) -> pd.DataFrame:
    """laf_pool_mean, laf_pool_std, laf_pool_size and laf_spread_ratio per trajectory row
    (C2, time-lagged ensemble / lagged-average forecasting).

    A GEFS reforecast cycle carries only 5 of the operational feed's 31 members
    (CLAUDE.md known limitations). This widens it cheaply: this cycle's own real members
    are pooled with up to `window - 1` EARLIER cycles that are also valid for the same
    date, each contributing its ensemble MEAN as one extra pooled value - not its
    individual members, which forecast_trajectories/forecast_history already discard for
    memory. Causal: only cycles issued on or before this row's own init_date are pooled.

    This cycle's own contribution (mean0, var0, n0) is a plain, non-circular mean/variance
    over its raw member values - the same simplification `ensemble_spread` (below) already
    makes; a genuinely circular pooled variance is not implemented. Earlier cycles enter
    only through their (circular-aware) trajectory mean, one pseudo-member each, with no
    within-cycle variance of their own to add - forecast_trajectories never carried it.
    Combining them uses the exact pooled-variance identity for grouped samples, not an
    approximation:

        SS = (n0-1)*var0 + n0*(mean0-pool_mean)^2 + sum_j (mean_j-pool_mean)^2
        pool_var = SS / (n0 + n_prior - 1)

    n0=1 correctly contributes zero within-cycle variance ((n0-1)=0), not a fabricated
    value. No earlier cycle at all (n_prior=0) makes the pool identical to this cycle's own
    members, so laf_spread_ratio is exactly 1 by construction - guarded explicitly so a
    cycle whose members happen to agree exactly (own_std=0, real for a dry-day rainfall
    forecast) never turns into a 0/0 division.

    Too few cycles/members for a variance is NaN, never 0 - the same discipline as C1.
    """
    feats = ["laf_pool_mean", "laf_pool_std", "laf_pool_size"]
    all_feats = feats + ["laf_spread_ratio"]
    if trajectories.empty:
        return pd.DataFrame(columns=_TRAJECTORY_KEYS + all_feats)

    own = fc.copy()
    own["value"] = pd.to_numeric(own["value"], errors="coerce")
    # Match forecast_trajectories' own explicit conversion - own_stats' merge key must be
    # the same dtype as trajectories', or the merge below raises rather than silently
    # coercing (real failure: fc's raw valid_date/init_date arrived as object dtype).
    own["valid_date"] = pd.to_datetime(own["valid_date"])
    own["init_date"] = pd.to_datetime(own["init_date"])
    own = own[own["value"].notna() & own["init_date"].notna() & own["region_id"].notna()]
    own = own[~((own["variable"] == "soil_moisture_pct") & (own["value"] >= _SOIL_SATURATED))]
    g = own.groupby(_TRAJECTORY_KEYS, sort=False, observed=True)["value"]
    # "var" (pandas' built-in, ddof=1 by default - identical to var(ddof=1)) uses the
    # cythonised groupby path; a Python lambda here does not; measured real difference at
    # full-year training volume, not visible in any small test. See CLAUDE.md: this repo
    # fails on volume, not on a green suite.
    own_stats = g.agg(mean0="mean", var0="var", n0="count").reset_index()
    del own

    out = trajectories.sort_values(_TRAJECTORY_KEYS, ignore_index=True)
    out = out.merge(own_stats, on=_TRAJECTORY_KEYS, how="left")
    del own_stats

    k = max(int(window), 1)
    grp = out.groupby(["region_id", "variable", "valid_date"], sort=False,
                      observed=True)["fc_mean"]
    # column j (1-indexed) is the ensemble mean j cycles back; NaN once a trajectory has
    # no earlier cycle - never treated as zero.
    prior = (np.column_stack([grp.shift(j).to_numpy(dtype=float) for j in range(1, k)])
            if k > 1 else np.empty((len(out), 0)))

    n0 = out["n0"].to_numpy(dtype=float)
    mean0 = out["mean0"].to_numpy(dtype=float)
    var0 = out["var0"].to_numpy(dtype=float)
    n_prior = np.sum(~np.isnan(prior), axis=1) if prior.size else np.zeros(len(out))
    sum_prior = np.nansum(prior, axis=1) if prior.size else np.zeros(len(out))
    pool_size = n0 + n_prior

    with np.errstate(invalid="ignore", divide="ignore"):
        pool_mean = (n0 * mean0 + sum_prior) / pool_size
        within = np.where(n0 >= 2, (n0 - 1) * var0, 0.0)
        own_between = n0 * (mean0 - pool_mean) ** 2
        prior_between = (np.nansum((prior - pool_mean[:, None]) ** 2, axis=1)
                        if prior.size else np.zeros(len(out)))
        ss = within + own_between + prior_between
        pool_var = np.where(pool_size >= 2, ss / (pool_size - 1), np.nan)
        pool_std = np.sqrt(pool_var)

        own_std = np.where(n0 >= 2, np.sqrt(var0), np.nan)
        ratio = np.where(n_prior == 0, 1.0, pool_std / own_std)
        ratio = np.where(n0 < 2, np.nan, ratio)
        # Members that agree exactly (own_std = 0) against a pool that does not: the
        # ratio is undefined, not infinite. Real 2026-09-21: humidity_pct at saturation,
        # 630-2,205 inf rows in every year 2000-2016, which XGBoost refuses to train on.
        ratio = np.where((own_std == 0) & (n_prior > 0), np.nan, ratio)

    out["laf_pool_mean"] = pool_mean
    out["laf_pool_std"] = pool_std
    out["laf_pool_size"] = pool_size
    out["laf_spread_ratio"] = ratio
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
