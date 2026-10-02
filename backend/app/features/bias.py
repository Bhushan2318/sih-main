"""Per-district forecast bias, fitted on training cycles only (label version 2).

A bust should mean the forecast failed, not that it is always off by the same amount. A
fixed per-district/variable/lead error is 64% of squared error for temperature and humidity
and 90% for soil moisture (docs/known-issues.md, 2026-09-25), and GEFS 2 m humidity runs
~16 %RH drier than ERA5 at Day 1 (Nov 2017). Ordinary bias correction removes that.

The table holds the mean signed error of the ensemble mean, forecast minus observation, for
each (district, variable, lead day, season), over training cycles only. `apply_bias`
subtracts it from every member forecast before anything compares a forecast with its
observation, so the regressor target, the event label and the thresholds all describe the
corrected error, and the corrected forecast is what the model reads. The same function runs
in training, validation, test, live scoring and re-scoring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app import contracts

BIAS_KEYS = ["region_id", "variable", "lead_time_days", "season"]

# A cell needs this many training events for its mean to be a bias rather than noise. At
# 2000-2015 daily cycles a (district, variable, lead, season) cell holds roughly 970 (Oct-Nov)
# to 1,950 (Jun-Sep) events, so only genuinely thin cells fall below it; those back off to a
# coarser level (LEVELS). A key thin at every level has no bias - the forecast then has no
# corrected value, which is missing, never zero.
# 30 is the usual floor for estimating a mean (its standard error is then under a fifth
# of the error's own spread). It matters only for thin keys: the CI sample's 12 training
# cycles give soil moisture (Days 1-3) 36 events per district and wind speed (Days 1-5)
# 60, which a floor of 100 dropped from the label altogether.
MIN_BIAS_EVENTS = 30


def _keys_as_str(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for k in BIAS_KEYS:
        col = df[k]
        out[k] = col.astype(int) if k == "lead_time_days" else col.astype(str)
    return out


def accumulate_bias(acc: "pd.DataFrame | None", events: pd.DataFrame) -> pd.DataFrame:
    """Add one chunk of events (`BIAS_KEYS` + `fc_mean`, `obs`) to running sums.

    Only label variables are counted. Returns a small table (one row per cell), so a
    seventeen-year fit never holds more than one year of events at a time.
    """
    ev = events[label_variable_mask(events["variable"])]
    err = ev["fc_mean"].to_numpy(dtype=float) - ev["obs"].to_numpy(dtype=float)
    ok = np.isfinite(err)
    if not ok.any():
        return acc
    part = _keys_as_str(ev.loc[ok]).assign(s=err[ok], n=1)
    part = part.groupby(BIAS_KEYS, sort=False)[["s", "n"]].sum().reset_index()
    if acc is None or acc.empty:
        return part
    return (pd.concat([acc, part], ignore_index=True)
            .groupby(BIAS_KEYS, sort=False)[["s", "n"]].sum().reset_index())


# Back-off levels, finest first. A cell too thin at one level takes its district's bias at
# the next: across leads for that season, then across the whole year. All three are means
# over training events of the same district and variable, so a coarser level is a smoother
# estimate of the same offset, not a different quantity. A key that is too thin even at the
# coarsest level has no bias at all.
LEVELS = ("lead_season", "season", "all")
_ANY_LEAD = -1
_ANY_SEASON = "*"


def finish_bias_table(acc: "pd.DataFrame | None") -> pd.DataFrame:
    """Running sums -> one row per (level, key) that has at least MIN_BIAS_EVENTS events:
    `BIAS_KEYS` + `bias`, `n`, `level`. Coarser levels carry lead -1 and/or season "*"."""
    cols = BIAS_KEYS + ["bias", "n", "level"]
    if acc is None or acc.empty:
        return pd.DataFrame(columns=cols)
    a = acc.copy()
    a["n"] = a["n"].astype(int)
    by_season = (a.groupby(["region_id", "variable", "season"], sort=False)[["s", "n"]]
                 .sum().reset_index().assign(lead_time_days=_ANY_LEAD))
    by_all = (a.groupby(["region_id", "variable"], sort=False)[["s", "n"]].sum()
              .reset_index().assign(lead_time_days=_ANY_LEAD, season=_ANY_SEASON))
    parts = []
    for level, t in zip(LEVELS, (a, by_season, by_all)):
        t = t[t["n"] >= MIN_BIAS_EVENTS].copy()
        t["bias"] = t["s"] / t["n"]
        t["level"] = level
        parts.append(t[cols])
    out = pd.concat(parts, ignore_index=True)
    out["lead_time_days"] = out["lead_time_days"].astype(int)
    out["n"] = out["n"].astype(int)
    order = {lv: i for i, lv in enumerate(LEVELS)}
    return (out.assign(_o=out["level"].map(order))
               .sort_values(["_o"] + BIAS_KEYS).drop(columns="_o").reset_index(drop=True))


def fit_bias_table(events: pd.DataFrame) -> pd.DataFrame:
    """The table from one frame of training events (`BIAS_KEYS` + `fc_mean`, `obs`)."""
    return finish_bias_table(accumulate_bias(None, events))


def bias_for(frame: pd.DataFrame, table: pd.DataFrame) -> np.ndarray:
    """Each row's bias, looked up by `BIAS_KEYS`; NaN where the table has no value. Rows
    of variables outside the table (wind direction) get NaN too - callers leave them alone.

    Looked up through a small key -> value dict over the frame's distinct key
    combinations, never one string per row: a cached training year has ~76 M rows.
    """
    if frame.empty:
        return np.empty(0, dtype=float)
    lookup = {(str(r), str(v), int(l), str(s)): float(b)
              for r, v, l, s, b in table[BIAS_KEYS + ["bias"]].itertuples(index=False)}

    def _get(r, v, k, s):
        """Finest level first, then the district-season across leads, then all year."""
        for key in ((r, v, k, s), (r, v, _ANY_LEAD, s), (r, v, _ANY_LEAD, _ANY_SEASON)):
            if key in lookup:
                return lookup[key]
        return np.nan
    rc, r_cats = _codes(frame["region_id"])
    vc, v_cats = _codes(frame["variable"])
    sc, s_cats = _codes(frame["season"])
    lead = pd.to_numeric(frame["lead_time_days"], errors="coerce").to_numpy(dtype=float)
    lc = np.where(np.isfinite(lead), lead, -1).astype(np.int64)
    n_lead = int(lc.max()) + 1 if (lc >= 0).any() else 1
    # Dense (district, variable, lead, season) grid: ~666 x 8 x 11 x 6 cells, built from
    # the frame's own category lists, then indexed by integer codes.
    dense = np.full((len(r_cats), len(v_cats), n_lead, len(s_cats)), np.nan)
    for i, r in enumerate(r_cats):
        for j, v in enumerate(v_cats):
            for k in range(n_lead):
                for m, s in enumerate(s_cats):
                    dense[i, j, k, m] = _get(str(r), str(v), k, str(s))
    out = np.full(len(frame), np.nan)
    ok = (rc >= 0) & (vc >= 0) & (sc >= 0) & (lc >= 0)
    out[ok] = dense[rc[ok], vc[ok], lc[ok], sc[ok]]
    return out


def label_variable_mask(variable: pd.Series) -> np.ndarray:
    """Rows whose variable is a label variable - through category codes, never one string
    per row."""
    codes, cats = _codes(variable)
    keep = np.array([str(c) in contracts.LABEL_VARIABLES for c in cats] + [False])
    return keep[np.where(codes >= 0, codes, len(cats))]


def _codes(col: pd.Series):
    """Integer codes and their categories, without one string per row."""
    if isinstance(col.dtype, pd.CategoricalDtype):
        return col.cat.codes.to_numpy().astype(np.int64), list(col.cat.categories)
    codes, cats = pd.factorize(col, sort=False)
    return codes.astype(np.int64), list(cats)


def apply_bias(frame: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """In place, for label-variable rows: `forecast_value` becomes the bias-corrected
    forecast, `abs_error` its error against `observed_value` (when that column exists),
    and `bias_correction` holds the bias removed. A row whose cell has no bias gets NaN
    forecast and error - it cannot be corrected, so it is not compared. Other variables'
    rows are untouched; their `bias_correction` is NaN."""
    b = bias_for(frame, table)
    if "forecast_value_raw" not in frame.columns:
        # GEFS's own forecast, kept for display: the site shows the forecast that might
        # bust, not the corrected value the model reads.
        frame["forecast_value_raw"] = frame["forecast_value"].to_numpy(dtype=float)
    frame["bias_correction"] = b
    label = label_variable_mask(frame["variable"])
    fv = frame["forecast_value"].to_numpy(dtype=float).copy()
    fv[label] = fv[label] - b[label]
    frame["forecast_value"] = fv
    if "observed_value" in frame.columns:
        err = (frame["abs_error"].to_numpy(dtype=float).copy() if "abs_error" in frame
               else np.full(len(frame), np.nan))
        err[label] = np.abs(fv[label] - frame["observed_value"].to_numpy(dtype=float)[label])
        frame["abs_error"] = err
    return frame
