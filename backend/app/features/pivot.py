"""Long -> wide pivot to the classifier's grain: one row per forecast *event*
(region_id, valid_date, lead_time_days, init_date).

For each event and each modelled variable:
  pred_err_<var>   mean over members of the regressor's error prediction (OOF at train
                   time, plain predict at inference)
  conf_<var>       mean over members of clip(1 - pred_err / p90_error[var], 0, 1)
  spread_<var>     mean ensemble_spread for that variable at the event

Event-level:
  region_id (cat), lead_time_days, month, season (cat),
  historical_bust_frequency_region_season,
  spread_mean / spread_max across variables

Label (train only):  y = 1 if  max_var (actual_err_<var> / bust_threshold[var]) >= 1
where actual_err is the ensemble-MEAN forecast error for that event+variable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.features.engineering import EVENT_KEYS, _season

CONF_FLOOR = 0.0


def build_event_frame(
    paired: pd.DataFrame,
    pred_err: pd.Series,
    p90_error: dict,
    bust_threshold: dict | None,
    historical_bust_freq: dict | None = None,
) -> pd.DataFrame:
    """`paired` is the per-member frame from engineering.build_training_frame, restricted
    to one split. `pred_err` is a Series aligned to paired.index with each row's predicted
    absolute error. `bust_threshold` None -> no label column (inference)."""
    df = paired.copy()
    df["pred_err"] = pred_err.reindex(df.index).to_numpy()
    df["p90"] = df["variable"].map(p90_error)
    df["conf"] = (1.0 - df["pred_err"] / df["p90"]).clip(CONF_FLOOR, 1.0)

    # ensemble-mean forecast per (event, variable) -> actual error for the label
    em = (df.groupby(EVENT_KEYS + ["variable"], observed=True)
            .agg(fc_mean=("forecast_value", "mean"),
                 obs=("observed_value", "mean"),
                 spread=("ensemble_spread", "mean"),
                 pred_err=("pred_err", "mean"),
                 conf=("conf", "mean"))
            .reset_index())
    em["actual_err"] = (em["fc_mean"] - em["obs"]).abs()

    pe = em.pivot_table(index=EVENT_KEYS, columns="variable",
                        values=["pred_err", "conf", "spread", "actual_err"], observed=True)
    pe.columns = [f"{a}_{b}" for a, b in pe.columns]
    pe = pe.reset_index()

    # event-level context
    pe["valid_date"] = pd.to_datetime(pe["valid_date"])
    pe["month"] = pe["valid_date"].dt.month
    pe["season"] = _season(pe["month"])
    pe["region_id"] = pe["region_id"].astype("category")
    pe["lead_time_days"] = pe["lead_time_days"].astype(int)

    spread_cols = [c for c in pe.columns if c.startswith("spread_")]
    pe["spread_mean"] = pe[spread_cols].mean(axis=1)
    pe["spread_max"] = pe[spread_cols].max(axis=1)

    if historical_bust_freq is not None:
        key = list(zip(pe["region_id"].astype(str), pe["season"].astype(str)))
        pe["historical_bust_frequency_region_season"] = [
            historical_bust_freq.get(k, np.nan) for k in key
        ]
    else:
        pe["historical_bust_frequency_region_season"] = np.nan

    if bust_threshold:
        ratios = []
        for var, thr in bust_threshold.items():
            col = f"actual_err_{var}"
            if col in pe.columns and thr and thr > 0:
                ratios.append(pe[col] / thr)
        pe["bust_ratio"] = pd.concat(ratios, axis=1).max(axis=1) if ratios else np.nan
        pe["y_bust"] = (pe["bust_ratio"] >= 1.0).astype(int)

    return pe


def classifier_feature_columns(event_df: pd.DataFrame) -> list:
    """Feature columns for the bust classifier - the OOF-predicted signals plus event
    context. Deliberately excludes every actual_err_* / bust_ratio / y_bust column."""
    per_var = [c for c in event_df.columns
               if c.startswith(("pred_err_", "conf_"))
               or (c.startswith("spread_") and c not in ("spread_mean", "spread_max"))]
    context = ["lead_time_days", "month", "spread_mean", "spread_max",
               "historical_bust_frequency_region_season", "region_id", "season"]
    ordered, seen = [], set()
    for c in per_var + context:
        if c in event_df.columns and c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered
