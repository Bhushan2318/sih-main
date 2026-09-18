from __future__ import annotations

import numpy as np
import pandas as pd

from app.features.engineering import EVENT_KEYS, JUMP_FEATURES, _season

CONF_FLOOR = 0.0


def build_event_frame(
    paired: pd.DataFrame,
    pred_err: pd.Series,
    p90_error: dict,
    bust_threshold: dict | None,
    historical_bust_freq: dict | None = None,
    copy_input: bool = True,
) -> pd.DataFrame:
    """`copy_input=False` skips the defensive copy for a caller that owns `paired`
    exclusively (freshly loaded, never read again afterward) - real crash
    2026-09-17/18: the copy doubled a 76.5M-row training-year frame's footprint
    right before the groupby below needed its own ~584 MiB contiguous block, and
    that block was what ran out. Freeing fold_models first (see
    _build_pooled_year_events_worker.py) was not enough on its own."""
    df = paired.copy() if copy_input else paired
    df["pred_err"] = pred_err.reindex(df.index).to_numpy()
    # `variable` is categorical in the retrain's paired frame. Series.map on a categorical
    # returns a categorical when every category maps to a distinct value - true for a
    # real year, where no two variables share a p90 - and a categorical cannot be divided.
    df["p90"] = df["variable"].map(p90_error).astype(np.float32)
    pred_err32 = df["pred_err"].astype(np.float32)
    df["conf"] = (1.0 - pred_err32 / df["p90"]).clip(CONF_FLOOR, 1.0)

    # Jumpiness is one value per (event, variable) repeated on every member row, so the
    # mean is that value. Absent when the frame predates C1.
    jump = [c for c in JUMP_FEATURES if c in df.columns]
    em = (df.groupby(EVENT_KEYS + ["variable"], observed=True)
            .agg(fc_mean=("forecast_value", "mean"),
                 obs=("observed_value", "mean"),
                 spread=("ensemble_spread", "mean"),
                 pred_err=("pred_err", "mean"),
                 conf=("conf", "mean"),
                 **{c: (c, "mean") for c in jump})
            .reset_index())
    em["actual_err"] = (em["fc_mean"] - em["obs"]).abs()

    pe = em.pivot_table(index=EVENT_KEYS, columns="variable",
                        values=["pred_err", "conf", "spread", "actual_err"] + jump,
                        observed=True)
    pe.columns = [f"{a}_{b}" for a, b in pe.columns]
    pe = pe.reset_index()

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
    per_var = [c for c in event_df.columns
               if c.startswith(("pred_err_", "conf_", "jump_"))
               or (c.startswith("spread_") and c not in ("spread_mean", "spread_max"))]
    context = ["lead_time_days", "month", "spread_mean", "spread_max",
               "historical_bust_frequency_region_season", "region_id", "season"]
    ordered, seen = [], set()
    for c in per_var + context:
        if c in event_df.columns and c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered
