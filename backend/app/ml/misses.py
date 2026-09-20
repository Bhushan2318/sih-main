"""The held-out cases this model got most wrong.

Every other number the site publishes argues that the model works. This one is here to
show where it does not, which is only worth anything if the cases are picked honestly:
ranked by how confidently wrong the model was, on the held-out split only, with the
variable that actually busted named rather than summarised.

Pure functions over an eval-events frame. Nothing here imports the trainer, reads the
store, or knows about an API.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

_ERR_PREFIX = "actual_err_"


def _dominant_variable(row: pd.Series, thresholds: dict) -> tuple:
    """Which variable busted hardest on this row, as (name, error, threshold).

    "Hardest" is the largest error *relative to that variable's own threshold*, not the
    largest absolute error - the thresholds differ by two orders of magnitude between,
    say, temperature (4.45 degrees) and wind direction (169 degrees), so an absolute
    comparison would report wind direction almost every time.
    """
    best: tuple = (None, None, None)
    best_ratio = 0.0
    for var, thr in thresholds.items():
        if not thr:
            continue
        val = row.get(f"{_ERR_PREFIX}{var}")
        if val is None or not np.isfinite(val):
            continue
        ratio = float(val) / float(thr)
        if ratio > best_ratio:
            best_ratio, best = ratio, (var, float(val), float(thr))
    return best


def _rows(df: pd.DataFrame, thresholds: dict, ascending: bool, k: int) -> list:
    if df.empty:
        return []
    ordered = df.sort_values("model_proba", ascending=ascending, kind="mergesort").head(k)
    out = []
    for _, row in ordered.iterrows():
        var, err, thr = _dominant_variable(row, thresholds)
        # Date only: eval events carry a timestamp, and the time is always midnight.
        valid = row.get("valid_date")
        if valid is not None and not pd.isna(valid):
            valid = pd.Timestamp(valid).date()
        out.append({
            "region_id": row.get("region_id"),
            "valid_date": str(valid) if valid is not None and not pd.isna(valid) else None,
            "lead_time_days": int(row["lead_time_days"]) if pd.notna(row.get("lead_time_days")) else None,
            "bust_probability": round(float(row["model_proba"]), 4),
            "variable": var,
            "actual_error": None if err is None else round(err, 4),
            "threshold": None if thr is None else round(thr, 4),
        })
    return out


def worst_misses(
    events: pd.DataFrame,
    thresholds: dict,
    split: str = "test",
    k: int = 5,
    region_names: Optional[dict] = None,
) -> dict:
    """The k most confidently wrong calls in each direction.

    ``missed_busts`` - it busted and the model said it would not, lowest probability
    first. These are the expensive ones: a forecaster who trusted the forecast got no
    warning.

    ``false_alarms`` - it did not bust and the model said it would, highest probability
    first. These cost credibility rather than money, and a model with none of them is
    usually one that never warns about anything.

    Held-out rows only. Scoring misses on rows the model trained on would flatter it, and
    is the same mistake as reporting training accuracy.
    """
    if events is None or events.empty or "split" not in events.columns:
        return {"split": split, "missed_busts": [], "false_alarms": []}

    held = events[events["split"] == split]
    needed = {"y_bust", "model_proba"}
    if held.empty or not needed.issubset(held.columns):
        return {"split": split, "missed_busts": [], "false_alarms": []}

    held = held[held["model_proba"].notna() & held["y_bust"].notna()]
    missed = _rows(held[held["y_bust"] == 1], thresholds, ascending=True, k=k)
    alarms = _rows(held[held["y_bust"] == 0], thresholds, ascending=False, k=k)

    if region_names:
        for row in (*missed, *alarms):
            row["region_name"] = region_names.get(row["region_id"])

    return {"split": split, "missed_busts": missed, "false_alarms": alarms}
