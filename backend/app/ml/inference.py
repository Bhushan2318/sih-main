"""Read-side inference: load the current run's models and score the newest forecast
cycle in the canonical store.

Everything here traces to a real trained run. If no run is current, or the store has no
forecasts, callers get an explicit "not trained / no data" answer - never a placeholder
number.

The scored cycle is the latest `init_date` present in the canonical store. Results are
cached in-process keyed by (run_id, init_date, dataset row count) so repeated dashboard
polls don't re-run XGBoost; the cache invalidates automatically when a retrain flips
`current.json` or a new upload changes the row count.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from app.features import engineering as fe
from app.features import pivot as pv
from app.ml import registry
from app.ml.thresholds import Thresholds
from app.storage import parquet_store


@dataclass
class ModelState:
    """A fully-loaded, ready-to-serve view of the current training run."""

    run_id: str
    regressors: dict                      # variable -> (model, feature_columns)
    classifier: object
    classifier_columns: list
    thresholds: Thresholds
    historical_bust_freq: dict
    shap_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    manifest: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    @property
    def variables(self) -> list:
        return sorted(self.regressors)


@dataclass
class ScoredCycle:
    """Per (region, lead) predictions for one forecast cycle."""

    run_id: str
    init_date: pd.Timestamp
    events: pd.DataFrame          # one row per (region_id, valid_date, lead_time_days)
    per_variable: pd.DataFrame    # one row per (region_id, lead_time_days, variable)
    n_rows_scored: int


_lock = threading.Lock()
_state_cache: tuple | None = None      # (run_id, ModelState)
_score_cache: tuple | None = None      # (cache_key, ScoredCycle)


def load_model_state(run_id: Optional[str] = None) -> Optional[ModelState]:
    """Load (and memoise) the current run. None when nothing has been trained yet."""
    global _state_cache
    rid = run_id or registry.current_run_id()
    if rid is None:
        return None

    with _lock:
        if _state_cache and _state_cache[0] == rid:
            return _state_cache[1]

    regressors = registry.load_regressors(rid)
    clf, clf_cols = registry.load_classifier(rid)
    thr = registry.load_thresholds(rid)
    if not regressors or clf is None or thr is None:
        return None

    shap_path = registry.run_dir(rid) / "shap_summary.parquet"
    shap_summary = pd.read_parquet(shap_path) if shap_path.exists() else pd.DataFrame()

    import json
    def _read(name):
        p = registry.run_dir(rid) / name
        return json.loads(p.read_text()) if p.exists() else {}

    state = ModelState(
        run_id=rid,
        regressors=regressors,
        classifier=clf,
        classifier_columns=clf_cols,
        thresholds=thr,
        historical_bust_freq=registry.load_historical_bust_freq(rid),
        shap_summary=shap_summary,
        manifest=_read("manifest.json"),
        metrics=_read("metrics.json"),
    )
    with _lock:
        _state_cache = (rid, state)
    return state


def invalidate_caches() -> None:
    """Called after a retrain or a new ingest."""
    global _state_cache, _score_cache
    with _lock:
        _state_cache = None
        _score_cache = None


def score_latest_cycle(state: Optional[ModelState] = None) -> Optional[ScoredCycle]:
    """Score every (region, lead day) of the most recent forecast cycle in the store."""
    global _score_cache
    state = state or load_model_state()
    if state is None:
        return None

    canonical = parquet_store.read_dataset()
    if canonical.empty:
        return None
    fc_rows = canonical[canonical["value_type"] == "forecast"]
    if fc_rows.empty:
        return None

    latest_init = pd.to_datetime(fc_rows["init_date"]).max()
    cache_key = (state.run_id, str(latest_init), len(canonical))
    with _lock:
        if _score_cache and _score_cache[0] == cache_key:
            return _score_cache[1]

    # Keep the target cycle plus every observation, so features that look back at earlier
    # valid_dates (rate-of-change, verified error lag) can still be computed.
    cycle_mask = (canonical["value_type"] == "forecast") & (
        pd.to_datetime(canonical["init_date"]) == latest_init
    )
    subset = canonical[cycle_mask | (canonical["value_type"] == "observed")]

    frame = fe.build_training_frame(
        subset,
        historical_bust_freq=state.historical_bust_freq or None,
        require_observed=False,
    )
    if frame.empty:
        return None

    # ---- per-variable predicted error + confidence --------------------------------
    pred = pd.Series(np.nan, index=frame.index, dtype=float)
    for var, (model, cols) in state.regressors.items():
        mask = frame["variable"] == var
        if not mask.any():
            continue
        X = _prep(frame.loc[mask], cols, fe_categorical=True)
        pred.loc[mask] = model.predict(X)
    frame["pred_err"] = pred

    scored = frame[frame["pred_err"].notna()].copy()
    if scored.empty:
        return None

    # ---- event grain + bust probability -------------------------------------------
    events = pv.build_event_frame(
        scored, scored["pred_err"], state.thresholds.p90_error,
        bust_threshold=None, historical_bust_freq=state.historical_bust_freq or None,
    )
    X_evt = _prep(events, state.classifier_columns, fe_categorical=False)
    events["bust_probability"] = state.classifier.predict_proba(X_evt)[:, 1]
    events["risk_band"] = [state.thresholds.band_for(p) for p in events["bust_probability"]]

    # dominant variable = the one furthest past its own bust threshold
    events["dominant_variable"] = _dominant_variable(events, state.thresholds.bust_threshold)

    per_variable = _per_variable_table(scored, state)

    result = ScoredCycle(
        run_id=state.run_id,
        init_date=latest_init,
        events=events,
        per_variable=per_variable,
        n_rows_scored=len(scored),
    )
    with _lock:
        _score_cache = (cache_key, result)
    return result


def _prep(df: pd.DataFrame, cols: list, fe_categorical: bool) -> pd.DataFrame:
    """Reindex to the model's frozen column order; cast the two categoricals."""
    categorical = {"region_id", "season"}
    X = pd.DataFrame(index=df.index)
    for c in cols:
        if c in df.columns:
            X[c] = df[c]
        else:
            X[c] = np.nan       # feature absent in this slice -> genuinely missing
    for c in cols:
        if c in categorical:
            X[c] = X[c].astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _dominant_variable(events: pd.DataFrame, bust_threshold: dict) -> list:
    ratio_cols = {}
    for var, thr in bust_threshold.items():
        col = f"pred_err_{var}"
        if col in events.columns and thr:
            ratio_cols[var] = events[col] / thr
    if not ratio_cols:
        return [None] * len(events)
    ratios = pd.DataFrame(ratio_cols, index=events.index)
    if ratios.isna().all(axis=1).all():
        return [None] * len(events)
    return ratios.idxmax(axis=1, skipna=True).where(ratios.notna().any(axis=1)).tolist()


def _per_variable_table(scored: pd.DataFrame, state: ModelState) -> pd.DataFrame:
    """Ensemble-mean forecast, predicted error, confidence and (where the forecast has
    verified) the observed value, per region x lead x variable."""
    g = (
        scored.groupby(["region_id", "lead_time_days", "variable", "valid_date"],
                       observed=True)
        .agg(
            predicted_value=("forecast_value", "mean"),
            observed_value=("observed_value", "mean"),
            predicted_error=("pred_err", "mean"),
            ensemble_spread=("ensemble_spread", "mean"),
            ensemble_member_count=("ensemble_member_count", "max"),
        )
        .reset_index()
    )
    p90 = g["variable"].map(state.thresholds.p90_error)
    g["confidence"] = (1.0 - g["predicted_error"] / p90).clip(0.0, 1.0)
    g["bust_threshold"] = g["variable"].map(state.thresholds.bust_threshold)
    return g


def model_validation_metrics(state: ModelState) -> dict:
    """The persisted per-model metrics, preferring the held-out test split."""
    metrics = state.metrics or {}
    regs = {}
    for var, m in (metrics.get("regressors") or {}).items():
        chosen = m.get("test") or m.get("val") or m.get("train") or {}
        regs[var] = {
            "mae": chosen.get("mae"),
            "rmse": chosen.get("rmse"),
            "r2": chosen.get("r2"),
            "baseline_mae": chosen.get("baseline_mae_predict_mean"),
            "split": "test" if m.get("test") else ("val" if m.get("val") else "train"),
            "n": chosen.get("n"),
        }
    clf = metrics.get("classifier") or {}
    chosen_clf = clf.get("test") or clf.get("val") or clf.get("train") or {}
    return {
        "regressors": regs,
        "classifier": {
            **{k: chosen_clf.get(k) for k in
               ("n", "bust_rate", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier")},
            "split": "test" if clf.get("test") else ("val" if clf.get("val") else "train"),
        },
    }
