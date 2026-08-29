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
import xgboost as xgb

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
    classifier: xgb.XGBClassifier
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
_score_cache: "dict[tuple, ScoredCycle]" = {}   # cache_key -> ScoredCycle (bounded)
_SCORE_CACHE_MAX = 24                  # guided replay flips between many historical cycles


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
    global _state_cache
    with _lock:
        _state_cache = None
        _score_cache.clear()
    try:  # deferred: replay_service imports this module
        from app.services import replay_service
        replay_service.invalidate()
    except Exception:  # noqa: BLE001 - never let cache cleanup break an ingest
        pass


def score_latest_cycle(state: Optional[ModelState] = None) -> Optional[ScoredCycle]:
    """Score every (region, lead day) of the most recent forecast cycle in the store."""
    return score_cycle(state, init_date=None)


def available_cycles() -> list[pd.Timestamp]:
    """Every forecast ``init_date`` in the canonical store, newest first. Drives the
    guided-replay cycle picker."""
    canonical = parquet_store.read_dataset(columns=["value_type", "init_date"])
    if canonical.empty:
        return []
    fc = canonical[canonical["value_type"] == "forecast"]
    if fc.empty:
        return []
    return sorted(pd.to_datetime(fc["init_date"]).dropna().unique(), reverse=True)


def score_cycle(
    state: Optional[ModelState] = None,
    init_date: "Optional[pd.Timestamp | str]" = None,
) -> Optional[ScoredCycle]:
    """Score every (region, lead day) of one forecast cycle.

    ``init_date=None`` scores the most recent cycle in the store (the live-dashboard
    path). Passing an explicit ``init_date`` scores that historical cycle instead - the
    guided replay steps through such a cycle lead day by lead day. Results are memoised
    per (run_id, init_date, store fingerprint), and the cache is probed *before* the
    parquet read so a warm call is essentially free - the dashboard leans on this for
    every lead-day switch.
    """
    state = state or load_model_state()
    if state is None:
        return None

    # Resolve the target init and probe the cache without touching the row data. The
    # fingerprint is dir names + mtimes; latest_forecast_init_date reads one column.
    fp = parquet_store.store_fingerprint()
    if init_date is None:
        latest = parquet_store.latest_forecast_init_date()
        if latest is None:
            return None
        target_init = pd.Timestamp(latest).normalize()
    else:
        target_init = pd.Timestamp(init_date).normalize()

    cache_key = (state.run_id, str(target_init), fp)
    with _lock:
        hit = _score_cache.get(cache_key)
        if hit is not None:
            return hit

    canonical = parquet_store.read_dataset()
    if canonical.empty:
        return None
    fc_rows = canonical[canonical["value_type"] == "forecast"]
    if fc_rows.empty:
        return None
    inits = pd.to_datetime(fc_rows["init_date"])
    if init_date is not None and not (inits.dt.normalize() == target_init).any():
        return None

    # Keep the target cycle plus every observation, so features that look back at earlier
    # valid_dates (rate-of-change, verified error lag) can still be computed.
    keep = ((canonical["value_type"] == "forecast")
            & (pd.to_datetime(canonical["init_date"]).dt.normalize() == target_init)) \
        | (canonical["value_type"] == "observed")
    # pd.DataFrame(...) pins the type: boolean-mask indexing is typed as Series | DataFrame
    # by the pandas stubs. Runtime no-op - build_training_frame copies its argument anyway.
    subset = pd.DataFrame(canonical[keep])

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

    scored = frame.loc[frame["pred_err"].notna()].copy()
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
        init_date=target_init,
        events=events,
        per_variable=per_variable,
        n_rows_scored=len(scored),
    )
    with _lock:
        if len(_score_cache) >= _SCORE_CACHE_MAX:
            _score_cache.pop(next(iter(_score_cache)))   # drop oldest insert
        _score_cache[cache_key] = result
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
    if bool(ratios.isna().to_numpy().all()):
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
    g["confidence"] = g["predicted_error"].div(p90).rsub(1.0).clip(0.0, 1.0)
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
