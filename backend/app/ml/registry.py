from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import xgboost as xgb

from app.config import settings
from app.db.base import resolve_path
from app.ml.thresholds import Thresholds

MODEL_DIR = resolve_path(settings.model_dir)
CURRENT_JSON = MODEL_DIR / "current.json"


def new_run_id() -> str:
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_dir(run_id: str) -> Path:
    return MODEL_DIR / run_id


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def save_regressor(run_id: str, variable: str, model: xgb.XGBRegressor, feature_columns: list) -> None:
    d = run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    model.save_model(d / f"{variable}_regressor.json")
    _merge_feature_columns(run_id, {f"regressor::{variable}": feature_columns})


def save_classifier(run_id: str, model: xgb.XGBClassifier, feature_columns: list) -> None:
    d = run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    model.save_model(d / "classifier.json")
    _merge_feature_columns(run_id, {"classifier": feature_columns})


def _merge_feature_columns(run_id: str, update: dict) -> None:
    path = run_dir(run_id) / "feature_columns.json"
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur.update(update)
    _atomic_write(path, json.dumps(cur, indent=2))


def save_thresholds(run_id: str, thr: Thresholds) -> None:
    thr.to_json(run_dir(run_id) / "thresholds.json")


def save_historical_bust_freq(run_id: str, hbf: dict) -> None:
    payload = {f"{r}||{s}": v for (r, s), v in hbf.items()}
    _atomic_write(run_dir(run_id) / "historical_bust_freq.json", json.dumps(payload, indent=2))


def load_historical_bust_freq(run_id: str) -> dict:
    path = run_dir(run_id) / "historical_bust_freq.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {tuple(k.split("||")): v for k, v in raw.items()}


def save_jump_climatology(run_id: str, climatology: dict) -> None:
    """(region_id, variable) -> mean |forecast jump| on the training split (C1)."""
    payload = {f"{r}||{v}": x for (r, v), x in climatology.items()}
    _atomic_write(run_dir(run_id) / "jump_climatology.json", json.dumps(payload, indent=2))


def load_jump_climatology(run_id: str) -> dict:
    """Empty for a run trained before C1 - its models never saw jump_rel_climatology."""
    path = run_dir(run_id) / "jump_climatology.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {tuple(k.split("||")): v for k, v in raw.items()}


def save_metrics(run_id: str, metrics: dict) -> None:
    _atomic_write(run_dir(run_id) / "metrics.json", json.dumps(metrics, indent=2, default=str))


def save_manifest(run_id: str, manifest: dict) -> None:
    _atomic_write(run_dir(run_id) / "manifest.json", json.dumps(manifest, indent=2, default=str))


def set_current(run_id: str) -> None:
    _atomic_write(CURRENT_JSON, json.dumps({"run_id": run_id,
                                            "set_at": datetime.now(timezone.utc).isoformat()}))


def current_run_id() -> Optional[str]:
    if not CURRENT_JSON.exists():
        return None
    try:
        return json.loads(CURRENT_JSON.read_text())["run_id"]
    except (json.JSONDecodeError, KeyError):
        return None


def _restore_categorical(model) -> None:
    """Every model here was trained with `enable_categorical=True` (region_id and season
    are categorical), but XGBoost's JSON round-trip does not carry that sklearn-wrapper
    flag back. Real failure 2026-09-22: SHAP refused a reloaded classifier - "Invalid
    columns: season: cat" - and the explanation silently degraded to feature importance.
    With the flag restored the SHAP values equal the in-memory model's exactly."""
    model.set_params(enable_categorical=True)


def load_feature_columns(run_id: str) -> dict:
    path = run_dir(run_id) / "feature_columns.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_regressors(run_id: str) -> dict:
    d = run_dir(run_id)
    cols = load_feature_columns(run_id)
    out = {}
    for p in sorted(d.glob("*_regressor.json")):
        var = p.name[: -len("_regressor.json")]
        m = xgb.XGBRegressor()
        m.load_model(p)
        _restore_categorical(m)
        out[var] = (m, cols.get(f"regressor::{var}", []))
    return out


def load_classifier(run_id: str):
    p = run_dir(run_id) / "classifier.json"
    if not p.exists():
        return None, []
    m = xgb.XGBClassifier()
    m.load_model(p)
    _restore_categorical(m)
    return m, load_feature_columns(run_id).get("classifier", [])


def load_baselines(run_id: str) -> Optional[dict]:
    path = run_dir(run_id) / "baselines.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def load_misses(run_id: str) -> Optional[dict]:
    """The worst held-out calls, written beside the model by scripts/run_baselines.py.

    Same shape of contract as load_baselines: absent for any run trained before this
    existed, so every caller must handle None rather than assume it.
    """
    path = run_dir(run_id) / "misses.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def load_thresholds(run_id: str) -> Optional[Thresholds]:
    p = run_dir(run_id) / "thresholds.json"
    return Thresholds.from_json(p) if p.exists() else None


def list_runs() -> list:
    if not MODEL_DIR.exists():
        return []
    return sorted(p.name for p in MODEL_DIR.iterdir() if p.is_dir() and p.name.startswith("run_"))


def load_metrics(run_id: str) -> Optional[dict]:
    """The metrics a run recorded, or None if it has none readable."""
    path = run_dir(run_id) / "metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
