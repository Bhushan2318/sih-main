"""Phase 3: thresholds, feature engineering, leakage guards, and a reduced end-to-end
retrain against the real canonical data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import engineering as fe
from app.features import pivot as pv
from app.ingestion.pipeline import confirm_mapping, ingest_upload
from app.ml import registry
from app.ml.thresholds import Thresholds, compute_error_thresholds, compute_risk_bands
from tests.conftest import ERA5_CSV, GEFS_CSV


# --------------------------------------------------------------------------- thresholds

def test_error_thresholds_are_percentiles():
    df = pd.DataFrame({"variable": ["t"] * 100, "abs_error": np.arange(100.0)})
    thr = compute_error_thresholds(df, percentile=90.0)
    assert 88 <= thr["t"] <= 91


def test_risk_bands_ordered_and_serialise(tmp_path):
    proba = np.clip(np.random.default_rng(0).beta(2, 5, 500), 0, 1)
    cuts = compute_risk_bands(proba)
    assert 0 < cuts["medium"] < cuts["high"] < 1
    t = Thresholds(bust_threshold={"t": 3.0}, p90_error={"t": 5.0}, risk_band_cuts=cuts)
    assert t.band_for(0.0) == "low"
    assert t.band_for(cuts["high"] + 1e-6) == "high"
    p = tmp_path / "thr.json"
    t.to_json(p)
    back = Thresholds.from_json(p)
    assert back.bust_threshold == t.bust_threshold
    assert back.band_for(0.99) == "high"


def test_risk_bands_degenerate_input():
    cuts = compute_risk_bands(np.zeros(5))
    assert cuts["medium"] < cuts["high"]


# ---------------------------------------------------------------- feature engineering

@pytest.fixture(scope="module")
def paired_real():
    from app.storage.parquet_store import read_dataset  # noqa: PLC0415
    # this fixture assumes the module-scoped ingest below has populated the store
    return None


def test_pairing_and_target(_ingested_slice):
    from app.storage.parquet_store import read_dataset

    canon = read_dataset()
    paired = fe.build_training_frame(canon)
    assert not paired.empty
    assert "abs_error" in paired.columns
    # target is a non-negative magnitude
    assert (paired["abs_error"] >= 0).all()
    # each paired row carries both sides
    assert paired[["forecast_value", "observed_value"]].notna().all().all()
    # ensemble spread computed across members
    assert paired["ensemble_member_count"].max() >= 2
    # a row's own concurrent-variable column is blanked
    t = paired[paired["variable"] == "temperature_c"]
    assert t["fc_temperature_c"].isna().all()


def test_classifier_features_exclude_actual_error(_ingested_slice):
    from app.storage.parquet_store import read_dataset

    canon = read_dataset()
    paired = fe.build_training_frame(canon)
    cycles = sorted(paired["init_date"].dropna().unique())
    tr = paired[paired["init_date"].isin(cycles[: max(2, len(cycles) - 1)])]
    oof = pd.Series(np.random.default_rng(1).random(len(tr)), index=tr.index)
    p90 = {v: 5.0 for v in tr["variable"].unique()}
    thr = {v: 3.0 for v in tr["variable"].unique()}
    ev = pv.build_event_frame(tr, oof, p90, thr)

    feats = pv.classifier_feature_columns(ev)
    assert "y_bust" in ev.columns
    assert not any(f.startswith("actual_err_") for f in feats)
    assert "bust_ratio" not in feats and "y_bust" not in feats
    assert any(f.startswith("pred_err_") for f in feats)


# ----------------------------------------------------------------- end-to-end retrain

@pytest.fixture(scope="module")
def _retrain(_ingested_slice):
    from app.ml.train_pipeline import full_retrain
    return full_retrain(make_current=True)


def test_full_retrain_end_to_end(_retrain):
    report = _retrain
    assert report.status == "success", report.error
    assert report.made_current
    assert len(report.modelled_variables) >= 3
    assert report.classifier_metrics.get("train", {}).get("n", 0) > 0

    # thresholds are real numbers in canonical units
    for var, t in report.thresholds["bust_threshold"].items():
        assert t > 0

    # current.json points at this run and everything reloads
    rid = registry.current_run_id()
    assert rid == report.run_id
    regs = registry.load_regressors(rid)
    assert set(regs) == set(report.modelled_variables)
    clf, cols = registry.load_classifier(rid)
    assert clf is not None and len(cols) > 5
    thr = registry.load_thresholds(rid)
    assert thr.risk_band_cuts["medium"] <= thr.risk_band_cuts["high"]

    # a failed-style guard: manifest records the shap method honestly
    import json
    manifest = json.loads((registry.run_dir(rid) / "manifest.json").read_text())
    assert manifest["shap_method"] in {"shap", "feature_importance_fallback", "none"}


def test_regressors_beat_mean_baseline_on_temperature(_retrain):
    m = _retrain.regressor_metrics.get("temperature_c", {})
    ref = m.get("test") or m.get("val") or m.get("train")
    assert ref is not None
    assert ref["mae"] < ref["baseline_mae_predict_mean"]


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def _ingested_slice(request):
    """Ingest a multi-cycle slice of the real GEFS + ERA5 samples once for this module."""
    import shutil

    from app.db.base import SessionLocal, engine, init_db
    from app.db.models import Base
    from app.storage import parquet_store

    Base.metadata.drop_all(engine)
    shutil.rmtree(parquet_store.CANONICAL_DIR, ignore_errors=True)
    init_db()

    tmp = request.getfixturevalue("tmp_path_factory").mktemp("mlslice")
    # ~8 init cycles: enough for a 5/1/2 train/val/test split
    g = pd.read_csv(GEFS_CSV)
    keep_cycles = sorted(g["init_date"].unique())[:8]
    g = g[g["init_date"].isin(keep_cycles)]
    gp = tmp / "gefs.csv"; g.to_csv(gp, index=False)
    e = pd.read_csv(ERA5_CSV)
    ep = tmp / "era5.csv"; e.to_csv(ep, index=False)

    def _confirm(res, s):
        if res.status != "pending_confirmation":
            return res
        seen, conf = set(), []
        for p in sorted(res.mapping_proposals, key=lambda x: x["source_column"]):
            if p["role"] != "measurement" or p["decision"] != "needs_confirmation" or not p["suggested_variable"]:
                continue
            key = (p["suggested_variable"], p["suggested_value_type"])
            if key in seen:
                continue
            seen.add(key)
            conf.append({"source_column": p["source_column"], "variable": p["suggested_variable"],
                         "value_type": p["suggested_value_type"], "unit_conversion": p["unit_conversion"]})
        return confirm_mapping(s, res.batch_id, conf)

    s = SessionLocal()
    try:
        _confirm(ingest_upload(s, gp, gp.name), s)
        _confirm(ingest_upload(s, ep, ep.name), s)
        s.commit()
    finally:
        s.close()
    yield
    Base.metadata.drop_all(engine)
    shutil.rmtree(parquet_store.CANONICAL_DIR, ignore_errors=True)
