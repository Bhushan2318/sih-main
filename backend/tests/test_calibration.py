"""Bust probabilities that mean what they say.

The served classifier trained with scale_pos_weight = neg/pos, which pushes every
probability up, and was never recalibrated: its lowest reliability bin predicted 0.158
against an observed rate of ~0.07 on test (metrics.json). New runs drop the weighting and
fit Platt scaling on the validation year: p' = sigmoid(a * logit(p) + b), a > 0. It is
monotone, so every ranking - and the ROC-AUC the gate reads - is unchanged.

SYNTHETIC, LABELLED: the miscalibrated scores below are generated to check the arithmetic
of the fit; none reaches a reported metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml import calibration as cal
from app.ml import classifier as clf_mod


def _miscalibrated(n=20_000, seed=0):
    """Scores that are systematically too high: the true probability is a shrunk logit."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.02, 0.98, n)
    true = 1.0 / (1.0 + np.exp(-(0.5 * np.log(raw / (1 - raw)) - 1.0)))
    y = rng.binomial(1, true)
    return raw, y


def _reliability_gap(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    gaps = [abs(p[idx == b].mean() - y[idx == b].mean()) for b in range(bins) if (idx == b).sum() > 50]
    return float(np.mean(gaps))


def test_platt_scaling_repairs_a_systematic_overestimate():
    raw, y = _miscalibrated()
    c = cal.fit_platt(raw, y)
    fixed = cal.apply(c, raw)
    assert _reliability_gap(fixed, y) < 0.25 * _reliability_gap(raw, y)


def test_calibration_never_changes_the_ranking():
    from app.ml.verification import trapezoidal_auc
    raw, y = _miscalibrated()
    fixed = cal.apply(cal.fit_platt(raw, y), raw)
    assert trapezoidal_auc(y, fixed) == pytest.approx(trapezoidal_auc(y, raw), abs=1e-12)


def test_a_fit_that_would_reverse_the_ranking_is_refused():
    raw, y = _miscalibrated()
    assert cal.fit_platt(raw, 1 - y) is None
    assert np.array_equal(cal.apply(None, raw), raw)


def test_the_classifier_no_longer_weights_the_positive_class():
    assert "scale_pos_weight" not in clf_mod.XGB_PARAMS
    assert clf_mod.XGB_PARAMS["n_estimators"] >= 2000
    assert clf_mod.XGB_PARAMS["early_stopping_rounds"] >= 50


def test_the_artifact_applies_its_calibrator():
    class Model:
        def predict_proba(self, X):
            return np.column_stack([1 - X["x"].to_numpy(), X["x"].to_numpy()])
    art = clf_mod.ClassifierArtifact(model=Model(), feature_columns=["x"], metrics={},
                                     n_train=0, n_val=0, train_bust_rate=0.5,
                                     calibrator={"method": "platt", "a": 1.0, "b": -1.0})
    p = clf_mod.predict_bust_probability(art, pd.DataFrame({"x": [0.5]}))
    assert p[0] == pytest.approx(1.0 / (1.0 + np.exp(1.0)))


def test_the_calibrator_round_trips_through_the_registry(tmp_path, monkeypatch):
    from app.ml import registry
    monkeypatch.setattr(registry, "MODEL_DIR", tmp_path)
    registry.save_calibrator("run_x", {"method": "platt", "a": 0.8, "b": -0.3})
    assert registry.load_calibrator("run_x") == {"method": "platt", "a": 0.8, "b": -0.3}
    assert registry.load_calibrator("run_without") is None


def test_live_scoring_applies_a_runs_calibrator_and_leaves_older_runs_alone():
    from app.ml import inference

    class Model:
        def predict_proba(self, X):
            return np.column_stack([1 - X["x"].to_numpy(), X["x"].to_numpy()])

    class State:
        classifier = Model()
        classifier_columns = ["x"]
    X = pd.DataFrame({"x": [0.5]})
    assert inference.bust_probability(State(), X)[0] == pytest.approx(0.5)
    State.calibrator = {"method": "platt", "a": 1.0, "b": -1.0}
    assert inference.bust_probability(State(), X)[0] == pytest.approx(1.0 / (1.0 + np.exp(1.0)))


def test_the_classifier_fits_on_the_device_it_is_given_and_ships_for_cpu(monkeypatch):
    """The pooled run fits the classifier on CUDA when the regressors did. Measured
    2026-10-07 on the 17-year pool: on CPU it ran ~1 min per boosting round, up to ~50 h
    for the 3000-round budget. The shipped model must still be a CPU model - Render has no
    GPU - so the device is reset after the fit. Shape fixture: random values, plumbing
    only, never reaches a metric."""
    seen = []
    real = clf_mod.xgb.XGBClassifier

    class Spy(real):
        def fit(self, *a, **k):
            seen.append(self.get_params().get("device"))
            return super().fit(*a, **k)

    monkeypatch.setattr(clf_mod.xgb, "XGBClassifier", Spy)
    monkeypatch.setitem(clf_mod.XGB_PARAMS, "n_estimators", 5)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"lead_time_days": rng.integers(1, 11, 400).astype(float),
                       "y_bust": rng.integers(0, 2, 400)})
    art = clf_mod.train_bust_classifier(df, None, device="cpu")
    assert seen == ["cpu"]
    assert art.model.get_params()["device"] == "cpu"
