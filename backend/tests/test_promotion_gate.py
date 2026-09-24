"""The gate between a finished training run and the model that gets served.

`make_current` used to be unconditional: any run that completed without raising became
current, published its artifact and fired the deploy. A model that had quietly got worse
shipped itself, and the failure was indistinguishable from success.
"""

from __future__ import annotations

import json

import pytest

from app.ml import registry
from app.ml.train_pipeline import (
    MAX_ROC_AUC_REGRESSION, MIN_ROC_AUC, _held_out, _promotion_decision,
)


def _metrics(auc: float, split: str = "test") -> dict:
    return {
        "regressors": {
            "temperature_c": {split: {
                "mae": 1.2, "rmse": 1.5, "r2": 0.4, "n": 1000,
                "baseline_mae_predict_mean": 1.8,
            }},
        },
        "classifier": {split: {
            "n": 1000, "positives": 250, "bust_rate": 0.25,
            "precision": 0.7, "recall": 0.6, "f1": 0.65,
            "roc_auc": auc, "pr_auc": 0.6, "brier": 0.16,
        }},
    }


@pytest.fixture
def current_run(tmp_path, monkeypatch):
    """A run standing as current, with the metrics it recorded."""
    def _make(auc: float | None):
        run_id = "run_PREVIOUS"
        d = registry.run_dir(run_id)
        d.mkdir(parents=True, exist_ok=True)
        # Cleared each time: the run id is reused across tests, and a metrics file left by
        # an earlier one would silently stand in for the incumbent this test set up.
        (d / "metrics.json").unlink(missing_ok=True)
        if auc is not None:
            (d / "metrics.json").write_text(json.dumps(_metrics(auc)))
        registry.set_current(run_id)
        return run_id
    return _make


def test_a_first_model_is_promoted(monkeypatch):
    monkeypatch.setattr(registry, "current_run_id", lambda: None)
    ok, why = _promotion_decision(_metrics(0.84))
    assert ok and "0.84" in why


def test_an_improvement_is_promoted(current_run):
    current_run(0.80)
    ok, why = _promotion_decision(_metrics(0.86))
    assert ok
    assert "above" in why


def test_a_small_regression_still_ships(current_run):
    """Adding cycles moves the held-out split, so two runs are not scored on the same
    rows. A small difference is a different test set as much as a different model."""
    current_run(0.8428)
    ok, _ = _promotion_decision(_metrics(0.8428 - (MAX_ROC_AUC_REGRESSION / 2)))
    assert ok


def test_a_collapse_is_refused(current_run):
    """The case the gate exists for: a broken feature, an empty join, an inverted label."""
    current_run(0.84)
    ok, why = _promotion_decision(_metrics(0.60))
    assert not ok
    assert "NOT promoted" in why and "run_PREVIOUS" in why


def test_regression_exactly_at_the_tolerance_ships(current_run):
    current_run(0.84)
    ok, _ = _promotion_decision(_metrics(0.84 - MAX_ROC_AUC_REGRESSION))
    assert ok, "the boundary is inclusive; only a worse drop is refused"


def test_a_model_below_the_absolute_floor_is_refused(current_run):
    """Independent of any comparison: a classifier that cannot beat the base rate should
    not be served even if it is the first one, or better than a worse predecessor."""
    current_run(0.50)
    ok, why = _promotion_decision(_metrics(MIN_ROC_AUC - 0.01))
    assert not ok
    assert "floor" in why


def test_an_unreadable_incumbent_fails_closed(current_run):
    """Without a finite incumbent metric there is no safe regression comparison."""
    current_run(None)
    ok, why = _promotion_decision(_metrics(0.84))
    assert not ok
    assert "no finite held-out" in why


def test_a_run_without_metrics_is_not_promoted():
    ok, why = _promotion_decision({})
    assert not ok
    assert "no held-out classifier metrics" in why


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), True])
def test_nonfinite_or_boolean_classifier_auc_is_refused(bad):
    m = _metrics(0.84)
    m["classifier"]["test"]["roc_auc"] = bad
    ok, why = _promotion_decision(m)
    assert not ok
    assert "non-finite" in why


def test_degenerate_classifier_labels_are_refused():
    m = _metrics(0.84)
    m["classifier"]["test"].update(n=10, positives=0)
    ok, why = _promotion_decision(m)
    assert not ok and "degenerate" in why


def test_missing_regressor_health_is_refused():
    m = _metrics(0.84)
    del m["regressors"]
    ok, why = _promotion_decision(m)
    assert not ok and "regressor health" in why


@pytest.mark.parametrize("metric", ["mae", "rmse", "r2", "n"])
def test_nonfinite_regressor_health_is_refused(metric):
    m = _metrics(0.84)
    m["regressors"]["temperature_c"]["test"][metric] = float("nan")
    ok, why = _promotion_decision(m)
    assert not ok
    assert "regressor 'temperature_c'" in why and "non-finite" in why


def test_held_out_prefers_test_over_val():
    m = {"classifier": {"val": {"roc_auc": 0.70}, "test": {"roc_auc": 0.80}}}
    assert _held_out(m)["roc_auc"] == 0.80


def test_held_out_falls_back_to_val():
    m = {"classifier": {"val": {"roc_auc": 0.70}}}
    assert _held_out(m)["roc_auc"] == 0.70
