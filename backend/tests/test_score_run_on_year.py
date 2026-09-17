"""scripts/score_run_on_year.py: score a saved run's classifier against one full calendar
year, independent of that run's own train/val/test split.

Why this needs its own scorer rather than reusing the eval frame a normal retrain already
writes: two runs trained on different years cannot be compared by their own held-out
ROC-AUC (see docs/known-issues.md - the within-year split always tests on Nov-Dec, the
easiest season). Comparing several runs fairly needs every one of them scored on the same
year. Fixture: the real ingested slice test_ml.py already builds and trains on (module-
scoped, so this file shares it rather than retraining).
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_ml import _ingested_slice, _retrain  # noqa: F401 - reused fixtures


def test_score_run_on_year_returns_finite_probabilities_and_real_labels(_retrain):
    from app.ml import classifier as clf_mod
    from scripts.score_run_on_year import score_run_on_year

    report = _retrain
    assert report.status == "success", report.error

    events = score_run_on_year(report.run_id, 2019)
    assert not events.empty
    assert "model_proba" in events and "y_bust" in events
    assert events["model_proba"].between(0, 1).all()
    assert set(events["y_bust"].dropna().unique()) <= {0.0, 1.0}
    assert (events["scored_run_id"] == report.run_id).all()
    assert (events["scored_year"] == 2019).all()

    metrics = clf_mod._evaluate(events["y_bust"], events["model_proba"])
    assert np.isfinite(metrics["roc_auc"])


def test_score_run_on_year_uses_the_runs_own_historical_bust_freq_not_a_fresh_one(_retrain):
    """Recomputing historical_bust_frequency_region_season on the scoring year - instead
    of reusing the run's own train-split statistic - would leak that year's own bust rate
    into a feature meant to describe what the model learned before ever seeing it."""
    from app.ml import registry
    from scripts.score_run_on_year import score_run_on_year

    report = _retrain
    hbf = registry.load_historical_bust_freq(report.run_id)
    assert hbf, "the fixture run must have saved a non-empty historical bust frequency"

    events = score_run_on_year(report.run_id, 2019)
    # Every value present must come from the saved table, not be recomputed - spot check
    # by confirming the column only ever takes values that exist in the saved dict.
    saved_values = set(hbf.values())
    seen = set(events["historical_bust_frequency_region_season"].dropna().unique())
    assert seen <= saved_values, (
        f"{seen - saved_values} did not come from the run's own saved historical bust "
        f"frequency - looks recomputed on the scoring year instead")


def test_score_run_on_year_refuses_a_year_with_no_data(_retrain):
    from scripts.score_run_on_year import score_run_on_year

    report = _retrain
    with pytest.raises(ValueError, match="1999"):
        score_run_on_year(report.run_id, 1999)


def test_score_run_on_year_refuses_an_incomplete_run():
    from scripts.score_run_on_year import score_run_on_year

    with pytest.raises(ValueError, match="no complete saved model state"):
        score_run_on_year("run_this_does_not_exist_00000000T000000Z", 2019)
