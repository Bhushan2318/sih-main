"""EMOS / Non-homogeneous Gaussian Regression as a rung on the baseline ladder.

The ladder currently runs climatology -> lead-day -> spread -> lead+spread+season ->
Sanket. Every rung below the top is something we invented, which makes the comparison
weaker than it looks: beating four baselines of your own design is not the same as
beating the method the field actually uses.

EMOS (Gneiting et al., 2005, "Calibrated probabilistic forecasting using ensemble model
output statistics and minimum CRPS estimation") is that method. It is the standard
statistical post-processing for ensemble forecasts, and the obvious question from anyone
who works with ensembles is "did you compare against EMOS?".

What makes it EMOS rather than another logistic regression on spread: the ensemble spread
sets the *variance of a predictive distribution*, not a linear term in a probability. The
existing SpreadBaseline uses spread as a feature; this uses it as a scale parameter, which
is the whole point of the "non-homogeneous" in NGR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.baselines import ALL_BASELINES, EMOSBaseline, brier, fit_all

rng = np.random.default_rng(0)


def _events(n=4000, spread_scale=1.0, seed=0):
    """Errors whose magnitude genuinely scales with the ensemble spread - the structure
    EMOS exists to exploit. Not a metric fixture: this is plumbing, and no score from it
    is reported anywhere."""
    r = np.random.default_rng(seed)
    spread = r.gamma(2.0, spread_scale, n)
    err = np.abs(r.normal(0.0, spread))
    thr = float(np.percentile(err, 90))
    return pd.DataFrame({
        "actual_err_rainfall_mm": err,
        "spread_rainfall_mm": spread,
        "lead_time_days": r.integers(1, 11, n),
        "y_bust": (err >= thr).astype(int),
    })


def test_it_is_on_the_ladder():
    assert EMOSBaseline in ALL_BASELINES
    assert "EMOS" in fit_all(_events()).keys() or any(
        "emos" in k.lower() for k in fit_all(_events()))


def test_spread_sets_the_scale_not_a_coefficient():
    """The defining property. A wider ensemble must raise the bust probability for the
    same variable, because it widens the predictive distribution."""
    m = EMOSBaseline().fit(_events())
    narrow = pd.DataFrame({"spread_rainfall_mm": [0.2] * 50, "lead_time_days": [3] * 50})
    wide = pd.DataFrame({"spread_rainfall_mm": [5.0] * 50, "lead_time_days": [3] * 50})
    assert m.predict_proba(wide).mean() > m.predict_proba(narrow).mean()


def test_it_beats_climatology_when_spread_carries_signal():
    """If it cannot beat a constant on data built so that spread predicts error, the
    implementation is wrong - not the method."""
    tr, te = _events(seed=1), _events(seed=2)
    m = EMOSBaseline().fit(tr)
    p = m.predict_proba(te)
    clim = np.full(len(te), tr["y_bust"].mean())
    assert brier(te["y_bust"], p) < brier(te["y_bust"], clim)


def test_probabilities_stay_in_range():
    m = EMOSBaseline().fit(_events())
    p = m.predict_proba(_events(seed=3))
    assert p.min() > 0.0 and p.max() < 1.0 and np.isfinite(p).all()


def test_thresholds_come_from_training_rows_only():
    """The bust threshold is the 90th percentile of that variable's own error, computed
    on training data only. A baseline that recomputed it on the rows it is scored against
    would be reading the answer."""
    tr = _events(seed=4)
    m = EMOSBaseline().fit(tr)
    before = dict(m.thresholds_)
    m.predict_proba(_events(seed=5, spread_scale=9.0))
    assert m.thresholds_ == before


def test_a_variable_absent_at_prediction_time_is_survived():
    m = EMOSBaseline().fit(_events())
    p = m.predict_proba(pd.DataFrame({"lead_time_days": [1, 2, 3]}))
    assert len(p) == 3 and np.isfinite(p).all()


def test_no_usable_variable_falls_back_to_the_base_rate():
    """Refuse to invent a number: with nothing to condition on, the honest answer is
    climatology."""
    tr = pd.DataFrame({"y_bust": [0, 1, 0, 0], "lead_time_days": [1, 2, 3, 4]})
    m = EMOSBaseline().fit(tr)
    p = m.predict_proba(tr)
    assert np.allclose(p, tr["y_bust"].mean(), atol=1e-6)


def test_zero_spread_does_not_divide_by_zero():
    ev = _events()
    ev["spread_rainfall_mm"] = 0.0
    m = EMOSBaseline().fit(ev)
    assert np.isfinite(m.predict_proba(ev)).all()


def test_it_is_distinct_from_the_spread_baseline():
    """Two rungs that produce identical numbers are one rung."""
    tr, te = _events(seed=6), _events(seed=7)
    fitted = fit_all(tr)
    emos = fitted["EMOS"].predict_proba(te)
    spread = fitted["spread"].predict_proba(te)
    assert not np.allclose(emos, spread, atol=1e-3)


# ------------------------------------------- the dependence correction

def test_it_corrects_for_dependence_between_variables():
    """The independence product over-forecasts. With eight variables each exceeding its
    own 90th percentile 10% of the time, 1-(0.9^8) = 0.570 against an observed bust rate
    of 0.434 on the real data. That leaves the ranking intact and ruins the Brier score.

    One fitted parameter absorbs it: P(bust) = 1 - (prod P_no)^gamma. gamma below 1 means
    the variables move together, and gamma x n_variables is the effective number of
    independent ones - a number worth reporting rather than a fudge factor.
    """
    m = EMOSBaseline().fit(_events())
    assert 0.0 < m.dependence_ <= 1.0


def test_the_correction_improves_calibration_on_correlated_variables():
    """Two variables that always miss together. Treated as independent, the predicted
    bust rate is far above the observed one."""
    r = np.random.default_rng(11)
    n = 4000
    spread = r.gamma(2.0, 1.0, n)
    shared = np.abs(r.normal(0.0, spread))          # one shared driver
    ev = pd.DataFrame({
        "actual_err_a": shared, "spread_a": spread,
        "actual_err_b": shared * 1.01, "spread_b": spread,
        "lead_time_days": r.integers(1, 11, n),
    })
    thr = float(np.percentile(shared, 90))
    ev["y_bust"] = (shared >= thr).astype(int)

    m = EMOSBaseline().fit(ev)
    predicted = m.predict_proba(ev).mean()
    observed = ev["y_bust"].mean()
    assert abs(predicted - observed) < 0.10, \
        f"predicted {predicted:.3f} against observed {observed:.3f}"
    assert m.dependence_ < 0.95, "perfectly dependent variables must not look independent"


def test_the_correction_is_fitted_on_training_rows_only():
    tr = _events(seed=20)
    m = EMOSBaseline().fit(tr)
    before = m.dependence_
    m.predict_proba(_events(seed=21, spread_scale=7.0))
    assert m.dependence_ == before
