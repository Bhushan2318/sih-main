"""Isotonic Distributional Regression (IDR) as a rung on the baseline ladder.

Henzi, Ziegel & Gneiting (2021, JRSS-B), doi:10.1111/rssb.12450: a non-parametric
estimate of the conditional distribution of the outcome given a covariate, subject only
to a monotonicity constraint - no parametric family assumed, unlike EMOS's half-normal.
The paper states plainly that non-parametric isotonic *binary* regression is a special
case of IDR - and ``y_bust`` already is binary, so fitting IDR here is exactly isotonic
regression of ``y_bust`` on the covariate (spread), via
``sklearn.isotonic.IsotonicRegression`` - the same tool app/ml/verification.py's CORP
reliability curve (D3) already uses and trusts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from app.ml.baselines import ALL_BASELINES, IDRBaseline, brier, fit_all

rng = np.random.default_rng(0)


def _events(n=2000, seed=0):
    """Bust probability genuinely increasing in spread - the structure IDR exists to
    exploit. Not a metric fixture: this is plumbing, no score from it is reported."""
    r = np.random.default_rng(seed)
    spread = r.gamma(2.0, 1.0, n)
    p_bust = np.clip(spread / (spread.max() + 1e-6), 0.02, 0.98)
    y = r.binomial(1, p_bust)
    return pd.DataFrame({
        "spread_mean": spread,
        "lead_time_days": r.integers(1, 11, n),
        "y_bust": y,
    })


def test_it_is_on_the_ladder():
    assert IDRBaseline in ALL_BASELINES
    assert "IDR" in fit_all(_events())


def test_it_matches_isotonic_regression_fit_independently():
    """The defining claim: IDR of a binary outcome on one covariate IS isotonic
    regression of that outcome on the covariate. Fit sklearn's IsotonicRegression
    directly here - not through the module under test - and confirm IDRBaseline
    reproduces it exactly on the training covariate values."""
    tr = _events(seed=1)
    m = IDRBaseline().fit(tr)

    order = np.argsort(tr["spread_mean"].to_numpy())
    x_sorted = tr["spread_mean"].to_numpy()[order]
    y_sorted = tr["y_bust"].to_numpy()[order]
    independent_fit = IsotonicRegression(out_of_bounds="clip").fit(x_sorted, y_sorted)
    expected = independent_fit.predict(tr["spread_mean"].to_numpy())

    got = m.predict_proba(tr)
    # both are clipped to (eps, 1-eps) by the baseline; compare loosely at that scale
    assert got == pytest.approx(np.clip(expected, 1e-6, 1 - 1e-6), abs=1e-4)


def test_bust_probability_is_non_decreasing_in_spread():
    """Stochastic monotonicity is the entire constraint IDR fits under - a wider
    ensemble must never predict a LOWER bust probability than a narrower one."""
    m = IDRBaseline().fit(_events())
    spreads = np.linspace(0.1, 8.0, 40)
    p = m.predict_proba(pd.DataFrame({"spread_mean": spreads}))
    assert np.all(np.diff(p) >= -1e-9)


def test_it_beats_climatology_when_spread_carries_signal():
    tr, te = _events(seed=2), _events(seed=3)
    m = IDRBaseline().fit(tr)
    p = m.predict_proba(te)
    clim = np.full(len(te), tr["y_bust"].mean())
    assert brier(te["y_bust"], p) < brier(te["y_bust"], clim)


def test_probabilities_stay_in_range():
    m = IDRBaseline().fit(_events())
    p = m.predict_proba(_events(seed=4))
    assert p.min() > 0.0 and p.max() < 1.0 and np.isfinite(p).all()


def test_falls_back_to_spread_max_when_spread_mean_absent():
    tr = _events(seed=5).rename(columns={"spread_mean": "spread_max"})
    m = IDRBaseline().fit(tr)
    assert m._covariate == "spread_max"
    assert np.isfinite(m.predict_proba(tr)).all()


def test_no_usable_covariate_falls_back_to_the_base_rate():
    tr = pd.DataFrame({"y_bust": [0, 1, 0, 0, 1, 0], "lead_time_days": [1, 2, 3, 4, 5, 6]})
    m = IDRBaseline().fit(tr)
    p = m.predict_proba(tr)
    assert np.allclose(p, tr["y_bust"].mean(), atol=1e-6)


def test_fit_is_not_redone_at_prediction_time():
    tr = _events(seed=6)
    m = IDRBaseline().fit(tr)
    before = m._iso.predict(np.array([1.0, 2.0, 3.0])).copy()
    m.predict_proba(_events(seed=7))
    after = m._iso.predict(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(before, after)


def test_it_is_distinct_from_emos():
    tr, te = _events(seed=8), _events(seed=9)
    fitted = fit_all(tr)
    idr = fitted["IDR"].predict_proba(te)
    emos = fitted["EMOS"].predict_proba(te)
    assert not np.allclose(idr, emos, atol=1e-3)
