"""Method of analogs as a rung on the baseline ladder: the empirical bust rate among
the k most similar training cases, no fitted model at all.

Hamill & Whitaker (2006, Monthly Weather Review), doi:10.1175/MWR3237.1,
"Probabilistic Quantitative Precipitation Forecasts Based on Reforecast Analogs" - the
technique this project's own reforecast archive was built for. Implemented directly as
``sklearn.neighbors.KNeighborsClassifier``, whose ``predict_proba`` with uniform
weighting already *is* "the empirical event rate among the k nearest neighbours."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.baselines import ALL_BASELINES, AnalogBaseline, brier, fit_all

rng = np.random.default_rng(0)


def _events(n=2000, seed=0):
    """Two well-separated clusters in (lead_time_days, spread_mean) with different bust
    rates - the structure a neighbour lookup exists to exploit. Not a metric fixture."""
    r = np.random.default_rng(seed)
    half = n // 2
    lead_a = r.integers(1, 4, half)          # early lead, low spread, rarely busts
    spread_a = r.gamma(2.0, 0.3, half)
    y_a = r.binomial(1, 0.05, half)
    lead_b = r.integers(7, 11, n - half)     # late lead, high spread, often busts
    spread_b = r.gamma(2.0, 3.0, n - half)
    y_b = r.binomial(1, 0.80, n - half)
    return pd.DataFrame({
        "lead_time_days": np.concatenate([lead_a, lead_b]),
        "spread_mean": np.concatenate([spread_a, spread_b]),
        "y_bust": np.concatenate([y_a, y_b]),
    })


def test_it_is_on_the_ladder():
    assert AnalogBaseline in ALL_BASELINES
    assert "analog" in fit_all(_events())


def test_single_nearest_neighbour_matches_by_hand():
    """k=1: the predicted probability for a query point must be exactly the label of
    whichever training point is closest to it after standardisation - worked out by
    hand here, not read back from the fitted model's own internals."""
    tr = pd.DataFrame({
        "lead_time_days": [1, 1, 9, 9],
        "spread_mean": [0.1, 0.1, 5.0, 5.0],
        "y_bust": [0, 0, 1, 1],
    })
    m = AnalogBaseline(n_neighbors=1).fit(tr)
    # a query near (1, 0.1) is obviously closest to the first two rows (label 0);
    # a query near (9, 5.0) is obviously closest to the last two rows (label 1)
    query = pd.DataFrame({"lead_time_days": [1, 9], "spread_mean": [0.1, 5.0]})
    p = m.predict_proba(query)
    assert p[0] < 0.5 < p[1]


def test_three_nearest_neighbours_matches_a_hand_counted_vote():
    """k=3, a query exactly at one training point: with a tie broken by inclusion order
    in scikit-learn's ball tree, the point coincides with itself (distance 0) plus its
    two genuinely nearest neighbours - counted by hand from the constructed distances."""
    tr = pd.DataFrame({
        "lead_time_days": [5, 5, 5, 5, 5],
        "spread_mean": [1.0, 1.1, 1.2, 9.0, 9.5],
        "y_bust": [0, 0, 1, 1, 1],
    })
    # query at spread=1.0: 3 nearest by |distance| are 1.0(0), 1.1(0), 1.2(1) -> 1/3 bust
    m = AnalogBaseline(n_neighbors=3).fit(tr)
    p = m.predict_proba(pd.DataFrame({"lead_time_days": [5], "spread_mean": [1.0]}))
    assert p[0] == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_standardisation_changes_which_neighbour_is_nearest():
    """lead_time_days spans single digits, spread_mean here spans hundreds - without
    scaling, distance is decided almost entirely by spread. Verified directly (not
    asserted from theory): an *unscaled* 1-NN on this exact training set picks the
    y=1 point as nearest to the query; AnalogBaseline's standardised distance picks a
    y=0 point instead - a real, checked flip, not a hypothetical one."""
    from sklearn.neighbors import KNeighborsClassifier

    lead = np.array([1.0, 10.0, 3.0, 6.0])
    spread = np.array([1000.0, 2000.0, 1500.0, 1800.0])
    y = np.array([0, 1, 0, 1])
    tr = pd.DataFrame({"lead_time_days": lead, "spread_mean": spread, "y_bust": y})
    query = pd.DataFrame({"lead_time_days": [1.0], "spread_mean": [2000.0]})

    unscaled = KNeighborsClassifier(n_neighbors=1).fit(
        np.column_stack([lead, spread]), y)
    unscaled_label = unscaled.predict(np.array([[1.0, 2000.0]]))[0]
    assert unscaled_label == 1, "the construction itself must produce the flip to test it"

    m = AnalogBaseline(n_neighbors=1).fit(tr)
    p = m.predict_proba(query)
    assert p[0] < 0.5, "standardised distance must pick the opposite (y=0) neighbour"


def test_it_beats_climatology_when_neighbours_carry_signal():
    tr, te = _events(seed=1), _events(seed=2)
    m = AnalogBaseline().fit(tr)
    p = m.predict_proba(te)
    clim = np.full(len(te), tr["y_bust"].mean())
    assert brier(te["y_bust"], p) < brier(te["y_bust"], clim)


def test_probabilities_stay_in_range():
    m = AnalogBaseline().fit(_events())
    p = m.predict_proba(_events(seed=3))
    assert p.min() > 0.0 and p.max() < 1.0 and np.isfinite(p).all()


def test_neighbour_count_is_capped_at_available_training_rows():
    """The default n_neighbors=50 must not crash a small training set."""
    tr = pd.DataFrame({
        "lead_time_days": [1, 2, 3, 4, 5],
        "spread_mean": [0.5, 1.0, 1.5, 2.0, 2.5],
        "y_bust": [0, 1, 0, 1, 0],
    })
    m = AnalogBaseline().fit(tr)
    assert m._knn is not None
    assert np.isfinite(m.predict_proba(tr)).all()


def test_single_class_training_falls_back_to_the_base_rate():
    tr = pd.DataFrame({
        "lead_time_days": [1, 2, 3, 4],
        "spread_mean": [0.5, 1.0, 1.5, 2.0],
        "y_bust": [0, 0, 0, 0],
    })
    m = AnalogBaseline().fit(tr)
    assert m._knn is None
    p = m.predict_proba(tr)
    assert np.allclose(p, 0.0, atol=1e-5)


def test_it_is_distinct_from_lead_spread_season_baseline():
    tr, te = _events(seed=4), _events(seed=5)
    fitted = fit_all(tr)
    analog = fitted["analog"].predict_proba(te)
    logistic = fitted["lead+spread+season"].predict_proba(te)
    assert not np.allclose(analog, logistic, atol=1e-3)
