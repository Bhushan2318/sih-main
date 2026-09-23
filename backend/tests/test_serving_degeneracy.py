"""The gate's blind spot: a model with good held-out metrics that serves one number.

run_20260910T064804Z passed every check the promotion gate makes. Its held-out ROC-AUC
was 0.8411, its Brier 0.1656, and its held-out probabilities were spread evenly across
the five calibration bins (11.3 / 23.2 / 16.5 / 20.9 / 28.1 per cent). It was promoted,
and it served 642 of 666 districts in the bust band at a median probability of 0.977.

Both facts hold at once because ROC-AUC is rank-based: invariant under any monotone
transform of the scores, so crushing every probability toward 1 leaves it untouched. The
held-out histogram missed it too, because the held-out rows were the 34 districts that
had observations when the model was trained, scored against the observation set that
existed then.

The fixtures below are shaped from real measurements over the 2018-12-31 cycle, all ten
lead days, 6,660 events - not invented. Per CLAUDE.md, a green suite here is a statement
about plumbing; the real evidence is the same check run against the real runs and the
real store, which is recorded in the module docstring.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.serving_sanity import (
    MAX_BUST_BAND_SHARE, MAX_RAIL_SHARE, MIN_SERVED_IQR, degeneracy_verdict,
)

RNG = np.random.default_rng(20260923)


def _bands(p, medium=0.45, high=0.82):
    """Band each probability the way thresholds.band_for does, for fixture realism."""
    return np.array(["high" if x >= high else "medium" if x >= medium else "low"
                     for x in p], dtype=object)


# run_20260910T064804Z: median 0.977, IQR 0.054, 46.2% above 0.99, high band 93.8%.
BROKEN = np.concatenate([RNG.uniform(0.94, 1.0, 6247), RNG.uniform(0.10, 0.60, 413)])

# run_20260922T043925Z, the 17-year run that fixed it: median 0.187, IQR 0.119, 0% above
# 0.99, low band 90.4%, high band 1.8%. Concentrated - but against the other rail.
GOOD = np.concatenate([RNG.uniform(0.02, 0.45, 6020), RNG.uniform(0.45, 0.82, 520),
                       RNG.uniform(0.82, 0.98, 120)])


def test_refuses_the_distribution_that_shipped():
    ok, reason = degeneracy_verdict(BROKEN, _bands(BROKEN))
    assert ok is False
    assert "high" in reason and "band" in reason


def test_passes_the_run_that_fixed_it():
    """The regression that matters most: the first version of this check refused GOOD.

    It floored the interquartile range at 0.15, and the working model serves 0.119 -
    narrower than the broken one might suggest, because a quiet day genuinely is quiet.
    A guard that refuses the fix is worse than no guard.
    """
    ok, reason = degeneracy_verdict(GOOD, _bands(GOOD))
    assert ok is True, reason


def test_concentration_alone_is_not_the_signal():
    """GOOD is 90.4% in one band and must still pass; BROKEN is 93.8% and must not.

    Any rule phrased as "too much of the map in a single band" fails this pair, which is
    why the check is asymmetric about which band.
    """
    good_share = float(np.mean(_bands(GOOD) == "low"))
    broken_share = float(np.mean(_bands(BROKEN) == "high"))
    assert good_share > MAX_BUST_BAND_SHARE
    assert broken_share > MAX_BUST_BAND_SHARE
    assert degeneracy_verdict(GOOD, _bands(GOOD))[0] is True
    assert degeneracy_verdict(BROKEN, _bands(BROKEN))[0] is False


def test_catches_saturation_without_bands():
    """Bands are optional; saturation against a rail is visible before banding."""
    ok, reason = degeneracy_verdict(BROKEN)
    assert ok is False
    assert "pinned" in reason


def test_catches_saturation_against_zero_too():
    """A model that says nothing ever busts is as useless as one that says all does."""
    dead = np.concatenate([np.full(6000, 0.0005), RNG.uniform(0.2, 0.6, 660)])
    ok, reason = degeneracy_verdict(dead)
    assert ok is False
    assert "0.0" in reason


def test_refuses_a_single_repeated_value():
    ok, reason = degeneracy_verdict(np.full(666, 0.5))
    assert ok is False
    assert "constant" in reason


def test_refuses_too_few_scored_districts_rather_than_guessing():
    """Silence is not a pass. Too small a sample cannot show degeneracy either way."""
    ok, reason = degeneracy_verdict(np.array([0.1, 0.9]))
    assert ok is False
    assert "too few" in reason.lower()


def test_ignores_nan_but_refuses_when_nothing_is_left():
    ok, _ = degeneracy_verdict(np.concatenate([GOOD, np.full(50, np.nan)]))
    assert ok is True
    ok, reason = degeneracy_verdict(np.full(666, np.nan))
    assert ok is False
    assert "no scored" in reason.lower()


def test_nan_filtering_keeps_probabilities_and_bands_aligned():
    """A NaN dropped from one array and not the other would silently misband everything."""
    p = np.concatenate([BROKEN[:100], np.full(20, np.nan), BROKEN[100:]])
    b = _bands(np.nan_to_num(p, nan=0.99))
    ok, _ = degeneracy_verdict(p, b)
    assert ok is False


@pytest.mark.parametrize("share", [MAX_RAIL_SHARE + 0.05, 0.5])
def test_rail_ceiling_is_a_ceiling(share):
    n = 4000
    k = int(n * share)
    v = np.concatenate([np.full(k, 0.995), RNG.uniform(0.1, 0.8, n - k)])
    assert degeneracy_verdict(v)[0] is False


def test_iqr_floor_only_catches_a_flat_line():
    """The floor must sit well below what a real quiet day serves."""
    assert MIN_SERVED_IQR < 0.119


def test_a_model_broken_before_training_is_not_this_check_s_job():
    """The documented blind spot, asserted so nobody assumes more than this catches.

    run_20260922T100055Z has a temperature regressor predicting absolute errors from
    -2218.9 to +438.99 where the true range is 0 to 2.69, held-out r2 -682,626. Scored on a
    real cycle it produced median 0.279 / IQR 0.401 / bust band 13.1%, against the good
    model's 0.287 / 0.407 / 13.4% - measured 2026-09-23, 6,660 events each.

    It passes, and should: the classifier was trained on those broken values, so the model
    is internally consistent and its distribution is not degenerate. Catching it belongs at
    the regressor stage. If someone later widens this check until this test fails, they have
    turned a distribution check into something else, and should delete this test knowingly
    rather than discover it.
    """
    broken = np.concatenate([RNG.uniform(0.02, 0.16, 1665), RNG.uniform(0.16, 0.56, 3330),
                             RNG.uniform(0.56, 0.99, 1665)])
    ok, reason = degeneracy_verdict(broken, _bands(broken))
    assert ok is True, reason
