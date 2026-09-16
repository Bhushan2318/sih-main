"""D1/D2 of the verification package: block-bootstrap CIs and binormal Z-AUC.

Every metric function is checked against a value computed independently of the code
under test - either a closed-form hand computation (via ``math.erf``, never
``scipy.stats``, since that is what ``binormal_auc`` itself calls) or a literal
Mann-Whitney count done by hand. See ``app/ml/verification.py`` for the formulas and
citations.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.ml.verification import binormal_auc, block_bootstrap_ci, trapezoidal_auc


def _phi(x: float) -> float:
    """Standard normal CDF via the stdlib erf - independent of scipy.stats.norm,
    which is what binormal_auc uses internally. The cross-check."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# trapezoidal_auc
# ---------------------------------------------------------------------------

def test_trapezoidal_auc_matches_hand_counted_mann_whitney():
    """3 negatives, 3 positives, counted by hand: of the 9 (positive, negative) pairs,
    exactly 7 have the positive scored higher (2 are decided by 0.2 losing to the two
    negatives above it), 0 ties. AUC = 7/9."""
    y = [0, 0, 0, 1, 1, 1]
    p = [0.1, 0.4, 0.35, 0.2, 0.6, 0.9]
    assert trapezoidal_auc(y, p) == pytest.approx(7.0 / 9.0)


def test_trapezoidal_auc_perfect_separation_is_one():
    y = [0, 0, 0, 1, 1, 1]
    p = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert trapezoidal_auc(y, p) == pytest.approx(1.0)


def test_trapezoidal_auc_nan_on_single_class():
    assert math.isnan(trapezoidal_auc([0, 0, 0], [0.1, 0.5, 0.9]))


# ---------------------------------------------------------------------------
# binormal_auc
# ---------------------------------------------------------------------------

def test_binormal_auc_matches_independent_closed_form():
    """Construct probabilities whose probit transform is EXACTLY a known set of
    z-scores (no-bust: -2,-1,0,1,2; bust: 0,1,2,3,4), so the binormal model's inputs
    are known by construction rather than estimated. mu0=0, mu1=2, var0=var1=2.5
    (sample variance, ddof=1) are exact by hand; the expected AUC is
    Phi((mu1-mu0)/sqrt(var0+var1)) via the stdlib erf, not scipy."""
    neg_z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    pos_z = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert neg_z.mean() == 0.0 and pos_z.mean() == 2.0
    assert neg_z.var(ddof=1) == pytest.approx(2.5)
    assert pos_z.var(ddof=1) == pytest.approx(2.5)

    expected = _phi((pos_z.mean() - neg_z.mean()) / math.sqrt(2 * 2.5))

    neg_p = [_phi(z) for z in neg_z]
    pos_p = [_phi(z) for z in pos_z]
    y = [0] * len(neg_p) + [1] * len(pos_p)
    p = neg_p + pos_p

    got = binormal_auc(y, p)
    assert got == pytest.approx(expected, abs=1e-9)
    assert got == pytest.approx(0.814453315238651, abs=1e-9)


def test_binormal_auc_close_to_trapezoidal_when_truly_binormal_and_well_sampled():
    """Consistency check, not a metric fixture: when the generating process really is
    two normals with a large, balanced sample, the two estimators should agree closely
    - this is the regime the binormal model is exact for. Random arrays are plumbing
    only; no score from this test is reported anywhere."""
    rng = np.random.default_rng(7)
    neg = rng.normal(0.35, 0.12, 2000)
    pos = rng.normal(0.65, 0.12, 2000)
    p = np.clip(np.concatenate([neg, pos]), 1e-6, 1 - 1e-6)
    y = np.array([0] * 2000 + [1] * 2000)
    assert abs(binormal_auc(y, p) - trapezoidal_auc(y, p)) < 0.03


def test_binormal_auc_nan_on_too_few_per_class():
    assert math.isnan(binormal_auc([0, 1], [0.1, 0.9]))


def test_binormal_auc_nan_on_single_class():
    assert math.isnan(binormal_auc([0, 0, 0], [0.1, 0.5, 0.9]))


# ---------------------------------------------------------------------------
# block_bootstrap_ci
# ---------------------------------------------------------------------------

def test_block_bootstrap_ci_resamples_whole_cycles_not_rows():
    """The D1 requirement, made exact with 2 cycles so the resampling distribution is
    hand-derivable. Cycle A is 30 rows all y=1, cycle B is 30 rows all y=0;
    metric_fn = bust rate. With replacement over {A, B} there are exactly 3 possible
    resample compositions: AA (bust rate 1.0, probability 1/4), AB or BA (0.5,
    probability 1/2), BB (0.0, probability 1/4). Sorted, the 0.0 outcome occupies the
    bottom 25% of the resampling distribution and 1.0 the top 25%, so the 2.5th/97.5th
    percentiles - comfortably inside each 25%-wide bucket at n=1000 draws - land
    exactly on 0.0 and 1.0."""
    y = np.array([1] * 30 + [0] * 30)
    p = np.array([0.9] * 30 + [0.1] * 30)  # irrelevant to this metric_fn
    cycles = np.array(["A"] * 30 + ["B"] * 30)

    result = block_bootstrap_ci(
        y, p, cycles, metric_fn=lambda yy, pp: float(np.mean(yy)),
        n_resamples=1000, seed=0,
    )
    assert result["point"] == pytest.approx(0.5)
    assert result["lo"] == pytest.approx(0.0)
    assert result["hi"] == pytest.approx(1.0)
    assert result["n_cycles"] == 2


def test_block_bootstrap_ci_is_wider_than_a_naive_row_bootstrap_on_the_same_data():
    """The critical claim from the brief, demonstrated rather than asserted: on data
    with only 2 independent cycles, a row-level bootstrap over the pooled 60 rows
    drastically understates the uncertainty, because it treats 30 perfectly-correlated
    rows as 30 independent observations. This reimplements the naive row bootstrap
    inline (deliberately not in app/ml/verification.py - offering it would invite
    someone to use it) purely to measure the gap."""
    y = np.array([1] * 30 + [0] * 30)
    p = np.array([0.9] * 30 + [0.1] * 30)
    cycles = np.array(["A"] * 30 + ["B"] * 30)
    metric = lambda yy, pp: float(np.mean(yy))  # noqa: E731

    by_cycle = block_bootstrap_ci(y, p, cycles, metric_fn=metric, n_resamples=1000, seed=0)

    rng = np.random.default_rng(0)
    n = len(y)
    naive_draws = np.array([
        metric(y[rng.integers(0, n, size=n)], p) for _ in range(1000)
    ])
    naive_lo, naive_hi = np.percentile(naive_draws, [2.5, 97.5])

    assert (by_cycle["hi"] - by_cycle["lo"]) > 3 * (naive_hi - naive_lo)


def test_block_bootstrap_ci_deterministic_given_seed():
    rng = np.random.default_rng(1)
    n_cycles = 12
    cycles = np.repeat(np.arange(n_cycles), 8)
    y = rng.integers(0, 2, size=len(cycles))
    p = rng.uniform(0, 1, size=len(cycles))

    r1 = block_bootstrap_ci(y, p, cycles, n_resamples=200, seed=42)
    r2 = block_bootstrap_ci(y, p, cycles, n_resamples=200, seed=42)
    assert r1 == r2


def test_block_bootstrap_ci_single_cycle_returns_degenerate_interval():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.7])
    cycles = np.array(["only"] * 4)

    result = block_bootstrap_ci(y, p, cycles, n_resamples=100, seed=0)
    assert result["n_resamples"] == 0
    assert result["lo"] == result["hi"] == result["point"]


def test_block_bootstrap_ci_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        block_bootstrap_ci([0, 1], [0.1, 0.9], ["a", "b", "c"])
