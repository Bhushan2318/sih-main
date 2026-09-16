"""D1/D2/D3 of the verification package: block-bootstrap CIs, binormal Z-AUC, and the
CORP reliability curve / Brier decomposition.

Every metric function is checked against a value computed independently of the code
under test - either a closed-form hand computation (via ``math.erf``, never
``scipy.stats``, since that is what ``binormal_auc`` itself calls), a literal
Mann-Whitney count done by hand, or (for D3) a 4-point pool-adjacent-violators trace
worked by hand and cross-checked against ``sklearn.isotonic.IsotonicRegression``
directly rather than through the module under test. See ``app/ml/verification.py`` for
the formulas and citations.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from app.ml.verification import (
    binormal_auc,
    block_bootstrap_ci,
    brier_decomposition,
    corp_reliability_curve,
    trapezoidal_auc,
)


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


# ---------------------------------------------------------------------------
# corp_reliability_curve / brier_decomposition
#
# Hand-traced 4-point PAV example. y = [1, 0, 1, 1] in forecast order (x =
# [0.2, 0.4, 0.6, 0.8], already sorted, so sort order is a no-op here and doesn't
# obscure the trace):
#
#   start: blocks (1) (0) (1) (1)
#   (1) then (0): 1 > 0 violates non-decreasing -> pool -> (0.5, n=2) (1) (1)
#   (0.5) then (1): 0.5 <= 1, no violation
#   (1) then (1): no violation
#   final PAV fit: [0.5, 0.5, 1, 1]
#
# ybar = 0.75, UNC = 0.75*0.25 = 0.1875
# BS_f = mean((f-y)^2) = ((0.5-1)^2+(0.5-0)^2+(1-1)^2+(1-1)^2)/4 = 0.5/4 = 0.125
# DSC = UNC - BS_f = 0.0625
# BS_x = mean((x-y)^2) = (0.64+0.16+0.16+0.04)/4 = 0.25
# MCB = BS_x - BS_f = 0.125
# check: MCB - DSC + UNC = 0.125 - 0.0625 + 0.1875 = 0.25 = BS_x  (exact by construction)
# ---------------------------------------------------------------------------

_PAV_X = [0.2, 0.4, 0.6, 0.8]
_PAV_Y = [1, 0, 1, 1]


def test_corp_reliability_curve_matches_a_hand_traced_pav_example():
    bins = corp_reliability_curve(_PAV_Y, _PAV_X)
    assert len(bins) == 2
    assert bins[0]["n"] == 2 and bins[0]["observed_rate"] == pytest.approx(0.5)
    assert bins[0]["predicted_mean"] == pytest.approx(0.3)  # mean(0.2, 0.4)
    assert bins[1]["n"] == 2 and bins[1]["observed_rate"] == pytest.approx(1.0)
    assert bins[1]["predicted_mean"] == pytest.approx(0.7)  # mean(0.6, 0.8)


def test_corp_reliability_curve_agrees_with_sklearn_isotonic_directly():
    """Independent check: fit IsotonicRegression ourselves (not through the module
    under test) and confirm each bin's observed_rate matches the fitted level."""
    x = np.array(_PAV_X)
    y = np.array(_PAV_Y, dtype=float)
    fitted = IsotonicRegression(out_of_bounds="clip").fit_transform(x, y)
    bins = corp_reliability_curve(y, x)
    got = np.concatenate([[b["observed_rate"]] * b["n"] for b in bins])
    assert got == pytest.approx(fitted)


def test_corp_reliability_curve_does_not_merge_close_but_distinct_blocks():
    """Regression test for a real bug: the block-boundary comparison originally used
    np.isclose(..., atol=1e-12) without overriding rtol, which defaults to 1e-5 - so two
    genuinely distinct PAV blocks whose means differ by less than ~3e-6 (relative) were
    silently merged into one reported bin. At this project's real held-out row counts
    (hundreds of thousands to millions), PAV block means can land this close together,
    so this needs a realistic scale to actually exercise, not just a toy example: two
    million-row tied-forecast groups with means exactly 1e-6 apart."""
    n = 1_000_000
    p = np.concatenate([np.full(n, 0.3), np.full(n, 0.300001)])
    y = np.concatenate([
        np.concatenate([np.ones(300000), np.zeros(n - 300000)]),
        np.concatenate([np.ones(300001), np.zeros(n - 300001)]),
    ])
    bins = corp_reliability_curve(y, p)
    assert len(bins) == 2, "two genuinely distinct PAV blocks were merged into one"
    assert bins[0]["observed_rate"] == pytest.approx(0.3)
    assert bins[1]["observed_rate"] == pytest.approx(0.300001)


def test_brier_decomposition_matches_the_hand_traced_example():
    d = brier_decomposition(_PAV_Y, _PAV_X)
    assert d["brier"] == pytest.approx(0.25)
    assert d["uncertainty"] == pytest.approx(0.1875)
    assert d["discrimination"] == pytest.approx(0.0625)
    assert d["miscalibration"] == pytest.approx(0.125)


def test_brier_decomposition_identity_holds_exactly():
    """MCB - DSC + UNC == BS, by construction (MCB and DSC are defined as score
    differences, not independent terms) - checked on random data, not just the
    hand-traced example, since this is the property the whole decomposition rests on."""
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 200)
    y = rng.binomial(1, p)
    d = brier_decomposition(y, p)
    assert d["miscalibration"] - d["discrimination"] + d["uncertainty"] == \
        pytest.approx(d["brier"], abs=1e-9)


def test_brier_decomposition_miscalibration_is_never_negative():
    """PAV recalibration is a least-squares projection onto the monotone cone, so it
    can only weakly improve the Brier score - MCB = BS(x) - BS(f) >= 0 always. Checked
    over many random forecasts, not asserted from theory alone."""
    rng = np.random.default_rng(4)
    for _ in range(20):
        n = rng.integers(10, 100)
        p = rng.uniform(0, 1, n)
        y = rng.binomial(1, rng.uniform(0, 1, n))  # forecast deliberately uncorrelated with y
        d = brier_decomposition(y, p)
        assert d["miscalibration"] >= -1e-9


def test_brier_decomposition_miscalibration_is_zero_when_already_calibrated():
    """If the forecast already equals its own PAV fit, recalibrating changes nothing -
    MCB must be exactly 0, not just close to 0."""
    rng = np.random.default_rng(5)
    x = np.sort(rng.uniform(0, 1, 30))
    y = rng.binomial(1, x)
    fitted = IsotonicRegression(out_of_bounds="clip").fit_transform(x, y.astype(float))
    d = brier_decomposition(y, fitted)  # use the PAV fit itself as "the forecast"
    assert d["miscalibration"] == pytest.approx(0.0, abs=1e-9)


def test_corp_reliability_curve_empty_input():
    assert corp_reliability_curve([], []) == []


def test_brier_decomposition_nan_on_empty_input():
    d = brier_decomposition([], [])
    assert all(math.isnan(v) for v in d.values())
