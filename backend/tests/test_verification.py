"""D1/D2/D3/D4/D6 of the verification package: block-bootstrap CIs, binormal Z-AUC, the
CORP reliability curve / Brier decomposition, relative economic value / SEDI, and split
conformal prediction.

Every metric function is checked against a value computed independently of the code
under test - either a closed-form hand computation (via ``math.erf``, never
``scipy.stats``, since that is what ``binormal_auc`` itself calls), a literal
Mann-Whitney count done by hand, a 4-point pool-adjacent-violators trace worked by hand
and cross-checked against ``sklearn.isotonic.IsotonicRegression`` directly, a raw
expected-cost simulation of the cost-loss decision model independent of D4's closed-form
formula, or (for D6) the literature's own central empirical claim - marginal coverage -
checked directly rather than assumed from the formula alone. See
``app/ml/verification.py`` for the formulas and citations.
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
    conformal_prediction_set,
    conformal_threshold,
    corp_reliability_curve,
    relative_economic_value,
    sedi,
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


# ---------------------------------------------------------------------------
# sedi
# ---------------------------------------------------------------------------

def test_sedi_matches_a_hand_computed_confusion_matrix():
    """8 hits, 2 misses, 2 false alarms, 88 correct rejections (n=100). H=0.8,
    F=2/90=0.02222. SEDI = (lnF-lnH-ln(1-F)+ln(1-H)) / (lnF+lnH+ln(1-F)+ln(1-H)),
    computed here with math.log directly, not through the module."""
    y = np.array([1] * 10 + [0] * 90)
    p = np.concatenate([
        np.array([0.9] * 8 + [0.1] * 2),      # positives: 8 predicted yes, 2 no
        np.array([0.9] * 2 + [0.1] * 88),     # negatives: 2 predicted yes, 88 no
    ])
    H, F = 8 / 10, 2 / 90
    ln = math.log
    expected = (ln(F) - ln(H) - ln(1 - F) + ln(1 - H)) / (ln(F) + ln(H) + ln(1 - F) + ln(1 - H))
    assert sedi(y, p, threshold=0.5) == pytest.approx(expected)
    assert expected == pytest.approx(0.9132360676324364)


def test_sedi_is_zero_at_no_skill():
    """H == F (the forecast carries no information beyond the base rate) gives SEDI 0
    exactly, by construction of the formula (numerator vanishes when H=F)."""
    y = np.array([1] * 30 + [0] * 70)
    # predict "yes" for the same fraction (30%) of positives and negatives alike
    p = np.concatenate([[0.9] * 9 + [0.1] * 21, [0.9] * 21 + [0.1] * 49])
    assert sedi(y, p, threshold=0.5) == pytest.approx(0.0, abs=1e-9)


def test_sedi_is_negative_when_worse_than_chance():
    y = np.array([1] * 10 + [0] * 90)
    # forecast "yes" more often for negatives than positives - anti-skillful
    p = np.concatenate([[0.9] * 2 + [0.1] * 8, [0.9] * 60 + [0.1] * 30])
    assert sedi(y, p, threshold=0.5) < 0


def test_sedi_nan_on_single_class():
    assert math.isnan(sedi([0, 0, 0], [0.1, 0.5, 0.9]))


# ---------------------------------------------------------------------------
# relative_economic_value
# ---------------------------------------------------------------------------

def _raw_cost_loss_value(h, f, s, alpha):
    """Independent re-derivation, not the module's closed-form formula: simulate the
    four expense terms directly (protect at cost C on a hit or false alarm, pay loss L
    on a miss, nothing on a correct rejection) and compare to climatology/perfect."""
    C, L = alpha, 1.0
    e_forecast = C * (s * h + (1 - s) * f) + L * s * (1 - h)
    e_ref = min(C, s * L)
    e_perfect = C * s
    return (e_ref - e_forecast) / (e_ref - e_perfect)


def test_relative_economic_value_is_one_for_a_perfect_forecast():
    y = np.array([1] * 20 + [0] * 80)
    p = np.array([0.9] * 20 + [0.1] * 80)  # perfectly separated
    curve = relative_economic_value(y, p, cost_loss_ratios=[0.1, 0.3, 0.5, 0.7, 0.9])
    for row in curve:
        assert row["value"] == pytest.approx(1.0, abs=1e-9)


def test_relative_economic_value_matches_independent_expense_simulation():
    """2 distinct forecast values, so the only non-trivial threshold's (H, F) can be
    worked out by hand: 3 of 10 events, forecast 'yes' for 2 of the 3 events and 1 of
    the 7 non-events at p=0.9 (H=2/3, F=1/7); the rest predict 'no'. Checked against
    test_raw_cost_loss_value, an independent expense simulation, not the module's own
    closed-form formula."""
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    base_rate = 0.3
    h, f = 2 / 3, 1 / 7

    curve = relative_economic_value(y, p, cost_loss_ratios=[0.2])
    got = curve[0]["value"]

    candidates = [0.0, 1.0, _raw_cost_loss_value(h, f, base_rate, 0.2)]
    # 0.0 and 1.0 stand for the "never"/"always protect" trivial rules' own values,
    # each exactly 0 at one end of the alpha range - included for completeness, but at
    # alpha=0.2 the real threshold should win.
    never_val = _raw_cost_loss_value(0.0, 0.0, base_rate, 0.2)
    always_val = _raw_cost_loss_value(1.0, 1.0, base_rate, 0.2)
    expected = max(never_val, always_val, _raw_cost_loss_value(h, f, base_rate, 0.2))
    assert got == pytest.approx(expected)


def test_relative_economic_value_is_never_negative():
    """"Always"/"never protect" are always available as trivial fallback rules and each
    exactly reproduces the climatology reference (value 0) at one end of the alpha
    range, so the best-over-thresholds curve can never dip below 0 - checked on random,
    deliberately unskilled data, not just the well-behaved fixtures above."""
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.35, 300)
    p = rng.uniform(0, 1, 300)  # pure noise, uncorrelated with y
    curve = relative_economic_value(y, p)
    assert all(row["value"] >= -1e-9 for row in curve)


def test_relative_economic_value_nan_on_degenerate_base_rate():
    curve = relative_economic_value([0, 0, 0], [0.1, 0.5, 0.9], cost_loss_ratios=[0.5])
    assert math.isnan(curve[0]["value"])


def test_relative_economic_value_stays_fast_at_realistic_scale():
    """Regression test for a real performance bug: the first version rebuilt each
    candidate threshold's hit/false-alarm counts with a fresh full-array scan
    (O(n * distinct thresholds), ~quadratic with mostly-unique continuous forecasts -
    measured 2.1s at n=20,000, clearly super-linear). A second version fixed the
    candidate-building cost but still looped candidates x cost-loss ratios in pure
    Python, which did not finish in 60s at n=1,000,000 x 99 ratios. The final version
    is a sort + cumulative sum plus one broadcast matrix, no Python-level loop over
    rows or ratios - this must stay well under a minute at real held-out row counts, on
    the CI runner this is timed on."""
    import time
    rng = np.random.default_rng(9)
    n = 300_000
    y = rng.binomial(1, 0.3, n)
    p = rng.uniform(0, 1, n)
    start = time.time()
    curve = relative_economic_value(y, p, cost_loss_ratios=np.linspace(0.01, 0.99, 99))
    elapsed = time.time() - start
    assert elapsed < 15.0, f"took {elapsed:.1f}s - the O(n^2)/Python-loop bug is back"
    assert len(curve) == 99


# ---------------------------------------------------------------------------
# conformal_threshold / conformal_prediction_set
# ---------------------------------------------------------------------------

def test_conformal_threshold_matches_a_hand_computed_order_statistic():
    """n=19, alpha=0.1 -> k=ceil(20*0.9)=18. Scores are exactly 0.05, 0.10, ..., 0.95
    (all y=0, so score=p directly) - the 18th smallest is 0.90, read straight off the
    list, not computed by any quantile-interpolation machinery."""
    p = np.array([0.05 * i for i in range(1, 20)])
    y = np.zeros(19, dtype=int)
    assert conformal_threshold(y, p, alpha=0.1) == pytest.approx(0.90)


def test_conformal_threshold_at_the_k_equals_n_boundary():
    """n=9, alpha=0.1 -> k=ceil(10*0.9)=9=n exactly: q_hat must be the single largest
    score, the boundary case where the correction saturates at the whole sample."""
    p = np.array([0.05 * i for i in range(1, 10)])
    y = np.zeros(9, dtype=int)
    assert conformal_threshold(y, p, alpha=0.1) == pytest.approx(0.45)


def test_conformal_prediction_set_matches_hand_computed_membership():
    """q_hat=0.90 (from the n=19 example above): a confident no-bust point (p=0.05)
    gets only {no_bust}; a confident bust point (p=0.95) gets only {bust}; a middling
    point (p=0.5) gets both - genuinely uncertain under this generous threshold."""
    q_hat = 0.90
    sets = conformal_prediction_set(np.array([0.05, 0.95, 0.5]), q_hat)
    assert sets[0] == {"no_bust": True, "bust": False}
    assert sets[1] == {"no_bust": False, "bust": True}
    assert sets[2] == {"no_bust": True, "bust": True}


def test_conformal_prediction_set_can_legitimately_be_empty():
    """n=9, alpha=0.5 -> k=ceil(10*0.5)=5 -> q_hat=0.25 (5th of 0.05..0.45). At p=0.5,
    neither label's nonconformity score (0.5 for both) clears 0.25 - a real, literature-
    documented phenomenon, not a bug to paper over."""
    p_calib = np.array([0.05 * i for i in range(1, 10)])
    y_calib = np.zeros(9, dtype=int)
    q_hat = conformal_threshold(y_calib, p_calib, alpha=0.5)
    assert q_hat == pytest.approx(0.25)
    result = conformal_prediction_set(np.array([0.5]), q_hat)
    assert result == [{"no_bust": False, "bust": False}]


def test_conformal_threshold_nan_on_empty_calibration_set():
    assert math.isnan(conformal_threshold([], [], alpha=0.1))


def test_conformal_prediction_set_is_fully_uncertain_when_q_hat_is_nan():
    sets = conformal_prediction_set(np.array([0.1, 0.5, 0.9]), float("nan"))
    assert all(s == {"no_bust": True, "bust": True} for s in sets)


def test_conformal_prediction_achieves_marginal_coverage():
    """The literature's central empirical claim, checked directly rather than trusted
    from the formula: averaged over many independent calibration draws, the fraction of
    test points whose TRUE label is included in their own prediction set should be
    >= 1 - alpha. This is a statement about the AVERAGE over calibration draws, not a
    per-trial guarantee (q_hat is itself random) - individual trials scatter around the
    target, which is why this asserts the mean across 200 trials, not every trial."""
    alpha = 0.1
    n_calib, n_test = 2000, 3000
    coverages = []
    for trial in range(200):
        r = np.random.default_rng(trial)
        y_calib = r.binomial(1, 0.4, n_calib)
        p_calib = np.clip(r.beta(2 + 3 * y_calib, 5 - 2 * y_calib), 1e-6, 1 - 1e-6)
        q_hat = conformal_threshold(y_calib, p_calib, alpha)

        y_test = r.binomial(1, 0.4, n_test)
        p_test = np.clip(r.beta(2 + 3 * y_test, 5 - 2 * y_test), 1e-6, 1 - 1e-6)
        sets = conformal_prediction_set(p_test, q_hat)
        covered = [s["bust"] if y else s["no_bust"] for s, y in zip(sets, y_test)]
        coverages.append(np.mean(covered))

    mean_coverage = float(np.mean(coverages))
    assert mean_coverage == pytest.approx(1 - alpha, abs=0.01), \
        f"mean coverage {mean_coverage:.4f} across 200 trials, target {1 - alpha}"
