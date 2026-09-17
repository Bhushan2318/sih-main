"""Verification statistics for the baseline ladder.

Pure functions over already-scored ``(y_true, y_prob)`` arrays, plus a per-row cycle id
for block-bootstrap resampling. No trainer import, no pipeline dependency, no I/O -
callers own reading the eval-events frame and writing the result. Consumed by
``scripts/run_baselines.py``, which already has both the arrays and the ``init_date``
cycle id on the held-out test split.

Five things live here, per D1/D2/D3/D4/D6 of ``docs/team-brief-2026-09-15-updated.md``
Section 6:

- ``block_bootstrap_ci``: a confidence interval for any ladder metric, resampling whole
  forecast cycles rather than rows.
- ``binormal_auc`` / ``trapezoidal_auc``: the two AUC estimators side by side.
- ``corp_reliability_curve`` / ``brier_decomposition``: CORP (Consistent, Optimally
  binned, Reproducible - Dimitriadis, Gneiting & Jordan, PNAS 2021,
  doi:10.1073/pnas.2016191118) reliability diagrams via isotonic regression, and the
  matching exact Brier score decomposition.
- ``relative_economic_value`` / ``sedi``: the cost-loss decision-model value curve
  (the general model is Richardson, 2000, QJRMS, doi:10.1002/qj.49712656313; the same
  Shanker, Sarkar & Mamgain, 2024, QJRMS, doi:10.1002/qj.4674 already cited for D2's
  binormal Z-AUC also covers relative economic value for NCMRWF's own ensemble system,
  and is the more directly relevant citation here) and the Symmetric Extremal
  Dependence Index for rare events (Ferro & Stephenson, 2011, Weather and Forecasting,
  doi:10.1175/WAF-D-10-05030.1).
- ``conformal_threshold`` / ``conformal_prediction_set``: split conformal prediction
  (Vovk, Gammerman & Shafer, 2005; Angelopoulos & Bates, 2023, Foundations and Trends
  in Machine Learning 16(4):494-591, arXiv:2107.07511) - a distribution-free, exact
  marginal coverage guarantee on top of the classifier's own probability, with no
  assumption on the model or data distribution beyond exchangeability of the
  calibration and test points.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def trapezoidal_auc(y_true, y_prob) -> float:
    """The empirical / trapezoidal ROC-AUC - equivalent to the Mann-Whitney U statistic.

    Exposed here (rather than only inline at the call site) so it sits next to
    ``binormal_auc`` as the same shape of function, for direct comparison on one rung.
    """
    y = np.asarray(y_true, int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(y_prob, float)))


def binormal_auc(y_true, y_prob) -> float:
    """Binormal / Z-transform ROC-AUC estimator.

    Shanker, Sarkar & Mamgain (NCMRWF, QJRMS 2024), doi:10.1002/qj.4674: the
    trapezoidal/empirical AUC is a step function of the ranks of a small number of
    positive cases, so for rare extreme events scored by a small ensemble it
    underestimates skill relative to a binormal estimate - it cannot register *how far*
    apart the two classes' scores are once they stop overlapping, only that they do.

    The binormal model (Dorfman & Alf, 1969) assumes each class's forecast probability is
    normal after a probit transform. Each probability ``p`` is mapped to a Z-score via
    ``Phi^-1(p)``; the bust and no-bust classes' Z-scores are each summarised by a mean and
    variance, and

        AUC_binormal = Phi( (mu_bust - mu_no_bust) / sqrt(var_bust + var_no_bust) )

    which is exact when the transformed scores really are normal, and degrades gracefully
    (via the same formula) when they are only approximately so - that approximation is the
    entire empirical claim of the cited paper.

    Returns ``nan`` when either class has fewer than 2 members (no variance to estimate) or
    is degenerate (zero variance after clipping), the same convention as
    ``trapezoidal_auc``'s ``nan`` on a single-class input.
    """
    y = np.asarray(y_true, int)
    p = np.clip(np.asarray(y_prob, float), 1e-6, 1.0 - 1e-6)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")

    z_pos, z_neg = stats.norm.ppf(pos), stats.norm.ppf(neg)
    var_pos, var_neg = z_pos.var(ddof=1), z_neg.var(ddof=1)
    denom = np.sqrt(var_pos + var_neg)
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float(stats.norm.cdf((z_pos.mean() - z_neg.mean()) / denom))


def block_bootstrap_ci(
    y_true,
    y_prob,
    cycle_ids,
    metric_fn: MetricFn = trapezoidal_auc,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Confidence interval for ``metric_fn``, resampling whole forecast cycles.

    CRITICAL, and the entire reason this function exists rather than a one-line call to an
    off-the-shelf row bootstrap: resample BY CYCLE (``cycle_ids``, e.g. ``init_date``),
    never by row. Rows from the same forecast cycle share the same synoptic situation and
    the same ensemble - they are not independent draws of the metric. Resampling
    individual rows treats N correlated rows as N independent ones and produces an
    interval far narrower than the real sampling uncertainty; resampling whole cycles
    (with replacement, same number of cycles per draw as observed) respects the actual
    unit of independence in this data.

    ``metric_fn`` takes ``(y_true, y_prob)`` for a set of rows and returns a float; the
    default is ``trapezoidal_auc``, but any ladder metric (Brier, F1, ...) works the same
    way, which is what "on every ladder rung" in D1 means in practice - one function, any
    metric, called once per rung.
    """
    y = np.asarray(y_true)
    p = np.asarray(y_prob, float)
    c = np.asarray(cycle_ids)
    if not (len(y) == len(p) == len(c)):
        raise ValueError("y_true, y_prob, cycle_ids must be the same length")

    point = metric_fn(y, p)
    unique_cycles = np.unique(c)
    n_cycles = len(unique_cycles)
    if n_cycles < 2:
        return {
            "point": point, "lo": point, "hi": point,
            "n_resamples": 0, "n_cycles": int(n_cycles), "ci": ci,
        }

    rows_by_cycle = {cyc: np.flatnonzero(c == cyc) for cyc in unique_cycles}
    rng = np.random.default_rng(seed)

    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        chosen = rng.choice(unique_cycles, size=n_cycles, replace=True)
        idx = np.concatenate([rows_by_cycle[cyc] for cyc in chosen])
        draws[i] = metric_fn(y[idx], p[idx])

    draws = draws[np.isfinite(draws)]
    alpha = (1.0 - ci) / 2.0
    if draws.size == 0:
        lo = hi = float("nan")
    else:
        lo = float(np.percentile(draws, 100 * alpha))
        hi = float(np.percentile(draws, 100 * (1.0 - alpha)))
    return {
        "point": point, "lo": lo, "hi": hi,
        "n_resamples": int(draws.size), "n_cycles": int(n_cycles), "ci": ci,
    }


def _pav_fit(y_true, y_prob):
    """Sort by forecast, run isotonic regression (PAV) of outcome on forecast.

    Returns ``(y_sorted, p_sorted, fitted)`` - ``fitted`` is the PAV-recalibrated
    "conditional event probability" (CEP) estimate, the same length as the inputs,
    constant within each pooled block. Shared by both public functions below so their
    numbers can never drift apart from using two different isotonic fits.
    """
    y = np.asarray(y_true, float)
    p = np.asarray(y_prob, float)
    if len(y) == 0:
        return y, p, np.empty(0, dtype=float)
    order = np.argsort(p, kind="stable")
    y_sorted, p_sorted = y[order], p[order]
    iso = IsotonicRegression(out_of_bounds="clip")
    fitted = iso.fit_transform(p_sorted, y_sorted)
    return y_sorted, p_sorted, fitted


def corp_reliability_curve(y_true, y_prob) -> list:
    """CORP reliability diagram: the PAV-recalibrated forecast, grouped into the blocks
    isotonic regression actually produced.

    Dimitriadis, Gneiting & Jordan (2021, PNAS), doi:10.1073/pnas.2016191118: naive
    fixed-width binning (as in ``classifier._reliability``) is not a consistent
    estimator of the true calibration curve - its shape depends on an arbitrary choice
    of bin count and edges. CORP instead fits the provably-optimal (least-squares)
    monotone recalibration via the pool-adjacent-violators algorithm, with no bin count
    to choose: points are pooled only where the data itself forces it, via
    ``sklearn.isotonic.IsotonicRegression``.

    Returns the same ``{predicted_mean, observed_rate, n}`` shape as the existing
    ``CalibrationBin`` type (see ``app/api/schemas.py``), so this can sit alongside the
    naive curve without a contract change - one row per PAV block, in ascending
    forecast order.
    """
    y_sorted, p_sorted, fitted = _pav_fit(y_true, y_prob)
    if len(y_sorted) == 0:
        return []
    bins, start = [], 0
    n = len(fitted)
    for i in range(1, n + 1):
        # rtol=0 deliberately: PAV block values are bit-identical (all points in a block
        # are literally assigned the same computed float), not merely numerically close,
        # so this is an exact-equality-up-to-float-noise check. The default rtol=1e-5
        # would instead treat any two blocks whose means differ by less than ~3e-6
        # (relative) as the same block - a real risk at this project's row counts, not
        # just a theoretical one.
        if i == n or not np.isclose(fitted[i], fitted[start], atol=1e-12, rtol=0):
            bins.append({
                "predicted_mean": float(p_sorted[start:i].mean()),
                "observed_rate": float(fitted[start]),
                "n": int(i - start),
            })
            start = i
    return bins


def brier_decomposition(y_true, y_prob) -> dict:
    """Exact CORP-consistent Brier score decomposition: BS = MCB - DSC + UNC.

    Dimitriadis, Gneiting & Jordan (2021, PNAS), doi:10.1073/pnas.2016191118. Given the
    PAV-recalibrated forecast ``f`` from ``_pav_fit``:

    - UNC (uncertainty) = ybar * (1 - ybar): the Brier score of the constant
      climatology forecast, independent of the forecast under evaluation.
    - DSC (discrimination) = UNC - BS(f, y): how much recalibration improves on
      climatology.
    - MCB (miscalibration) = BS(x, y) - BS(f, y): the mean score lost to the original
      forecast not already being PAV-calibrated.

    MCB and DSC are defined as score DIFFERENCES, not as an independent squared-error
    term against the original forecast - that is what makes ``BS = MCB - DSC + UNC``
    an exact identity rather than an approximation, and it is also why MCB is
    guaranteed non-negative: PAV recalibration is a least-squares projection, so it can
    only weakly improve the Brier score, never worsen it.
    """
    y = np.asarray(y_true, float)
    p = np.asarray(y_prob, float)
    if len(y) == 0:
        return {"brier": float("nan"), "uncertainty": float("nan"),
                "discrimination": float("nan"), "miscalibration": float("nan")}

    y_sorted, p_sorted, fitted = _pav_fit(y, p)
    ybar = float(y_sorted.mean())
    uncertainty = ybar * (1.0 - ybar)
    bs_x = float(np.mean((p_sorted - y_sorted) ** 2))
    bs_f = float(np.mean((fitted - y_sorted) ** 2))
    discrimination = uncertainty - bs_f
    miscalibration = bs_x - bs_f
    return {
        "brier": bs_x,
        "uncertainty": uncertainty,
        "discrimination": discrimination,
        "miscalibration": miscalibration,
    }


def _hit_false_alarm_rates(y_true: np.ndarray, pred: np.ndarray) -> tuple:
    """H (hit rate / POD) and F (false-alarm rate / POFD) for one binary forecast."""
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    hits = int(np.sum((pred == 1) & (y_true == 1)))
    false_alarms = int(np.sum((pred == 1) & (y_true == 0)))
    h = hits / n_pos if n_pos else 0.0
    f = false_alarms / n_neg if n_neg else 0.0
    return h, f


def sedi(y_true, y_prob, threshold: float = 0.5) -> float:
    """Symmetric Extremal Dependence Index (SEDI).

    Ferro & Stephenson (2011, Weather and Forecasting), doi:10.1175/WAF-D-10-05030.1:
    built for rare binary events, where scores like CSI/HSS degenerate toward trivial
    values as the event's base rate shrinks. Each variable's own bust threshold really
    is rare by construction - the 90th percentile of that variable's own error - but
    the label this is actually scored on, ``y_bust`` ("did ANY of ~8 variables bust"),
    is not: CLAUDE.md's own measured figure is ~43% (1 - 0.9**8 ~= 0.57 before
    dependence pulls it down), not the 10% the per-variable definition might suggest.
    SEDI is included anyway - it costs nothing to report correctly at any base rate -
    but it does not carry the strong rare-event case here that the per-variable
    framing implies, and that gap is stated rather than left for a reader to assume
    wrong. Given hit rate H and false-alarm rate F at one decision threshold (0.5, the
    same convention ``classifier._evaluate`` already uses for precision/recall/F1):

        SEDI = (ln F - ln H - ln(1-F) + ln(1-H)) / (ln F + ln H + ln(1-F) + ln(1-H))

    Bounded in [-1, 1]: 0 for no skill (H == F), 1 in the limit of a perfect forecast
    (H -> 1, F -> 0), negative when the forecast is worse than chance (H < F).
    """
    y = np.asarray(y_true, int)
    pred = (np.asarray(y_prob, float) >= threshold).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    h, f = _hit_false_alarm_rates(y, pred)
    eps = 1e-6
    h = min(max(h, eps), 1.0 - eps)
    f = min(max(f, eps), 1.0 - eps)
    ln_h, ln_f = np.log(h), np.log(f)
    ln_1h, ln_1f = np.log(1.0 - h), np.log(1.0 - f)
    denom = ln_f + ln_h + ln_1f + ln_1h
    if not np.isfinite(denom) or denom == 0:
        return float("nan")
    return float((ln_f - ln_h - ln_1f + ln_1h) / denom)


def _cost_loss_value(h, f, base_rate: float, alpha):
    """Cost-loss economic value, vectorised over ``h``/``f``/``alpha`` (broadcast
    together; ``base_rate`` is always a scalar) - not looped in Python, since this is
    evaluated over every candidate threshold times every cost-loss ratio, and a
    Python-level loop over both is too slow at this project's real held-out row counts
    (measured: even after removing the O(n^2) candidate-building cost below, a plain
    Python double loop over ~1e6 candidates x 99 ratios did not finish in 60s).

    Richardson (2000, QJRMS), doi:10.1002/qj.49712656313 - the same decision model
    Shanker, Sarkar & Mamgain (2024, QJRMS), doi:10.1002/qj.4674 apply to NCMRWF's own
    ensemble. A user who can pay a fixed cost C to protect against a loss L, with
    alpha = C/L, gets value

        V = (min(alpha, s) - (alpha*(s*H + (1-s)*F) + s*(1-H))) / (min(alpha, s) - alpha*s)

    relative to climatology (V=0, the better of "always protect"/"never protect") and a
    perfect forecast (V=1). Re-derived and checked directly against the raw expected-cost
    simulation before use here, not taken from memory of the formula alone - see
    test_verification.py.
    """
    h, f, alpha = np.asarray(h, float), np.asarray(f, float), np.asarray(alpha, float)
    min_as = np.minimum(alpha, base_rate)
    denom = min_as - alpha * base_rate
    numer = min_as - (alpha * (base_rate * h + (1 - base_rate) * f) + base_rate * (1 - h))
    with np.errstate(invalid="ignore", divide="ignore"):
        value = numer / denom
    return np.where(np.abs(denom) < 1e-12, np.nan, value)


def relative_economic_value(y_true, y_prob, cost_loss_ratios=None) -> list:
    """Relative economic value curve, per Richardson (2000) / Shanker, Sarkar & Mamgain
    (2024) - see ``_cost_loss_value`` for the formula and citations.

    For each cost-loss ratio alpha, sweeps every probability threshold actually present
    in ``y_prob`` (plus the two trivial "always protect" / "never protect" rules) and
    reports the BEST achievable value - a rational decision-maker picks whichever
    threshold suits their own alpha, not one fixed cutoff for everyone, and this is
    also what a frontend value-vs-alpha slider (F3) needs. Floored at 0 by
    construction: "always"/"never" always reproduce the climatology reference exactly
    at one end of the alpha range, so the max can never go below it.
    """
    y = np.asarray(y_true, int)
    p = np.asarray(y_prob, float)
    n = len(y)
    if cost_loss_ratios is None:
        cost_loss_ratios = np.linspace(0.01, 0.99, 99)
    alphas = np.asarray(cost_loss_ratios, float)

    base_rate = float(y.mean()) if n else float("nan")
    if n == 0 or not (0.0 < base_rate < 1.0):
        return [{"cost_loss_ratio": float(a), "value": float("nan")} for a in alphas]

    n_pos, n_neg = int(y.sum()), int(n - y.sum())

    # Every candidate "predict yes when p >= t" threshold, computed in one sort + one
    # cumulative-sum pass rather than re-scanning the whole array per threshold: sorting
    # by probability descending, the set of rows predicted "yes" at any threshold is
    # exactly some prefix of this order, so cumulative hit/false-alarm counts along it
    # give every achievable (H, F) pair in O(n log n) total, not O(n * distinct
    # thresholds) - the naive per-threshold rescan is quadratic at this project's real
    # held-out row counts (measured: ~2s at n=20,000, clearly super-linear, which
    # extrapolates to minutes-to-hours at the hundreds of thousands of rows a real
    # held-out split reaches - see test_verification.py's timing test).
    order = np.argsort(-p, kind="stable")
    p_sorted, y_sorted = p[order], y[order]
    cum_hits = np.cumsum(y_sorted)
    cum_false_alarms = np.cumsum(1 - y_sorted)

    # Only keep prefixes ending exactly at a tie boundary: splitting mid-tie would
    # produce an (H, F) pair no real ">= threshold" rule can actually achieve, since
    # points sharing one probability always get the same prediction.
    is_boundary = np.empty(n, dtype=bool)
    is_boundary[:-1] = p_sorted[:-1] != p_sorted[1:]
    is_boundary[-1] = True

    h_candidates = np.concatenate([[0.0], cum_hits[is_boundary] / n_pos])
    f_candidates = np.concatenate([[0.0], cum_false_alarms[is_boundary] / n_neg])
    # h_candidates[-1] == 1.0 and f_candidates[-1] == 1.0 already ("always protect" -
    # the full sorted prefix), so only "never protect" (the leading 0.0s) needs adding.

    # One (n_candidates, n_alphas) matrix via broadcasting, not a Python loop over
    # either axis: h/f_candidates as a column, alphas as a row.
    grid = _cost_loss_value(h_candidates[:, None], f_candidates[:, None],
                            base_rate, alphas[None, :])
    with np.errstate(invalid="ignore"):
        best = np.nanmax(np.where(np.isfinite(grid), grid, np.nan), axis=0)
    # nanmax raises a RuntimeWarning (already silenced above) and returns nan for a
    # column that is all-nan - exactly the fallback wanted, not an error.
    return [{"cost_loss_ratio": float(a), "value": float(v)} for a, v in zip(alphas, best)]


def conformal_threshold(y_calib, p_calib, alpha: float = 0.1) -> float:
    """Split conformal calibration threshold q-hat.

    Vovk, Gammerman & Shafer (2005); Angelopoulos & Bates (2023, Foundations and Trends
    in Machine Learning 16(4):494-591, arXiv:2107.07511 - formula and coverage bound
    verified directly against the source, not from memory). The nonconformity score for
    a calibration point is how far the model's predicted probability for that point's
    TRUE class was from certainty: ``1 - p`` if it busted, ``p`` if it did not. q-hat is
    the k-th smallest of these n calibration scores, with ``k = ceil((n+1)*(1-alpha))``
    (capped at n) - not an interpolated quantile, the literal order statistic the
    theorem is stated in terms of, since interpolation would not carry the same exact
    guarantee.

    This threshold, used with ``conformal_prediction_set``, gives an EXACT,
    distribution-free marginal coverage guarantee - no assumption on the model or the
    data distribution beyond exchangeability of the calibration and test points:

        1 - alpha <= P(Y_test in C(X_test)) <= 1 - alpha + 1/(n+1)

    Calibration set: the classifier's own held-out ``val`` split (see
    ``scripts/run_baselines.py``) - not ``train`` (used to fit the model) and not
    ``test`` (used to report every other metric on this ladder, which calibrating on
    would double-dip the same rows for two different jobs). One honest caveat: XGBoost's
    own early stopping already looks at ``val``'s loss to decide when to stop training,
    a mild exchangeability violation the conformal prediction literature commonly
    tolerates in practice, but a real one - stated here rather than left unmentioned,
    since carving out a dedicated fourth split is a pipeline change outside a pure
    function's scope.
    """
    y = np.asarray(y_calib, int)
    p = np.asarray(p_calib, float)
    n = len(y)
    if n == 0:
        return float("nan")
    scores = np.where(y == 1, 1.0 - p, p)
    k = min(int(np.ceil((n + 1) * (1.0 - alpha))), n)
    return float(np.sort(scores)[k - 1])


def conformal_prediction_set(y_prob, q_hat: float) -> list:
    """Which labels survive the calibrated threshold from ``conformal_threshold``, for
    each predicted bust probability.

    Label 1 (bust) is included if its nonconformity score does not exceed q-hat:
    ``1 - p <= q_hat``, i.e. ``p >= 1 - q_hat``. Label 0 (no-bust) is included if
    ``p <= q_hat``. Both included means genuinely uncertain - the calibrated threshold
    cannot rule out either outcome. Neither included (both false) is a real,
    literature-documented phenomenon for points where the model's probability sits too
    close to 0.5 relative to q-hat for either label's score to clear the bar - stated
    plainly rather than something this function quietly avoids or reinterprets.

    Returns one ``{"no_bust": bool, "bust": bool}`` dict per row. A non-finite q-hat
    (an empty or degenerate calibration set - see ``conformal_threshold``) has nothing
    to exclude anything with, so every row gets both labels: maximally uncertain is the
    honest answer, not a guess.
    """
    p = np.asarray(y_prob, float)
    if not np.isfinite(q_hat):
        return [{"no_bust": True, "bust": True} for _ in range(len(p))]
    include_no_bust = p <= q_hat
    include_bust = (1.0 - p) <= q_hat
    return [{"no_bust": bool(a), "bust": bool(b)}
           for a, b in zip(include_no_bust, include_bust)]
