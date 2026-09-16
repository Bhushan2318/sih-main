"""Verification statistics for the baseline ladder.

Pure functions over already-scored ``(y_true, y_prob)`` arrays, plus a per-row cycle id
for block-bootstrap resampling. No trainer import, no pipeline dependency, no I/O -
callers own reading the eval-events frame and writing the result. Consumed by
``scripts/run_baselines.py``, which already has both the arrays and the ``init_date``
cycle id on the held-out test split.

Three things live here, per D1/D2/D3 of ``docs/team-brief-2026-09-15-updated.md``
Section 6:

- ``block_bootstrap_ci``: a confidence interval for any ladder metric, resampling whole
  forecast cycles rather than rows.
- ``binormal_auc`` / ``trapezoidal_auc``: the two AUC estimators side by side.
- ``corp_reliability_curve`` / ``brier_decomposition``: CORP (Consistent, Optimally
  binned, Reproducible - Dimitriadis, Gneiting & Jordan, PNAS 2021,
  doi:10.1073/pnas.2016191118) reliability diagrams via isotonic regression, and the
  matching exact Brier score decomposition.
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
