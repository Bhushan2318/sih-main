"""D1/D2/D3/D4/D6 wiring: scripts/run_baselines.py must actually call into
app/ml/verification.py and put the results where /api/model/status already serves
baselines.json from - not just have the functions exist unused. See
app/ml/verification.py for the metric formulas themselves and their own hand-computed
tests.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.run_baselines import _fmt_ci, _metrics


def _fixture():
    """Each cycle has its own bust rate (a stand-in for one synoptic situation making a
    whole day's rows more or less bust-prone) - the real source of within-cycle
    correlation that D1 exists to respect. Without that per-cycle structure, resampling
    by cycle and by row would not actually differ, and the wiring test below would prove
    nothing.

    Also returns a separate calibration (y_calib, p_calib) pair, drawn the same way but
    from a disjoint seed - the stand-in for the `val` split D6's conformal calibration
    needs, distinct from the (y, proba) pair used for every other metric here."""
    rng = np.random.default_rng(3)
    n_cycles, rows_per_cycle = 10, 20
    cycle_prob = rng.uniform(0.15, 0.85, n_cycles)
    cycles = np.repeat(np.arange(n_cycles), rows_per_cycle)
    n = len(cycles)
    y = rng.binomial(1, cycle_prob[cycles])
    proba = np.clip(cycle_prob[cycles] + rng.normal(0, 0.05, n), 1e-6, 1 - 1e-6)
    ref = np.full(n, y.mean())

    r2 = np.random.default_rng(30)
    n_calib = 500
    y_calib = r2.binomial(1, 0.4, n_calib)
    p_calib = np.clip(r2.beta(2 + 3 * y_calib, 5 - 2 * y_calib), 1e-6, 1 - 1e-6)
    return y, proba, ref, cycles, y_calib, p_calib


def test_metrics_includes_block_bootstrap_ci_by_cycle():
    y, proba, ref, cycles, y_calib, p_calib = _fixture()
    m = _metrics(y, proba, ref, cycles, y_calib, p_calib)
    assert "roc_auc_ci" in m
    ci = m["roc_auc_ci"]
    assert ci["n_cycles"] == 10
    assert ci["lo"] <= m["roc_auc"] <= ci["hi"] or not np.isfinite(ci["lo"])


def test_metrics_includes_binormal_z_auc_alongside_trapezoidal():
    y, proba, ref, cycles, y_calib, p_calib = _fixture()
    m = _metrics(y, proba, ref, cycles, y_calib, p_calib)
    assert "z_auc" in m and "roc_auc" in m
    assert not math.isnan(m["z_auc"])


def test_metrics_ci_is_resampled_by_cycle_not_by_row():
    """If run_baselines started passing a row index instead of the cycle id, this catches
    it: resampling by row on this fixture (20 rows per cycle, strongly cycle-correlated y)
    collapses the CI far below what resampling by cycle gives."""
    y, proba, ref, cycles, y_calib, p_calib = _fixture()
    by_cycle = _metrics(y, proba, ref, cycles, y_calib, p_calib)["roc_auc_ci"]

    row_ids = np.arange(len(y))
    by_row = _metrics(y, proba, ref, row_ids, y_calib, p_calib)["roc_auc_ci"]

    assert (by_cycle["hi"] - by_cycle["lo"]) > (by_row["hi"] - by_row["lo"])


def test_fmt_ci_renders_bracketed_range():
    assert _fmt_ci({"lo": 0.5, "hi": 0.75}) == "[0.5000, 0.7500]"


def test_fmt_ci_dash_on_nan():
    assert _fmt_ci({"lo": float("nan"), "hi": float("nan")}) == "—"


def test_metrics_includes_corp_reliability_and_brier_decomposition():
    y, proba, ref, cycles, y_calib, p_calib = _fixture()
    m = _metrics(y, proba, ref, cycles, y_calib, p_calib)
    assert "corp_reliability" in m and "brier_decomposition" in m
    assert len(m["corp_reliability"]) >= 1
    bd = m["brier_decomposition"]
    assert bd["miscalibration"] - bd["discrimination"] + bd["uncertainty"] == \
        pytest.approx(bd["brier"], abs=1e-9)
    assert bd["miscalibration"] >= -1e-9


def test_metrics_includes_sedi_and_economic_value():
    y, proba, ref, cycles, y_calib, p_calib = _fixture()
    m = _metrics(y, proba, ref, cycles, y_calib, p_calib)
    assert "sedi" in m and "economic_value" in m
    assert -1.0 <= m["sedi"] <= 1.0
    assert len(m["economic_value"]) >= 1
    assert all(row["value"] >= -1e-9 for row in m["economic_value"])


def test_metrics_includes_conformal_fields_calibrated_on_the_val_split():
    """The defining wiring check: q_hat must come from (y_calib, p_calib) - a
    completely different array than (y, proba) - not accidentally recomputed from the
    test rows themselves, which would silently double-dip the same data for
    calibration and reporting."""
    y, proba, ref, cycles, y_calib, p_calib = _fixture()
    m = _metrics(y, proba, ref, cycles, y_calib, p_calib)
    assert {"conformal_alpha", "conformal_q_hat", "conformal_coverage",
           "conformal_mean_set_size"} <= m.keys()
    assert m["conformal_alpha"] == pytest.approx(0.1)
    assert np.isfinite(m["conformal_q_hat"])
    assert 1.0 <= m["conformal_mean_set_size"] <= 2.0

    from app.ml.verification import conformal_threshold
    expected_q_hat = conformal_threshold(y_calib, p_calib, alpha=0.1)
    assert m["conformal_q_hat"] == pytest.approx(expected_q_hat)


def test_metrics_conformal_coverage_is_measured_on_the_actual_test_rows():
    """Not the theoretical guarantee taken on faith: coverage here must equal what you
    get by independently checking, row by row, whether each test point's TRUE label
    survived calibration - computed here separately from _metrics' own internals."""
    from app.ml.verification import conformal_prediction_set, conformal_threshold

    y, proba, ref, cycles, y_calib, p_calib = _fixture()
    m = _metrics(y, proba, ref, cycles, y_calib, p_calib)

    q_hat = conformal_threshold(y_calib, p_calib, alpha=0.1)
    sets = conformal_prediction_set(proba, q_hat)
    covered = [s["bust"] if yy else s["no_bust"] for s, yy in zip(sets, y)]
    assert m["conformal_coverage"] == pytest.approx(np.mean(covered))


def test_metrics_conformal_fields_are_fully_uncertain_with_no_calibration_set():
    """An empty (y_calib, p_calib) - the val-split-missing case - must degrade to
    "fully uncertain" (mean set size 2.0), not crash and not silently report a
    misleadingly narrow coverage number computed from nothing."""
    y, proba, ref, cycles, _, _ = _fixture()
    m = _metrics(y, proba, ref, cycles, np.array([], dtype=int), np.array([]))
    assert math.isnan(m["conformal_q_hat"])
    assert m["conformal_mean_set_size"] == pytest.approx(2.0)
