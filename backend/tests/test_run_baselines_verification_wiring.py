"""D1/D2/D3 wiring: scripts/run_baselines.py must actually call into
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
    nothing."""
    rng = np.random.default_rng(3)
    n_cycles, rows_per_cycle = 10, 20
    cycle_prob = rng.uniform(0.15, 0.85, n_cycles)
    cycles = np.repeat(np.arange(n_cycles), rows_per_cycle)
    n = len(cycles)
    y = rng.binomial(1, cycle_prob[cycles])
    proba = np.clip(cycle_prob[cycles] + rng.normal(0, 0.05, n), 1e-6, 1 - 1e-6)
    ref = np.full(n, y.mean())
    return y, proba, ref, cycles


def test_metrics_includes_block_bootstrap_ci_by_cycle():
    y, proba, ref, cycles = _fixture()
    m = _metrics(y, proba, ref, cycles)
    assert "roc_auc_ci" in m
    ci = m["roc_auc_ci"]
    assert ci["n_cycles"] == 10
    assert ci["lo"] <= m["roc_auc"] <= ci["hi"] or not np.isfinite(ci["lo"])


def test_metrics_includes_binormal_z_auc_alongside_trapezoidal():
    y, proba, ref, cycles = _fixture()
    m = _metrics(y, proba, ref, cycles)
    assert "z_auc" in m and "roc_auc" in m
    assert not math.isnan(m["z_auc"])


def test_metrics_ci_is_resampled_by_cycle_not_by_row():
    """If run_baselines started passing a row index instead of the cycle id, this catches
    it: resampling by row on this fixture (20 rows per cycle, strongly cycle-correlated y)
    collapses the CI far below what resampling by cycle gives."""
    y, proba, ref, cycles = _fixture()
    by_cycle = _metrics(y, proba, ref, cycles)["roc_auc_ci"]

    row_ids = np.arange(len(y))
    by_row = _metrics(y, proba, ref, row_ids)["roc_auc_ci"]

    assert (by_cycle["hi"] - by_cycle["lo"]) > (by_row["hi"] - by_row["lo"])


def test_fmt_ci_renders_bracketed_range():
    assert _fmt_ci({"lo": 0.5, "hi": 0.75}) == "[0.5000, 0.7500]"


def test_fmt_ci_dash_on_nan():
    assert _fmt_ci({"lo": float("nan"), "hi": float("nan")}) == "—"


def test_metrics_includes_corp_reliability_and_brier_decomposition():
    y, proba, ref, cycles = _fixture()
    m = _metrics(y, proba, ref, cycles)
    assert "corp_reliability" in m and "brier_decomposition" in m
    assert len(m["corp_reliability"]) >= 1
    bd = m["brier_decomposition"]
    assert bd["miscalibration"] - bd["discrimination"] + bd["uncertainty"] == \
        pytest.approx(bd["brier"], abs=1e-9)
    assert bd["miscalibration"] >= -1e-9
