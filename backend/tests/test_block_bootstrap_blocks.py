"""Moving-block bootstrap: resample runs of consecutive cycles, not single cycles.

Resampling whole cycles (block_bootstrap_ci's default) respects that rows within one
cycle are correlated. It does not respect that consecutive cycles are correlated too: a
daily cycle shares 9 of its 10 valid dates with the next, and on the served run's 2017
test set the per-cycle ROC-AUC has a lag-1 autocorrelation of 0.85 (0.67 at 5 days).
Single-cycle resampling there gave an interval 2.5-3.6x narrower than 10-30 day blocks.

The data here are synthetic and labelled as such: series built so the answer is known,
used only to check the resampling arithmetic; no result reaches a reported metric.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ml import verification as ver


def test_block_indices_are_contiguous_circular_runs():
    rng = np.random.default_rng(0)
    idx = ver._moving_block_cycle_indices(n_cycles=100, block_len=10, rng=rng)
    assert len(idx) == 100
    runs = idx.reshape(10, 10)
    for run in runs:
        steps = np.diff(run) % 100
        assert np.all(steps == 1), f"not a contiguous run: {run}"


def test_a_block_length_of_one_is_the_cycle_bootstrap():
    rng = np.random.default_rng(0)
    idx = ver._moving_block_cycle_indices(n_cycles=50, block_len=1, rng=rng)
    assert len(idx) == 50 and idx.min() >= 0 and idx.max() < 50


def _persistent_series(n_cycles=360, run=30, rows=5, seed=1):
    """Synthetic: the bust state persists for `run` consecutive cycles, as a synoptic
    regime does, so consecutive cycles are strongly dependent."""
    rng = np.random.default_rng(seed)
    state = np.repeat(rng.integers(0, 2, n_cycles // run), run)
    cycles = np.repeat(np.arange(n_cycles), rows)
    y = state[cycles]
    p = np.full(len(y), 0.5)
    return y, p, cycles


def test_blocks_widen_the_interval_when_consecutive_cycles_are_dependent():
    y, p, cycles = _persistent_series()
    rate = lambda yy, pp: float(np.mean(yy))  # noqa: E731
    single = ver.block_bootstrap_ci(y, p, cycles, metric_fn=rate, n_resamples=500, seed=0)
    blocks = ver.block_bootstrap_ci(y, p, cycles, metric_fn=rate, n_resamples=500, seed=0,
                                    block_len=30)
    assert (blocks["hi"] - blocks["lo"]) > 2 * (single["hi"] - single["lo"])


def test_the_result_records_the_block_length_used():
    y, p, cycles = _persistent_series()
    r = ver.block_bootstrap_ci(y, p, cycles, n_resamples=50, seed=0, block_len=30)
    assert r["block_len"] == 30
    r1 = ver.block_bootstrap_ci(y, p, cycles, n_resamples=50, seed=0)
    assert r1["block_len"] == 1


def test_a_short_series_caps_the_block_so_resamples_still_vary():
    """With 10 cycles a 30-cycle block would only ever rotate the same set and the
    interval would collapse to the point estimate. The block is capped at a fifth of the
    cycles, and the cap is reported."""
    rng = np.random.default_rng(3)
    cycles = np.repeat(np.arange(10), 20)
    y = rng.integers(0, 2, len(cycles))
    p = rng.uniform(0, 1, len(cycles))
    r = ver.block_bootstrap_ci(y, p, cycles, n_resamples=200, seed=0, block_len=30)
    assert r["block_len"] == 2
    assert r["hi"] > r["lo"]


def test_blocks_are_deterministic_given_a_seed():
    y, p, cycles = _persistent_series()
    a = ver.block_bootstrap_ci(y, p, cycles, n_resamples=100, seed=7, block_len=30)
    b = ver.block_bootstrap_ci(y, p, cycles, n_resamples=100, seed=7, block_len=30)
    assert a == b


def test_ladder_intervals_use_thirty_day_blocks():
    from scripts.run_baselines import CI_BLOCK_CYCLES
    assert CI_BLOCK_CYCLES == 30
