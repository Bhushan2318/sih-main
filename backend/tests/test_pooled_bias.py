"""The pooled retrain fits the bias on training cycles only and every read applies it.

PLUMBING FIXTURES: small hand-built cached years (the cache's real column names and
dtypes), for arithmetic only; never used to produce a metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import bias
from app.ml import pooled_training as pt


def _cached_year(path, year, fc=30.0, obs=28.0, region="D1", variable="temperature_c",
                 n_days=150, members=2):
    rows = []
    for d in pd.date_range(f"{year}-06-01", periods=n_days):
        for m in range(members):
            rows.append({"region_id": region, "variable": variable, "season": "JJAS",
                         "init_date": d, "valid_date": d, "lead_time_days": 1,
                         "ensemble_member_id": f"m{m}", "forecast_value": fc + 0.1 * m,
                         "observed_value": obs, "abs_error": abs(fc + 0.1 * m - obs),
                         "ensemble_spread": 0.07})
    df = pd.DataFrame(rows)
    for c in ("region_id", "variable", "season"):
        df[c] = df[c].astype("category")
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def two_years(tmp_path):
    a = _cached_year(tmp_path / "paired_2000.parquet", 2000, fc=30.0, obs=28.0)
    b = _cached_year(tmp_path / "paired_2001.parquet", 2001, fc=31.0, obs=28.0)
    return {2000: a, 2001: b}


def test_the_bias_is_fitted_on_training_cycles_only(two_years):
    train = set(pd.date_range("2000-06-01", periods=150))
    t = pt.pooled_bias_table(two_years, [2000, 2001], train)
    t = t[t["level"] == "lead_season"]
    # Only 2000 is training: mean(fc_mean - obs) = (30.05 - 28) = 2.05, never 2001's 3.05.
    assert t["bias"].iloc[0] == pytest.approx(2.05)
    assert t["n"].iloc[0] == 150


def test_streamed_table_equals_one_fit_over_every_training_event(two_years):
    train = (set(pd.date_range("2000-06-01", periods=150))
             | set(pd.date_range("2001-06-01", periods=150)))
    t = pt.pooled_bias_table(two_years, [2000, 2001], train)
    t = t[t["level"] == "lead_season"]
    assert t["bias"].iloc[0] == pytest.approx((2.05 * 150 + 3.05 * 150) / 300)


def test_thresholds_are_computed_on_corrected_errors(two_years):
    train = set(pd.date_range("2000-06-01", periods=150))
    t = pt.pooled_bias_table(two_years, [2000, 2001], train)
    _, p90, thr = pt.pooled_stats(two_years, [2000, 2001], train, bias_table=t)
    # Corrected member errors are |30.0 - 2.05 - 28| = 0.05 and |30.1 - 2.05 - 28| = 0.05.
    assert p90["temperature_c"] == pytest.approx(0.05)
    assert thr["temperature_c"] == pytest.approx(0.0, abs=1e-9)   # event mean is exact


def test_wind_direction_gets_no_bust_threshold(tmp_path):
    p = _cached_year(tmp_path / "paired_2000.parquet", 2000, variable="wind_direction_deg",
                     fc=10.0, obs=350.0)
    train = set(pd.date_range("2000-06-01", periods=150))
    t = pt.pooled_bias_table({2000: p}, [2000], train)
    _, _, thr = pt.pooled_stats({2000: p}, [2000], train, bias_table=t)
    assert "wind_direction_deg" not in thr


def test_regressor_rows_carry_the_corrected_target_and_forecast(two_years):
    train = set(pd.date_range("2000-06-01", periods=150))
    t = pt.pooled_bias_table(two_years, [2000, 2001], train)
    got = {}

    def capture(data, label):
        got["X"], got["y"] = data, label
    it = pt._YearDataIter(two_years, [2000], "temperature_c", train,
                          ["forecast_value", "bias_correction"], {}, None, bias_table=t)
    it.next(capture)
    assert got["y"] == pytest.approx(np.full(len(got["y"]), 0.05))
    assert got["X"]["forecast_value"].iloc[0] == pytest.approx(30.0 - 2.05)
    assert got["X"]["bias_correction"].iloc[0] == pytest.approx(2.05)


def test_without_a_table_the_iterator_is_unchanged(two_years):
    """Label version 1 runs (and every test written before the bias) read raw errors."""
    train = set(pd.date_range("2000-06-01", periods=150))
    got = {}
    it = pt._YearDataIter(two_years, [2000], "temperature_c", train, ["forecast_value"],
                          {}, None)
    it.next(lambda data, label: got.update(y=label))
    assert got["y"][0] == pytest.approx(2.0)
