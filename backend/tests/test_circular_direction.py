"""Wind direction is an angle: 350 deg and 10 deg are 20 deg apart, not 340.

Feature version 2 does the arithmetic on the circle wherever direction is combined:
  - abs_error (the regressor target) is the shorter way round, never more than 180;
  - ensemble_spread is the circular standard deviation of the members;
  - the time-lagged ensemble pools unit vectors, not raw degrees.
Measured on Nov 2017 (93,240 events): 86.5% of the direction "busts" under plain
arithmetic were the 0/360 seam and the arithmetic mean, not a forecast failure.

Version 1 keeps the old arithmetic exactly, because runs trained with it - the served
run_20260922T043925Z among them - must keep receiving the inputs they were trained on.
A run with no `feature_version` in its manifest is version 1.

PLUMBING FIXTURES: hand-built forecast/observation rows for arithmetic only, never used
to produce a metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app import contracts
from app.features import engineering as fe

REGION = "IN-MH-PUNE"


def _canonical(directions, observed, variable="wind_direction_deg", init="2017-11-02",
               valid="2017-11-03", lead=2):
    fc = pd.DataFrame([{
        "region_id": REGION, "variable": variable, "value_type": "forecast",
        "valid_date": pd.Timestamp(valid), "init_date": pd.Timestamp(init),
        "lead_time_days": lead, "ensemble_member_id": f"m{i}", "value": float(d),
    } for i, d in enumerate(directions)])
    ob = pd.DataFrame([{
        "region_id": REGION, "variable": variable, "value_type": "observed",
        "valid_date": pd.Timestamp(valid), "value": float(observed),
    }])
    return pd.concat([fc, ob], ignore_index=True)


def _frame(directions, observed, version, **kw):
    return fe.build_training_frame(_canonical(directions, observed, **kw),
                                   feature_version=version)


def test_the_current_version_is_two_and_legacy_is_one():
    assert contracts.FEATURE_VERSION == 2
    assert contracts.LEGACY_FEATURE_VERSION == 1


# --------------------------------------------------------------- version 2: on the circle

def test_error_goes_the_short_way_round():
    f = _frame([355.0], observed=5.0, version=2)
    assert f["abs_error"].iloc[0] == pytest.approx(10.0)


def test_error_is_never_more_than_half_a_turn():
    f = _frame([90.0, 270.0, 10.0, 200.0, 359.0], observed=0.0, version=2)
    assert (f["abs_error"] <= 180.0 + 1e-9).all()


def test_spread_of_members_either_side_of_north_is_small():
    """350 and 10 are 20 degrees apart; a plain std calls them 226 degrees apart."""
    f = _frame([350.0, 10.0], observed=0.0, version=2)
    spread = f["ensemble_spread"].iloc[0]
    assert spread == pytest.approx(fe.circular_std_deg(np.array([350.0, 10.0])))
    assert spread < 20.0
    same = _frame([-10.0 % 360, 10.0], observed=0.0, version=2)["ensemble_spread"].iloc[0]
    assert spread == pytest.approx(same)


def test_circular_std_is_zero_for_agreeing_members_and_grows_with_scatter():
    assert fe.circular_std_deg(np.array([30.0, 30.0, 30.0])) == pytest.approx(0.0, abs=1e-6)
    a = fe.circular_std_deg(np.array([20.0, 40.0]))
    b = fe.circular_std_deg(np.array([0.0, 60.0]))
    assert 0 < a < b


def test_lagged_pool_mean_of_directions_either_side_of_north_points_north():
    f = fe.compute_time_lagged_ensemble(
        _canonical([350.0, 10.0], 0.0).query("value_type == 'forecast'"),
        _trajectories([(pd.Timestamp("2017-11-01"), 0.0)]), feature_version=2)
    m = f.loc[f["init_date"] == pd.Timestamp("2017-11-02"), "laf_pool_mean"].iloc[0]
    assert min(m, 360.0 - m) < 1.0, f"pool mean {m} should point north"


def _trajectories(prior):
    """An earlier cycle's ensemble mean for the same valid date, as
    forecast_trajectories produces it."""
    rows = [{"region_id": REGION, "variable": "wind_direction_deg",
             "valid_date": pd.Timestamp("2017-11-03"), "init_date": pd.Timestamp("2017-11-02"),
             "fc_mean": 0.0}]
    rows += [{"region_id": REGION, "variable": "wind_direction_deg",
              "valid_date": pd.Timestamp("2017-11-03"), "init_date": init, "fc_mean": m}
             for init, m in prior]
    return pd.DataFrame(rows)


# --------------------------------------------------------------- version 1: unchanged

def test_version_one_keeps_the_plain_arithmetic():
    f = _frame([350.0, 10.0], observed=5.0, version=1)
    assert f["ensemble_spread"].iloc[0] == pytest.approx(np.std([350.0, 10.0], ddof=1))
    assert sorted(f["abs_error"]) == pytest.approx([5.0, 345.0])


def test_other_variables_are_identical_in_both_versions():
    kw = dict(variable="temperature_c")
    one = _frame([30.0, 31.5, 29.0], observed=30.2, version=1, **kw)
    two = _frame([30.0, 31.5, 29.0], observed=30.2, version=2, **kw)
    pd.testing.assert_frame_equal(one, two)


# --------------------------------------------------------------- who gets which version

def test_a_run_without_a_feature_version_is_scored_as_version_one():
    from app.ml import inference
    assert inference.run_feature_version({}) == 1
    assert inference.run_feature_version({"feature_version": 2}) == 2


def test_a_cached_year_built_with_another_version_is_rebuilt(tmp_path, monkeypatch):
    from app.ml import pooled_training as pt
    built = []

    def fake_build(init_date_min=None, init_date_max=None, feature_version=None):
        built.append(feature_version)
        return pd.DataFrame({"init_date": [pd.Timestamp("2000-01-01")], "x": [1.0]}), 1
    monkeypatch.setattr(pt, "_build_paired_in_chunks", fake_build)
    path = pt.cache_year(2000, tmp_path)
    assert pt.cached_feature_version(path) == contracts.FEATURE_VERSION
    pt.cache_year(2000, tmp_path)
    assert built == [contracts.FEATURE_VERSION], "a current-version cache is reused"

    monkeypatch.setattr(contracts, "FEATURE_VERSION", contracts.FEATURE_VERSION + 1)
    monkeypatch.setattr(pt, "FEATURE_VERSION", contracts.FEATURE_VERSION)
    pt.cache_year(2000, tmp_path)
    assert len(built) == 2, "a cache from another feature version must be rebuilt"
