"""C2 - time-lagged ensemble: a poor-man's ensemble built by pooling this cycle's real
members with earlier cycles' ensemble means, for the same (region, variable, valid_date).

Every GEFS reforecast cycle carries only 5 of the operational feed's 31 members
(CLAUDE.md known limitations). A time-lagged ensemble (lagged-average forecasting) is the
standard, published way to cheaply widen that: earlier cycles that are still valid for the
same target date add information the 5-member cycle alone does not have. This pools each
earlier cycle's ensemble MEAN as one extra pseudo-member, not its individual members -
exactly the trajectory data C1's jumpiness already reads via forecast_trajectories /
forecast_history, so this needs no new data-reading path and no new memory risk.

Every expected value below is computed by hand in the comment beside it, not by
re-running the code under test. The input numbers are small, invented and labelled as
such: these tests pin arithmetic and causality, and nothing here reaches a metric.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.features import engineering as fe

VALID = pd.Timestamp("2017-11-10")


def _fc_rows(region, variable, valid, init_to_members):
    """Canonical forecast rows. `init_to_members` maps init date -> member values.
    ARITHMETIC FIXTURE - invented numbers, never used to produce a metric."""
    rows = []
    for init, members in init_to_members.items():
        init = pd.Timestamp(init)
        lead = (pd.Timestamp(valid) - init).days + 1          # valid = init + (lead - 1)
        for i, v in enumerate(members):
            rows.append({
                "region_id": region, "variable": variable, "value_type": "forecast",
                "valid_date": pd.Timestamp(valid), "init_date": init,
                "lead_time_days": lead, "ensemble_member_id": f"m{i}", "value": v,
            })
    return pd.DataFrame(rows)


def _laf(fc, **kw):
    traj = fe.forecast_trajectories(fc)
    out = fe.compute_time_lagged_ensemble(fc, traj, **kw)
    return out.set_index("init_date").sort_index()


# Ensemble means 30, 32, 30 on inits 8, 9, 10 November, all valid for 10 November.
# 8 Nov's members are non-identical ([28, 32]) so its own spread is not degenerately
# zero - the first-cycle ratio test below needs a real, non-zero denominator.
MEMBERS = {"2017-11-08": [28, 32], "2017-11-09": [32, 32], "2017-11-10": [29, 31]}


def test_the_pool_is_this_cycles_members_plus_earlier_cycles_means():
    j = _laf(_fc_rows("A", "temperature_c", VALID, MEMBERS), window=3)
    row = j.loc["2017-11-10"]
    # pool = [29, 31] (this cycle's own members) + [32, 30] (means of 9 Nov, 8 Nov):
    # mean = (29+31+32+30)/4 = 30.5
    assert row["laf_pool_mean"] == pytest.approx(30.5)
    assert row["laf_pool_size"] == 4
    # sample std (ddof=1) of [29, 31, 32, 30]: mean 30.5, deviations -1.5,0.5,1.5,-0.5,
    # squares 2.25+0.25+2.25+0.25=5, /3 = 1.666667, sqrt = 1.290994.
    assert row["laf_pool_std"] == pytest.approx(math.sqrt(5 / 3))


def test_the_ratio_compares_the_pool_to_this_cycles_own_spread():
    j = _laf(_fc_rows("A", "temperature_c", VALID, MEMBERS), window=3)
    row = j.loc["2017-11-10"]
    # this cycle's own members [29, 31]: mean 30, std = |31-29|/sqrt(2) = sqrt(2).
    own_std = math.sqrt(2)
    assert row["laf_spread_ratio"] == pytest.approx(math.sqrt(5 / 3) / own_std)


def test_the_window_bounds_how_many_earlier_cycles_are_pooled():
    j = _laf(_fc_rows("A", "temperature_c", VALID, MEMBERS), window=2)
    row = j.loc["2017-11-10"]
    # window=2 pools only the one cycle immediately before: [29, 31, 32], mean 30.667.
    assert row["laf_pool_size"] == 3
    assert row["laf_pool_mean"] == pytest.approx(92 / 3)


def test_the_first_cycle_has_no_earlier_mean_to_pool_and_the_ratio_is_one():
    """No history yet - the pool is exactly this cycle's own members, so the ratio is 1,
    not NaN and not zero: nothing was learned, but nothing is unknown either."""
    j = _laf(_fc_rows("A", "temperature_c", VALID, MEMBERS), window=3)
    row = j.loc["2017-11-08"]
    assert row["laf_pool_size"] == 2
    assert row["laf_pool_mean"] == pytest.approx(30.0)
    # pool == this cycle's own 2 members exactly (no earlier cycle): pool_std must equal
    # this cycle's own std, so the ratio is exactly 1 - by honest division, not a special
    # case, which is why 8 Nov's members are non-identical above (a real, non-zero std).
    assert row["laf_spread_ratio"] == pytest.approx(1.0)


def test_a_single_member_cycle_contributes_zero_within_cycle_variance():
    """n0=1 means (n0-1)=0: that group's own spread cannot be known and correctly
    contributes nothing to the pooled sum of squares - not NaN, not a fabricated value."""
    one_member = {"2017-11-09": [40], "2017-11-10": [50]}
    j = _laf(_fc_rows("A", "temperature_c", VALID, one_member), window=3)
    row = j.loc["2017-11-10"]
    # pool = [50] (this cycle's one member) + [40] (prior mean): mean 45,
    # sample std of [50, 40]: |50-40|/sqrt(2) = 7.071068.
    assert row["laf_pool_mean"] == pytest.approx(45.0)
    assert row["laf_pool_std"] == pytest.approx(10 / math.sqrt(2))


def test_regions_variables_and_valid_dates_do_not_mix():
    a = _fc_rows("A", "temperature_c", VALID, MEMBERS)
    b = _fc_rows("B", "temperature_c", VALID, {"2017-11-09": [0, 0], "2017-11-10": [100, 100]})
    fc = pd.concat([a, b], ignore_index=True)
    traj = fe.forecast_trajectories(fc)
    out = fe.compute_time_lagged_ensemble(fc, traj, window=3).set_index(
        ["region_id", "init_date"])
    a_row = out.loc[("A", pd.Timestamp("2017-11-10"))]
    b_row = out.loc[("B", pd.Timestamp("2017-11-10"))]
    assert a_row["laf_pool_mean"] == pytest.approx(30.5)
    # B's pool: [100, 100] (own members) + [0] (prior mean): mean 66.667.
    assert b_row["laf_pool_mean"] == pytest.approx(200 / 3)


def test_a_later_cycle_never_changes_an_earlier_row():
    """Causality: a forecast issued on 9 Nov can only know cycles issued by 9 Nov."""
    early_only = {k: v for k, v in MEMBERS.items() if k != "2017-11-10"}
    before = _laf(_fc_rows("A", "temperature_c", VALID, early_only), window=3)
    after = _laf(_fc_rows("A", "temperature_c", VALID, MEMBERS), window=3)
    cols = ["laf_pool_mean", "laf_pool_std", "laf_pool_size"]
    common = before.index.intersection(after.index)
    assert len(common) == 2
    pd.testing.assert_frame_equal(before.loc[common, cols], after.loc[common, cols])


# ----------------------------------------------------------- wired into the frames

def _canonical(fc: pd.DataFrame) -> pd.DataFrame:
    """Forecast rows plus an observation for every valid date they cover.
    PLUMBING FIXTURE - invented numbers, never used to produce a metric."""
    ob = (fc[["region_id", "variable", "valid_date"]].drop_duplicates()
          .assign(value_type="observed", value=31.0, init_date=pd.NaT,
                  lead_time_days=np.nan, ensemble_member_id=None))
    return pd.concat([fc, ob], ignore_index=True)


def test_the_training_frame_carries_the_laf_features_on_every_member_row():
    frame = fe.build_training_frame(_canonical(_fc_rows("A", "temperature_c", VALID, MEMBERS)))
    for col in fe.LAF_FEATURES:
        assert col in frame.columns
    last = frame[frame["init_date"] == pd.Timestamp("2017-11-10")]
    assert len(last) == 2                       # both members
    assert last["laf_pool_mean"].tolist() == pytest.approx([30.5, 30.5])


def test_history_outside_the_frame_feeds_the_pool_but_is_not_paired():
    full = _fc_rows("A", "temperature_c", VALID, MEMBERS)
    current = full[full["init_date"] == pd.Timestamp("2017-11-10")]
    history = full[full["init_date"] < pd.Timestamp("2017-11-10")]

    frame = fe.build_training_frame(_canonical(current), forecast_history=history)
    assert set(frame["init_date"]) == {pd.Timestamp("2017-11-10")}
    assert frame["laf_pool_mean"].iloc[0] == pytest.approx(30.5)
    assert frame["laf_pool_size"].iloc[0] == 4


def test_regressor_and_classifier_both_see_the_features():
    from app.features import pivot as pv
    from app.ml import regressors as reg_mod

    frame = fe.build_training_frame(_canonical(_fc_rows("A", "temperature_c", VALID, MEMBERS)))
    for col in fe.LAF_FEATURES:
        assert col in reg_mod.feature_columns(frame)

    pred = pd.Series(1.0, index=frame.index)
    ev = pv.build_event_frame(frame, pred, {"temperature_c": 5.0}, {"temperature_c": 3.0})
    row = ev.set_index("init_date").loc[pd.Timestamp("2017-11-10")]
    assert row["laf_pool_mean_temperature_c"] == pytest.approx(30.5)
    feats = pv.classifier_feature_columns(ev)
    assert "laf_pool_mean_temperature_c" in feats
    assert "laf_spread_ratio_temperature_c" in feats
