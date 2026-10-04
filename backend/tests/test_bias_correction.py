"""Busts on bias-corrected error: a bust means the forecast failed, not that it is always off.

Most temperature, humidity and soil-moisture "busts" were a district's steady bias: a fixed
per-district/variable/lead error is 64% of squared error for temperature and humidity and
90% for soil moisture (docs/known-issues.md, 2026-09-25). GEFS 2 m humidity runs ~16 %RH
drier than ERA5 at Day 1 (Nov 2017), and removing a per-district-lead mean shrank the
humidity bust threshold from 27.6 to 10.4 %RH. Ordinary bias correction removes that, so a
forecaster would not call it a failure.

The bias table is fitted on training cycles only, per (district, variable, lead, season),
and applied identically everywhere a forecast meets its label: training, validation, test,
live scoring and re-scoring. A cell with too few training events has no bias - NaN, never 0.

PLUMBING FIXTURES: hand-built events for arithmetic only, never used to produce a metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app import contracts
from app.features import bias


def _events(n, region="D1", variable="temperature_c", lead=1, season="JJAS",
            fc=30.0, obs=28.0):
    return pd.DataFrame({"region_id": region, "variable": variable,
                         "lead_time_days": lead, "season": season,
                         "fc_mean": np.full(n, fc, dtype=float),
                         "obs": np.full(n, obs, dtype=float)})


# --------------------------------------------------------------- fitting the table

def test_the_bias_is_the_mean_signed_error_of_each_cell():
    ev = pd.concat([_events(150, fc=30.0, obs=28.0),                 # +2.0
                    _events(150, region="D2", fc=20.0, obs=21.5)])   # -1.5
    t = bias.fit_bias_table(ev)
    got = t[t["level"] == "lead_season"].set_index("region_id")["bias"]
    assert got["D1"] == pytest.approx(2.0)
    assert got["D2"] == pytest.approx(-1.5)


def test_each_lead_and_season_has_its_own_bias():
    ev = pd.concat([_events(150, lead=1, fc=30.0, obs=29.0),
                    _events(150, lead=5, fc=30.0, obs=27.0),
                    _events(150, lead=1, season="DJF", fc=10.0, obs=10.5)])
    t = bias.fit_bias_table(ev)
    t = t[t["level"] == "lead_season"].set_index(["lead_time_days", "season"])["bias"]
    assert t[(1, "JJAS")] == pytest.approx(1.0)
    assert t[(5, "JJAS")] == pytest.approx(3.0)
    assert t[(1, "DJF")] == pytest.approx(-0.5)


def test_a_thin_cell_has_no_bias_not_zero():
    """Too few events at every level: no bias at all - the forecast then has no corrected
    value, which is missing, never zero."""
    t = bias.fit_bias_table(_events(bias.MIN_BIAS_EVENTS - 1))
    got = bias.bias_for(_members([30.0], [28.0]), t)
    assert np.isnan(got[0])


def test_a_thin_lead_backs_off_to_the_district_season_across_leads():
    """Few events at lead 3, plenty across leads: lead 3 gets the district-season bias."""
    ev = pd.concat([_events(150, lead=1, fc=30.0, obs=28.0),          # +2.0
                    _events(150, lead=2, fc=31.0, obs=28.0),          # +3.0
                    _events(10, lead=3, fc=40.0, obs=28.0)])          # +12, too thin alone
    t = bias.fit_bias_table(ev)
    got = bias.bias_for(_members([30.0], [28.0], lead=3), t)[0]
    assert got == pytest.approx((2.0 * 150 + 3.0 * 150 + 12.0 * 10) / 310)
    assert bias.bias_for(_members([30.0], [28.0], lead=1), t)[0] == pytest.approx(2.0)


def test_a_season_never_trained_on_backs_off_to_the_district_all_year():
    t = bias.fit_bias_table(_events(150, season="JJAS", fc=30.0, obs=28.5))
    got = bias.bias_for(_members([30.0], [28.0], season="DJF"), t)[0]
    assert got == pytest.approx(1.5)


def test_fitting_year_by_year_equals_fitting_at_once():
    """The pooled trainer streams one cached year at a time; the table must not depend
    on how the events were chunked."""
    a = _events(120, fc=30.0, obs=28.0)
    b = _events(80, fc=31.0, obs=28.0)
    whole = bias.fit_bias_table(pd.concat([a, b]))
    acc = bias.accumulate_bias(None, a)
    acc = bias.accumulate_bias(acc, b)
    streamed = bias.finish_bias_table(acc)
    pd.testing.assert_frame_equal(streamed, whole)
    finest = streamed[streamed["level"] == "lead_season"]
    assert finest["bias"].iloc[0] == pytest.approx((2.0 * 120 + 3.0 * 80) / 200)
    assert finest["n"].iloc[0] == 200


def test_wind_direction_has_no_bias():
    """Direction is not a label variable and a mean of signed angles is not meaningful."""
    t = bias.fit_bias_table(_events(150, variable="wind_direction_deg", fc=10.0, obs=350.0))
    assert t.empty


# --------------------------------------------------------------- applying it

def _members(fc, obs, region="D1", variable="temperature_c", lead=1, season="JJAS"):
    return pd.DataFrame({"region_id": region, "variable": variable,
                         "lead_time_days": lead, "season": season,
                         "forecast_value": np.asarray(fc, dtype=float),
                         "observed_value": np.asarray(obs, dtype=float)})


def _table(b=2.0, **keys):
    row = {"region_id": "D1", "variable": "temperature_c", "lead_time_days": 1,
           "season": "JJAS", "bias": b, "n": 500, "level": "lead_season"}
    row.update(keys)
    return pd.DataFrame([row])


def test_applying_corrects_the_forecast_and_its_error():
    df = _members([30.0, 31.0], [28.0, 28.0])
    out = bias.apply_bias(df, _table(2.0))
    assert out["forecast_value"].tolist() == pytest.approx([28.0, 29.0])
    assert out["abs_error"].tolist() == pytest.approx([0.0, 1.0])
    assert out["bias_correction"].tolist() == pytest.approx([2.0, 2.0])


def test_a_forecast_without_a_bias_cell_has_no_error_not_its_raw_one():
    df = _members([30.0], [28.0], region="D9")
    out = bias.apply_bias(df, _table(2.0))
    assert np.isnan(out["forecast_value"].iloc[0])
    assert np.isnan(out["abs_error"].iloc[0])


def test_rows_of_other_variables_are_left_alone():
    df = _members([10.0], [350.0], variable="wind_direction_deg")
    df["abs_error"] = 20.0
    out = bias.apply_bias(df, _table(2.0))
    assert out["forecast_value"].iloc[0] == 10.0
    assert out["abs_error"].iloc[0] == 20.0


def test_live_rows_without_an_observation_still_get_the_corrected_forecast():
    df = _members([30.0], [np.nan])
    out = bias.apply_bias(df, _table(2.0))
    assert out["forecast_value"].iloc[0] == pytest.approx(28.0)
    assert np.isnan(out["abs_error"].iloc[0])


def test_categorical_keys_work_like_strings():
    df = _members([30.0, 30.0], [28.0, 28.0])
    for c in ("region_id", "variable", "season"):
        df[c] = df[c].astype("category")
    out = bias.apply_bias(df, _table(2.0))
    assert out["forecast_value"].tolist() == pytest.approx([28.0, 28.0])


# --------------------------------------------------------------- the label

def test_wind_direction_is_not_a_label_variable():
    assert "wind_direction_deg" not in contracts.LABEL_VARIABLES
    assert "temperature_c" in contracts.LABEL_VARIABLES
    assert contracts.LABEL_VERSION == 2


def test_gefs_own_forecast_is_kept_for_display():
    """The site shows the forecast that might bust, not the corrected value the model reads."""
    df = _members([30.0, 31.0], [28.0, 28.0])
    out = bias.apply_bias(df, _table(2.0))
    assert out["forecast_value_raw"].tolist() == pytest.approx([30.0, 31.0])
    assert out["forecast_value"].tolist() == pytest.approx([28.0, 29.0])
