"""The label-drift diagnostic: per-year event errors, and what explains a bust-rate step.

Training bust rates sit at 0.39-0.45 for 2000-2013 and 0.49 for 2014-2015, with every year's
observations from the same ERA5 script. The diagnostic records each year's ensemble-mean
error per event and variable once, then answers from those files: per year and variable,
the signed bias, the RMSE and the rate above the training p90; and, held out one year at a
time, which bias key (none, season, month) leaves the least error.

SYNTHETIC, LABELLED: tiny hand-built frames for the arithmetic; no metric is produced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import diagnose_label_drift as dd


def _members(values, variable="temperature_c", lead=1, init="2014-01-05", region="IN-A"):
    init = pd.Timestamp(init)
    return pd.DataFrame({
        "region_id": region, "variable": variable, "value_type": "forecast",
        "init_date": init, "lead_time_days": lead,
        "valid_date": init + pd.Timedelta(days=lead - 1),
        "ensemble_member_id": [f"m{i}" for i in range(len(values))], "value": values})


def _obs(value, valid="2014-01-05", variable="temperature_c", region="IN-A"):
    return pd.DataFrame({"region_id": [region], "variable": [variable],
                         "valid_date": [pd.Timestamp(valid)], "value": [value]})


def test_event_error_is_the_ensemble_mean_minus_the_observation():
    ev = dd.event_errors(_members([10.0, 12.0, 14.0]), _obs(11.0))
    assert ev.loc[0, "temperature_c"] == pytest.approx(1.0)
    assert ev.loc[0, "month"] == 1 and ev.loc[0, "lead_time_days"] == 1


def test_wind_direction_error_is_circular():
    fc = _members([350.0, 10.0], variable="wind_direction_deg")
    ev = dd.event_errors(fc, _obs(20.0, variable="wind_direction_deg"))
    assert ev.loc[0, "wind_direction_deg"] == pytest.approx(-20.0, abs=1e-6)


def test_a_forecast_without_an_observation_is_missing_not_zero():
    ev = dd.event_errors(_members([1.0, 2.0]), _obs(0.0, valid="2014-02-01"))
    assert ev.empty or ev["temperature_c"].isna().all()


def _year_frame(year, bias, n=400, seed=0):
    rng = np.random.default_rng(seed + year)
    init = pd.date_range(f"{year}-01-01", periods=n, freq="D")[:n]
    month = init.month
    e = rng.normal(0, 1, n) + bias + np.where(np.isin(month, [6, 7, 8, 9]), 2.0, 0.0)
    return pd.DataFrame({"region_id": "IN-A", "init_date": init, "lead_time_days": 1,
                         "month": month, "temperature_c": e})


def test_a_seasonal_bias_is_found_by_the_season_key():
    frames = {y: _year_frame(y, bias=0.5) for y in range(2000, 2006)}
    mse = dd.held_out_mse(frames, keys=("none", "season", "month"))
    t = mse[mse.variable == "temperature_c"].set_index("key")["mse"]
    assert t["season"] < t["none"]
    assert t["season"] == pytest.approx(t["month"], rel=0.1)


def test_a_step_in_bias_shows_in_that_years_signed_error_and_bust_rate():
    frames = {y: _year_frame(y, bias=0.0 if y < 2004 else 1.5) for y in range(2000, 2006)}
    per_year = dd.per_year_summary(frames, train_years=range(2000, 2004))
    t = per_year[per_year.variable == "temperature_c"].set_index("year")
    assert t.loc[2005, "mean_error"] - t.loc[2001, "mean_error"] == pytest.approx(1.5, abs=0.3)
    assert t.loc[2005, "above_p90"] > 2 * t.loc[2001, "above_p90"]
    assert t.loc[2001, "above_p90"] == pytest.approx(0.1, abs=0.04)
