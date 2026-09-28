"""A packaged cycle keeps what the model said and picks up what happened.

A live cycle is scored in the CI run that ingests it, before any of its days have
happened. `score_cycle` returns an existing artifact as it stands, so every later run
packaged that first answer again: measured 2026-09-28 on the live bundle, the 2026-09-21
and 2026-09-22 cycles held 0 observations each while the store had them for 284 of their
cells, and the 666-district cycles from 2026-09-23 would never have verified at all.

So a run that finds an artifact re-reads the cycle's observations from the store and
leaves every prediction exactly as issued. Re-scoring instead would fill
`forecast_error_lag` from the very observations being verified against.

The frames below are plumbing stand-ins shaped like the real artifact, labelled as such;
nothing here produces a metric. The same behaviour on real sample data is
tests/test_api.py::test_a_cycle_packaged_before_its_observations_arrived_is_verified_by_a_later_run.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from app.ml import inference, precomputed
from app.ml.inference import ScoredCycle

INIT = pd.Timestamp("2026-09-21")
RIDS = ["IN-MH-NAGPUR", "IN-TN-CHENNAI"]
VARS = ["rainfall_mm", "temperature_c"]


def _issued() -> ScoredCycle:
    """Stand-in: a cycle scored before any observation existed."""
    rows = [(r, lead, v) for r in RIDS for lead in (1, 2) for v in VARS]
    pv = pd.DataFrame({
        "region_id": pd.Categorical([r for r, _, _ in rows]),
        "lead_time_days": [lead for _, lead, _ in rows],
        "variable": [v for _, _, v in rows],
        "valid_date": [INIT + pd.Timedelta(days=lead - 1) for _, lead, _ in rows],
        "predicted_value": np.arange(len(rows), dtype=float) + 10.0,
        "observed_value": np.nan,
        "predicted_error": np.arange(len(rows), dtype=float) / 10,
        "ensemble_spread": 1.0,
        "ensemble_member_count": 5,
        "confidence": 0.5,
        "bust_threshold": 3.0,
    })
    ev = pd.DataFrame({
        "region_id": pd.Categorical([r for r in RIDS for _ in (1, 2)]),
        "init_date": INIT,
        "valid_date": [INIT + pd.Timedelta(days=lead - 1) for _ in RIDS for lead in (1, 2)],
        "lead_time_days": [lead for _ in RIDS for lead in (1, 2)],
        "pred_err_rainfall_mm": [0.1, 0.2, 0.3, 0.4],
        "bust_probability": [0.9, 0.2, 0.6, 0.4],
        "risk_band": ["high", "low", "medium", "low"],
    })
    return ScoredCycle(run_id="run_A", init_date=INIT, events=ev, per_variable=pv,
                       n_rows_scored=40)


def _store_with(observed: pd.DataFrame, monkeypatch,
                forecast_file="gefs_operational_forecast_20260921_00z.csv",
                forecast_method="region_id") -> list:
    """Stand-in store: `observed`, and forecast rows for every stand-in district from
    `forecast_file`, placed by `forecast_method`. Returns the observation reads made."""
    calls = []

    def read_dataset(*a, **k):
        if k.get("value_types") == ["forecast"]:
            assert set(k.get("columns") or ()) <= {"source_file", "region_resolution_method"}, \
                "the refresh must never read forecast values"
            return pd.DataFrame({"source_file": forecast_file,
                                 "region_resolution_method": forecast_method}, index=RIDS)
        calls.append(k)
        assert k.get("value_types") == ["observed"]
        return observed.copy()

    monkeypatch.setattr(inference.parquet_store, "read_dataset", read_dataset)
    return calls


def _obs(rid, var, day, value, status="final"):
    return {"region_id": rid, "variable": var, "valid_date": day, "value": value,
            "value_type": "observed", "verification_status": status,
            "init_date": None, "lead_time_days": None, "ensemble_member_id": None}


def test_observations_that_arrived_later_are_filled_in(monkeypatch):
    issued = _issued()
    day1 = INIT.date()
    _store_with(pd.DataFrame([
        _obs("IN-MH-NAGPUR", "rainfall_mm", day1, 4.0),
        _obs("IN-MH-NAGPUR", "rainfall_mm", day1, 6.0),  # two rows for one cell: the mean
        _obs("IN-TN-CHENNAI", "temperature_c", day1, 29.5),
    ]), monkeypatch)

    later = inference.refresh_observations(issued)

    pv = later.per_variable.set_index(["region_id", "lead_time_days", "variable"])
    assert pv.loc[("IN-MH-NAGPUR", 1, "rainfall_mm"), "observed_value"] == 5.0
    assert pv.loc[("IN-TN-CHENNAI", 1, "temperature_c"), "observed_value"] == 29.5
    # A day not yet observed stays missing - never zero.
    assert pv.loc[("IN-MH-NAGPUR", 2, "rainfall_mm"), "observed_value"] != 0
    assert np.isnan(pv.loc[("IN-MH-NAGPUR", 2, "rainfall_mm"), "observed_value"])
    assert later.per_variable["observed_value"].notna().sum() == 2


def test_what_the_model_said_is_left_exactly_as_issued(monkeypatch):
    issued = _issued()
    _store_with(pd.DataFrame([_obs("IN-MH-NAGPUR", "rainfall_mm", INIT.date(), 4.0)]),
                monkeypatch)
    later = inference.refresh_observations(issued)

    pd.testing.assert_frame_equal(later.per_variable.drop(columns="observed_value"),
                                  issued.per_variable.drop(columns="observed_value"))
    pd.testing.assert_frame_equal(later.events[issued.events.columns], issued.events)
    assert (later.run_id, later.init_date, later.n_rows_scored) == \
        (issued.run_id, issued.init_date, issued.n_rows_scored)


def test_the_events_actual_error_agrees_with_the_refreshed_observations(monkeypatch):
    """The event frame carries |forecast mean - observed| per variable; after a refresh it
    must describe the same observations as the per-variable table, not the old ones."""
    issued = _issued()
    _store_with(pd.DataFrame([_obs("IN-MH-NAGPUR", "rainfall_mm", INIT.date(), 4.0)]),
                monkeypatch)
    later = inference.refresh_observations(issued)
    ev = later.events.set_index(["region_id", "lead_time_days"])
    pred = issued.per_variable.set_index(["region_id", "lead_time_days", "variable"])
    want = abs(pred.loc[("IN-MH-NAGPUR", 1, "rainfall_mm"), "predicted_value"] - 4.0)
    assert ev.loc[("IN-MH-NAGPUR", 1), "actual_err_rainfall_mm"] == want
    assert np.isnan(ev.loc[("IN-MH-NAGPUR", 2), "actual_err_rainfall_mm"])


def test_observations_are_read_for_the_cycles_own_window_only(monkeypatch):
    calls = _store_with(pd.DataFrame([_obs("IN-MH-NAGPUR", "rainfall_mm",
                                           INIT.date(), 4.0)]), monkeypatch)
    inference.refresh_observations(_issued())
    assert len(calls) == 1
    lo, hi = calls[0]["valid_date_min"], calls[0]["valid_date_max"]
    assert lo <= INIT.date() and hi >= (INIT + pd.Timedelta(days=9)).date()
    assert (hi - lo) <= dt.timedelta(days=20)


@pytest.mark.parametrize("method", ["name", "point_in_polygon", "nearest_polygon"])
def test_a_cycle_from_the_retired_city_point_feed_is_left_as_it_was(monkeypatch, method):
    """The live feed sampled 36 city points until the districts migration; the store now
    holds district area means for those days. Pairing the two would compare a point with a
    polygon mean - so the cycle keeps the observations it was scored with."""
    issued = _issued()
    issued.per_variable.loc[0, "observed_value"] = 7.0  # the point observation it had
    calls = _store_with(pd.DataFrame([
        _obs("IN-MH-NAGPUR", "rainfall_mm", INIT.date(), 4.0),
        _obs("IN-MH-NAGPUR", "temperature_c", INIT.date(), 30.0),
    ]), monkeypatch, forecast_method=method)

    later = inference.refresh_observations(issued)
    assert later is issued and not calls


@pytest.mark.parametrize("source,method", [
    ("gefs_operational_forecast_20260923_00z.csv", "region_id"),     # the district feed
    ("gefs_reforecast_india_2019.csv", "name"),                      # the reforecast samples
    ("gefs_reforecast_india_2017.parquet", "point_in_polygon"),      # the district store
])
def test_every_other_cycle_is_verified_anew(monkeypatch, source, method):
    """Area means found their districts by point_in_polygon in the reforecast store, so how
    a row was placed does not by itself say point or area; only the retired feed is held."""
    _store_with(pd.DataFrame([_obs("IN-MH-NAGPUR", "rainfall_mm", INIT.date(), 4.0)]),
                monkeypatch, forecast_file=source, forecast_method=method)
    later = inference.refresh_observations(_issued())
    assert later.per_variable["observed_value"].notna().sum() == 1


def test_an_observation_the_store_no_longer_holds_is_not_kept(monkeypatch):
    """The store is the truth: batches have been moved out of it on purpose (IMD rainfall,
    2026-09-26). A cycle is verified against what it holds now, as a fresh score would be."""
    issued = _issued()
    issued.per_variable.loc[0, "observed_value"] = 7.0
    _store_with(pd.DataFrame([_obs("IN-TN-CHENNAI", "rainfall_mm", INIT.date(), 1.0)]),
                monkeypatch)
    later = inference.refresh_observations(issued)
    assert np.isnan(later.per_variable.loc[0, "observed_value"])


def test_a_packaged_cycle_is_refreshed_not_rescored(tmp_path, monkeypatch):
    from scripts import package_for_deploy

    monkeypatch.setattr(precomputed, "default_dir", lambda: tmp_path)
    precomputed.write_scored_cycle(_issued())
    _store_with(pd.DataFrame([_obs("IN-MH-NAGPUR", "rainfall_mm", INIT.date(), 4.0)]),
                monkeypatch)

    def _must_not_score(*a, **k):
        raise AssertionError("re-scored a cycle that was already packaged")

    monkeypatch.setattr(inference, "score_cycle", _must_not_score)

    class _State:
        run_id = "run_A"

    assert package_for_deploy._score_and_write(_State(), INIT) > 0
    got = precomputed.read_scored_cycle("run_A", INIT)
    assert got.per_variable["observed_value"].notna().sum() == 1


def test_packaged_cycles_outside_the_window_are_pruned(tmp_path):
    """Nothing pruned them before: 14 cycles on 2026-09-28 against a window of 10, and every
    model's cycles would have stayed after a promotion."""
    window = [pd.Timestamp("2026-09-27") - pd.Timedelta(days=i) for i in range(3)]
    for init in window + [pd.Timestamp("2026-09-10")]:
        precomputed.write_scored_cycle(
            ScoredCycle("run_A", init, _issued().events, _issued().per_variable, 1), tmp_path)
    precomputed.write_scored_cycle(
        ScoredCycle("run_OLD", window[0], _issued().events, _issued().per_variable, 1), tmp_path)
    (tmp_path / "not-a-cycle").mkdir()

    removed = precomputed.prune(tmp_path, "run_A", window)

    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [precomputed.cycle_dir(tmp_path, "run_A", i).name for i in window] + ["not-a-cycle"])
    assert len(removed) == 2
    for init in window:
        assert precomputed.read_scored_cycle("run_A", init, tmp_path) is not None
