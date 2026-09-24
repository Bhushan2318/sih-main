"""Never score a (variable, lead day) the training archive does not hold.

The GEFSv12 reforecast archive stops 10 m wind at 120 h and soil moisture at 72 h
(`VAR_SPEC["...]["max_lead_h"]` in the fetch script). Live GEFS carries all ten days, so
without a cap the regressors were asked about wind on Days 6-10 and soil moisture on Days
4-10 - leads they never saw - and the live map named wind the "driver" of a Day-9 bust.
Checked on the serving bundle 2026-09-25: reforecast soil moisture exists for Days 1-3 only.

The frames below are small plumbing frames (labelled, not real data); nothing here
produces a metric.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

from app import contracts
from app.features import engineering as fe
from app.features import history
from app.ml import inference

BACKEND = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fetch_gefs_caps", BACKEND / "scripts" / "fetch_gefs_reforecast_sample.py")
fetch = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fetch
_spec.loader.exec_module(fetch)


def test_caps_match_what_the_archive_fetch_actually_holds():
    spec = fetch.VAR_SPEC
    assert contracts.ARCHIVE_MAX_LEAD_DAYS == {
        "soil_moisture_pct": spec["soilw_bgrnd"]["max_lead_h"] // 24,
        "wind_speed_ms": spec["ugrd_hgt"]["max_lead_h"] // 24,
        "wind_direction_deg": spec["ugrd_hgt"]["max_lead_h"] // 24,
    }
    assert spec["ugrd_hgt"]["max_lead_h"] == spec["vgrd_hgt"]["max_lead_h"]
    # every other paired variable runs the full ten days, so needs no cap
    for key in ("tmp_2m", "spfh_2m", "pres_sfc", "pwat_eatm", "apcp_sfc"):
        assert spec[key]["max_lead_h"] == 240


def _rows():
    """Plumbing frame: forecast rows at the cap edges, plus one observed row."""
    out = []
    for var, lead in [("soil_moisture_pct", 3), ("soil_moisture_pct", 4),
                      ("wind_speed_ms", 5), ("wind_speed_ms", 6),
                      ("wind_direction_deg", 6), ("rainfall_mm", 10)]:
        out.append({"region_id": "IN-TEST", "variable": var, "value_type": "forecast",
                    "lead_time_days": lead, "value": 1.0,
                    "init_date": pd.Timestamp("2026-09-24"),
                    "valid_date": pd.Timestamp("2026-09-24") + pd.Timedelta(days=lead - 1)})
    out.append({"region_id": "IN-TEST", "variable": "soil_moisture_pct",
                "value_type": "observed", "lead_time_days": None, "value": 30.0,
                "init_date": None, "valid_date": pd.Timestamp("2026-09-30")})
    return pd.DataFrame(out)


def test_drop_keeps_archive_leads_and_observations():
    kept = fe.drop_beyond_archive_leads(_rows())
    got = set(zip(kept["variable"], kept["value_type"], kept["lead_time_days"].fillna(-1)))
    assert got == {
        ("soil_moisture_pct", "forecast", 3), ("wind_speed_ms", "forecast", 5),
        ("rainfall_mm", "forecast", 10), ("soil_moisture_pct", "observed", -1),
    }


def test_score_cycle_never_builds_features_from_beyond_the_archive(monkeypatch):
    class _State:
        run_id = "run_PLUMBING"
        historical_bust_freq = None
        jump_climatology = None

    monkeypatch.setattr(inference.precomputed, "read_scored_cycle", lambda *a, **k: None)
    monkeypatch.setattr(inference.parquet_store, "store_fingerprint", lambda: "fp")
    monkeypatch.setattr(inference.settings, "serving_read_only", False)
    inference._score_cache.clear()
    rows = _rows()
    monkeypatch.setattr(
        inference.parquet_store, "read_dataset",
        lambda value_types=None, **k: rows[rows.value_type.isin(value_types)].copy())
    monkeypatch.setattr(inference.parquet_store, "distinct_forecast_init_dates", lambda: [])
    seen = {}

    def _capture(subset, **k):
        seen["subset"] = subset
        return pd.DataFrame()

    monkeypatch.setattr(inference.fe, "build_training_frame", _capture)
    inference.score_cycle(_State(), init_date="2026-09-24")
    fc = seen["subset"][seen["subset"].value_type == "forecast"]
    assert not ((fc.variable == "soil_moisture_pct") & (fc.lead_time_days > 3)).any()
    assert not (fc.variable.isin(["wind_speed_ms", "wind_direction_deg"])
                & (fc.lead_time_days > 5)).any()
    assert (fc.variable == "rainfall_mm").any()


def test_jumpiness_history_is_capped_the_same_way(monkeypatch):
    rows = _rows()
    monkeypatch.setattr(history.parquet_store, "read_dataset",
                        lambda **k: rows[rows.value_type == "forecast"].copy())
    seen = {}

    def _capture(fc):
        seen["fc"] = fc
        return pd.DataFrame(columns=fe.TRAJECTORY_COLUMNS)

    monkeypatch.setattr(history.fe, "forecast_trajectories", _capture)
    history.forecast_history(pd.Timestamp("2026-09-25"), [pd.Timestamp("2026-09-24").date()])
    fc = seen["fc"]
    assert set(zip(fc.variable, fc.lead_time_days)) == {
        ("soil_moisture_pct", 3), ("wind_speed_ms", 5), ("rainfall_mm", 10)}
