"""Replay needs the day an event peaked, not just its init day, to open on the lead day
the event is remembered for. `app.services.replay_cases.ReplayCase.peak_valid_date`
already carries this (it is what `valid_date = init + (lead - 1)` is checked against when a
case is built); only the API schema and `_event_cycles` were not passing it through.

Plumbing-only: no retrain, no real cycle scored. See test_ml.py for the end-to-end case
(a real GEFS/ERA5 slice, scored and written as a case, then read back through Replay).
"""
from __future__ import annotations

import datetime as dt

from app.api import schemas
from app.services import replay_cases, replay_service


class _State:
    run_id = "run_PLUMBING"


def _fake_case() -> replay_cases.ReplayCase:
    return replay_cases.ReplayCase(
        id="fake", title="Fake event", init_date=dt.date(2018, 8, 13),
        peak_valid_date=dt.date(2018, 8, 15), focus_region_id="IN-KL-IDUKKI",
        focus_variable="rainfall_mm",
    )


def _fake_summary_json() -> dict:
    """What `read_case_summary` returns for an already-built case: exactly the fields
    `summary_from_scored` writes today, with no `peak_valid_date` - a run built before this
    field existed. The value must still come through, from the catalogued case rather than
    from re-running every past build."""
    return {
        "init_date": "2018-08-13", "lead_days": [1, 2, 3], "n_regions": 666,
        "peak_bust_probability": 0.9, "peak_lead_day": 1, "peak_region_id": "IN-AN-X",
        "peak_region_name": "X", "n_high_regions_peak": 1, "verified": True,
        "verified_lead_days": 3, "peak_region_abs_error": None,
        "peak_region_relative_error": None, "peak_region_variable": None,
        "peak_region_unit": None, "medium_range_growth": 0.0,
    }


def test_event_cycles_expose_the_catalogued_peak_valid_date(monkeypatch):
    case = _fake_case()
    monkeypatch.setattr(replay_cases, "CASES", (case,))
    monkeypatch.setattr(replay_cases, "read_case_summary",
                         lambda run_id, c: _fake_summary_json())

    events = replay_service._event_cycles(_State())
    assert len(events) == 1
    assert events[0].peak_valid_date == case.peak_valid_date


def test_a_live_forecast_cycle_has_no_peak_valid_date():
    """peak_valid_date is an event fact - a live cycle summary must not carry a stray one."""
    assert "peak_valid_date" in schemas.ReplayCycleSummary.model_fields
    default = schemas.ReplayCycleSummary(init_date=dt.date(2026, 9, 25), kind="forecast")
    assert default.peak_valid_date is None
