"""Physically impossible member values never reach a training frame.

The 2000-2019 store holds two corrupt member-days from the reforecast archive (found
2026-10-04 by scripts/diagnose_label_drift.py): member p03 of 2004-03-05, Day 2 2 m
temperature, -23,380,072 to +21,808,336 C in all 666 districts; and member p01 of
2019-02-09, Day 1 column water vapour, ~12,500 kg/m2 (others 0.7-55). Averaged per
(district, lead, season), one such value moves a bias by thousands of degrees. They are
dropped (missing, never replaced) and counted.

PLUMBING FIXTURES: hand-built rows; the bounds are checked against real 2017 extremes.
"""
from __future__ import annotations

import pandas as pd

from app import contracts
from app.features import engineering as fe


def _rows(variable, values, value_type="forecast"):
    return pd.DataFrame({"variable": variable, "value_type": value_type, "value": values})


def test_the_two_corrupt_member_days_are_dropped():
    df = pd.concat([_rows("temperature_c", [25.0, -23380071.9, 21808336.3]),
                    _rows("atmospheric_moisture_kgm2", [10.7, 12509.8])], ignore_index=True)
    out, n = fe.drop_implausible(df)
    assert n == 3
    assert sorted(out["value"]) == [10.7, 25.0]


def test_real_extremes_are_kept():
    """Measured min/max in three 2017 windows (forecast and observed), all kept."""
    real = {"temperature_c": [-33.11, 42.94], "humidity_pct": [4.67, 100.0],
            "rainfall_mm": [0.0, 367.41], "pressure_hpa": [985.32, 1049.71],
            "wind_speed_ms": [0.0, 16.69], "wind_direction_deg": [0.0, 360.0],
            "soil_moisture_pct": [-0.0, 100.0], "atmospheric_moisture_kgm2": [0.2, 79.92]}
    df = pd.concat([_rows(v, x) for v, x in real.items()], ignore_index=True)
    out, n = fe.drop_implausible(df)
    assert n == 0 and len(out) == len(df)


def test_every_label_variable_has_bounds():
    assert set(contracts.PLAUSIBLE_RANGE) >= {
        "temperature_c", "humidity_pct", "rainfall_mm", "pressure_hpa", "wind_speed_ms",
        "wind_direction_deg", "soil_moisture_pct", "atmospheric_moisture_kgm2"}


def test_an_unknown_variable_and_nan_pass_through():
    df = pd.concat([_rows("cape_jkg", [1e6]), _rows("temperature_c", [float("nan")])],
                   ignore_index=True)
    out, n = fe.drop_implausible(df)
    assert n == 0 and len(out) == 2


def test_every_frame_builder_drops_them_first(monkeypatch):
    """build_training_frame is the one path training, scoring, live and replay share."""
    seen = []
    real = fe.drop_implausible

    def spy(df):
        out, n = real(df)
        seen.append(n)
        return out, n
    monkeypatch.setattr(fe, "drop_implausible", spy)
    df = pd.concat([_rows("temperature_c", [25.0, -23380071.9])], ignore_index=True)
    df = df.assign(region_id="IN-A", init_date=pd.Timestamp("2004-03-05"),
                   valid_date=pd.Timestamp("2004-03-06"), lead_time_days=2,
                   ensemble_member_id=["c00", "p03"], verification_status=None)
    try:
        fe.build_training_frame(df)
    except Exception:  # noqa: BLE001 - a two-row frame may not build; the guard ran first
        pass
    assert seen and seen[0] == 1
