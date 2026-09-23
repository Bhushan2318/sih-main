"""The live observation side reads districts too, through the one weight table.

The forecast side stopped sampling 36 city points (see test_live_district_grid.py). If the
observation side kept sampling them, the bust label would compare an area mean against a
nearest-point value, and CLAUDE.md is explicit that it must not: "Observations go through
the same weight table, so both sides of the bust label are area means over the same
polygon. There is exactly one weight table. Do not write a second one."

So this fetches the 4,902 grid cells the weight table covers - not 666 district centroids,
which would be a point sample wearing a district's name - and aggregates them with the
same `DistrictGridAggregator` the forecasts use.

Why this is affordable, measured against the real API on 2026-09-23: Open-Meteo accepts
comma-separated coordinates and returns one object per location. 300 real weight-table
cells, two days, all nine hourly variables including
total_column_integrated_water_vapour, came back HTTP 200 in 2.0 s and 1,139 KB. 4,902
cells is 17 such batches, about 33 s plus polite gaps for one day. CLAUDE.md's "80 of
4,902 cells in 3.5 hours, ~215 hours for one year" is a statement about backfilling
history one request per cell, not about a daily ingest, and was read as the latter for a
while - including by me.

No network here. Every request is stubbed; what is under test is the batching, the
aggregation and the wind handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.live import observations as obs
from app.utils.india_districts import get_aggregator


def test_the_city_table_is_gone_from_the_observation_path():
    assert not hasattr(obs, "load_cities")
    assert not hasattr(obs, "CITIES_JSON")


def test_cells_come_from_the_weight_table_not_from_centroids():
    """666 centroids would be a point sample; the weight table's cells are the field."""
    lats, lons = obs._cells()
    assert len(lats) == len(lons) == 4902, (
        f"expected the weight table's 4,902 cells, got {len(lats)}")
    # A centroid table would have one row per district and would be caught here.
    assert len(lats) != len(get_aggregator().region_ids)


def test_every_cell_is_requested_exactly_once_across_batches(monkeypatch):
    """A dropped batch is a silently smaller country, so account for all of them."""
    seen_lat: list = []
    calls = {"n": 0}

    def fake_batch(lats, lons, start, end, tier):
        calls["n"] += 1
        seen_lat.extend(np.asarray(lats).tolist())
        return [_hourly(15.0) for _ in lats]

    monkeypatch.setattr(obs, "_fetch_cell_batch", fake_batch)
    monkeypatch.setattr(obs, "POLITE_GAP_S", 0)
    frame, report = obs.fetch_observations(_D, _D, tier="final")

    all_lat, _ = obs._cells()
    assert len(seen_lat) == len(all_lat)
    assert np.allclose(sorted(seen_lat), sorted(all_lat.tolist()))
    assert calls["n"] == -(-len(all_lat) // obs.CELL_BATCH)
    assert report.cells == len(all_lat)


def test_a_constant_field_becomes_that_constant_per_district(monkeypatch):
    """The invariant that survives any weighting, so it catches a normalisation bug."""
    monkeypatch.setattr(obs, "_fetch_cell_batch",
                        lambda lats, lons, s, e, t: [_hourly(21.5) for _ in lats])
    monkeypatch.setattr(obs, "POLITE_GAP_S", 0)
    frame, report = obs.fetch_observations(_D, _D, tier="final")

    assert not frame.empty
    assert set(["region_id", "region_name", "state_id", "state_name",
                "latitude", "longitude", "date", "source"]).issubset(frame.columns)
    assert frame["region_id"].nunique() > 600
    assert np.allclose(frame["t2m_c"].dropna().to_numpy(), 21.5)


def test_wind_is_aggregated_as_components_not_as_degrees(monkeypatch):
    """350 degrees and 10 degrees average to 0, not to 180.

    Averaging a bearing in degrees is the classic circular-mean bug, and it would land in
    wind_direction_deg - a variable whose errors are already the widest in the model. The
    forecast side aggregates u and v per district and derives speed and direction from the
    aggregated components; this asserts the observation side does the same, because the
    two sides have to be commensurable to be subtracted.
    """
    def fake_batch(lats, lons, start, end, tier):
        # Half the cells at 350 deg, half at 10 deg, all at the same speed.
        return [_hourly(20.0, wind_dir=350.0 if i % 2 == 0 else 10.0, wind_speed=5.0)
                for i, _ in enumerate(lats)]

    monkeypatch.setattr(obs, "_fetch_cell_batch", fake_batch)
    monkeypatch.setattr(obs, "POLITE_GAP_S", 0)
    frame, _ = obs.fetch_observations(_D, _D, tier="final")

    d = frame["wdir10m_deg"].dropna().to_numpy()
    assert len(d) > 500
    # Near 0/360, never near 180. Compared as a circular distance from 0.
    off = np.minimum(np.abs(d - 0.0), np.abs(d - 360.0))
    assert off.max() < 15.0, f"direction drifted to {d[np.argmax(off)]:.1f} - averaged in degrees?"
    assert np.allclose(frame["wspd10m_ms"].dropna().to_numpy(), 5.0, atol=0.5)


def test_a_failed_batch_is_reported_and_does_not_fabricate_values(monkeypatch):
    """Refuse rather than patch: a lost batch must not become zeros (rule 3)."""
    def fake_batch(lats, lons, start, end, tier):
        if lats[0] == obs._cells()[0][0]:
            raise RuntimeError("HTTP 429 from the archive API")
        return [_hourly(12.0) for _ in lats]

    monkeypatch.setattr(obs, "_fetch_cell_batch", fake_batch)
    monkeypatch.setattr(obs, "POLITE_GAP_S", 0)
    frame, report = obs.fetch_observations(_D, _D, tier="final")
    assert report.failures, "a failed batch must be recorded"
    if not frame.empty:
        assert np.allclose(frame["t2m_c"].dropna().to_numpy(), 12.0), (
            "districts built from the surviving batches only; nothing invented")


def test_a_short_day_is_dropped_rather_than_averaged(monkeypatch):
    """Fewer than 24 hours is an incomplete day, and a partial mean is a wrong number."""
    monkeypatch.setattr(obs, "_fetch_cell_batch",
                        lambda lats, lons, s, e, t: [_hourly(18.0, hours=11) for _ in lats])
    monkeypatch.setattr(obs, "POLITE_GAP_S", 0)
    frame, _ = obs.fetch_observations(_D, _D, tier="final")
    assert frame.empty, "an 11-hour day must not produce a daily observation"


# --------------------------------------------------------------------------- helpers

_D = pd.Timestamp("2026-09-15").date()


def _hourly(temp: float, wind_dir: float = 90.0, wind_speed: float = 3.0,
            hours: int = 24) -> dict:
    """One location's hourly block, shaped as Open-Meteo returns it.

    Synthetic and labelled as such: it exercises shape, batching and the wind maths. The
    values carry no meaning and nothing here produces a metric.
    """
    times = pd.date_range(f"{_D}T00:00", periods=hours, freq="h")
    return {
        "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
        "temperature_2m": [temp] * hours,
        "relative_humidity_2m": [60.0] * hours,
        "precipitation": [0.0] * hours,
        "pressure_msl": [1008.0] * hours,
        "surface_pressure": [1000.0] * hours,
        "wind_speed_10m": [wind_speed] * hours,
        "wind_direction_10m": [wind_dir] * hours,
        "soil_moisture_0_to_7cm": [0.25] * hours,
        "total_column_integrated_water_vapour": [45.0] * hours,
    }
