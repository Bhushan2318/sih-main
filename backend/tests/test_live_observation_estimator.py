"""Live observations use the training side's estimator, on ERA5 itself.

Measured against the real archive API on 2026-10-04 (four 0.25 deg cells, 2026-09-20):
- Without `models=era5` Open-Meteo answers from its default model mix, snapped to a finer
  grid: 19.0 N 72.75 E came back from 19.016 N 72.781 E, at 01 UTC on 2026-09-20 with 2 m
  temperature 24.7 C and column water vapour 55.7 kg/m2, where ERA5's own cell reads 26.6 C
  and 45.0. With
  `models=era5` it answers from ERA5's 0.25 deg cell itself, at exactly the coordinates
  asked. The live rows were labelled ERA5 without being ERA5.
- `elevation` defaults to a 90 m terrain model, and Open-Meteo shifts temperature to it;
  `elevation=nan` turns that off, so the value is the cell's own.
- The day was the 00-23 UTC stamps. Open-Meteo stamps an accumulation at the end of its
  hour, so 00 UTC rain fell the evening before; the training side's day is (t-24h, t],
  the forecast's own (k-1)*24 < h <= k*24.

The training estimator (`to_cells_daily` + `to_districts_v2`): instantaneous fields are the
mean of the eight 3-hourly instants 03-24 UTC, rain sums all 24 hours, q comes from the
dewpoint, RH and wind are derived after the district mean, soil uses cells ERA5 calls land.

SYNTHETIC, LABELLED: hourly blocks are generated in the shape Open-Meteo returns, to check
the arithmetic. No metric is produced from them.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from app.live import observations as obs
from app.utils import district_observations as dobs
from app.utils import humidity
from app.utils.india_districts import geo_dir, get_aggregator, WEIGHTS_FILENAME

_D = pd.Timestamp("2026-09-15").date()


def _block(hours=48, start=None, **over) -> dict:
    """One location's hourly block from `start` (default 00 UTC on _D), Open-Meteo units."""
    start = start or pd.Timestamp(f"{_D}T00:00")
    times = pd.date_range(start, periods=hours, freq="h")
    base = {
        "temperature_2m": 25.0, "dew_point_2m": 18.0, "precipitation": 0.0,
        "pressure_msl": 1008.0, "surface_pressure": 950.0, "wind_speed_10m": 3.0,
        "wind_direction_10m": 90.0, "soil_moisture_0_to_7cm": 0.25,
        "total_column_integrated_water_vapour": 45.0,
    }
    out = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
    for k, v in base.items():
        v = over.get(k, v)
        out[k] = [float(v(t)) if callable(v) else float(v) for t in times]
    return out


@pytest.fixture
def stub(monkeypatch):
    """Replace the network with a per-cell block maker; returns the request log."""
    calls = []

    def install(make=lambda i, lat, lon: _block(), fail=lambda lats: False):
        def fake(lats, lons, start, end, tier):
            calls.append({"n": len(lats), "start": start, "end": end, "tier": tier})
            if fail(lats):
                raise RuntimeError("HTTP 429 from the archive API")
            blocks = [make(i, la, lo) for i, (la, lo) in enumerate(zip(lats, lons))]
            if tier == "provisional":       # not requested from the forecast API
                for b in blocks:
                    b.pop("total_column_integrated_water_vapour", None)
            return blocks
        monkeypatch.setattr(obs, "_fetch_cell_batch", fake)
        monkeypatch.setattr(obs, "POLITE_GAP_S", 0)
        return calls
    return install


# ------------------------------------------------------------------ the request

def test_the_final_tier_asks_for_era5_itself_without_terrain_adjustment(monkeypatch):
    seen = {}

    def fake_get(url, params):
        seen.update(url=url, params=params)
        return [{"hourly": {}}] * 3
    monkeypatch.setattr(obs, "_get_json", fake_get)
    obs._fetch_cell_batch(np.array([19.0, 18.75, 8.25]), np.array([72.75, 73.0, 73.0]),
                          _D, _D, "final")
    p = seen["params"]
    assert seen["url"] == obs.ARCHIVE_URL
    assert p["models"] == "era5"
    assert p["elevation"] == "nan,nan,nan", "one elevation per coordinate"
    assert p["cell_selection"] == "nearest"
    hourly = p["hourly"].split(",")
    assert "dew_point_2m" in hourly and "relative_humidity_2m" not in hourly
    assert p["start_date"] == _D.isoformat()
    assert p["end_date"] == (_D + timedelta(days=1)).isoformat(), \
        "the day ends at 00 UTC the next day, so that stamp must be asked for"


def test_the_provisional_tier_also_turns_off_terrain_adjustment(monkeypatch):
    seen = {}
    monkeypatch.setattr(obs, "_get_json",
                        lambda url, params: seen.update(url=url, params=params) or [{}, {}])
    obs._fetch_cell_batch(np.array([19.0, 18.75]), np.array([72.75, 73.0]), _D, _D,
                          "provisional")
    p = seen["params"]
    assert seen["url"] == obs.FORECAST_URL
    assert p["elevation"] == "nan,nan"
    assert "models" not in p, "ERA5 has no near-real-time feed; this tier is labelled so"
    assert p["forecast_days"] >= 1, "today's 00 UTC closes yesterday"


# ------------------------------------------------------------------ one cell's day

def test_the_day_is_01_to_24_utc_and_rain_is_summed_over_it():
    """Rain stamped 00 UTC on _D fell on the day before; 00 UTC on _D+1 closes _D."""
    def rain(t):
        if t == pd.Timestamp(f"{_D}T00:00"):
            return 50.0
        if t == pd.Timestamp(f"{_D}T00:00") + pd.Timedelta(days=1):
            return 7.0
        return 1.0 if t.date() == _D else 0.0
    day = obs._daily_for_batch([_block(precipitation=rain)])
    assert set(day) == {_D}, "only complete days: 00 UTC on _D alone and 01-23 on _D+1 are not"
    assert day[_D]["tp_mm"][0] == pytest.approx(23 * 1.0 + 7.0)


def test_instantaneous_fields_are_the_eight_3_hourly_instants():
    def temp(t):
        return 30.0 if t.hour % 3 == 0 else -100.0   # off-instant hours must not count
    day = obs._daily_for_batch([_block(temperature_2m=temp)])
    assert day[_D]["t2m_k"][0] == pytest.approx(30.0 + 273.15)


def test_q_comes_from_the_dewpoint_and_surface_pressure():
    day = obs._daily_for_batch([_block(dew_point_2m=20.0, surface_pressure=900.0)])
    want = humidity.specific_humidity_from_dewpoint(20.0 + 273.15, 90000.0)
    assert day[_D]["q2m_kgkg"][0] == pytest.approx(want, rel=1e-12)
    assert day[_D]["sp_pa"][0] == pytest.approx(90000.0)


def test_a_missing_instant_voids_that_cells_day_rather_than_averaging_short():
    def temp(t):
        return np.nan if t == pd.Timestamp(f"{_D}T12:00") else 25.0
    day = obs._daily_for_batch([_block(temperature_2m=temp), _block()])
    assert np.isnan(day[_D]["t2m_k"][0])
    assert day[_D]["t2m_k"][1] == pytest.approx(298.15)


# ------------------------------------------------------------------ districts

def _district_with_cells():
    w = pd.read_parquet(geo_dir() / WEIGHTS_FILENAME)
    sizes = w.groupby("region_id").size()
    rid = sizes[sizes >= 4].index[0]
    cells = w[w.region_id == rid][["lat", "lon", "weight"]].reset_index(drop=True)
    return rid, cells


def test_rh_is_derived_after_the_district_mean(stub):
    """The forecast side's order: average q, T and p over the district, then RH."""
    rid, cells = _district_with_cells()
    mine = {(round(a, 4), round(b, 4)): k for k, (a, b) in
            enumerate(zip(cells.lat, cells.lon))}

    def make(i, lat, lon):
        k = mine.get((round(lat, 4), round(lon, 4)))
        if k is None:
            return _block()
        return _block(temperature_2m=15.0 + 6.0 * k, dew_point_2m=5.0 + 4.0 * k)
    stub(make)
    frame, report = obs.fetch_observations(_D, _D, tier="final")
    got = frame.set_index("region_id").loc[rid, "rh2m_pct"]

    ww = cells.weight.to_numpy() / cells.weight.sum()
    k = np.arange(len(cells))
    t = 15.0 + 6.0 * k + 273.15
    q = humidity.specific_humidity_from_dewpoint(5.0 + 4.0 * k + 273.15, 95000.0)
    want = humidity.rh_from_specific_humidity(np.sum(ww * q), np.sum(ww * t), 95000.0)
    per_cell = np.sum(ww * humidity.rh_from_specific_humidity(q, t, 95000.0))
    assert got == pytest.approx(want, rel=1e-6)
    assert abs(got - per_cell) > 1e-3


def test_a_failed_batch_voids_every_district_it_touches(stub):
    """A district built from part of its area is a wrong number, not a smaller one."""
    lats, lons = obs._cells()
    first = obs.CELL_BATCH
    failed = set(zip(np.round(lats[:first], 4), np.round(lons[:first], 4)))
    stub(fail=lambda la: la[0] == lats[0])
    frame, report = obs.fetch_observations(_D, _D, tier="final")
    assert report.failures

    w = pd.read_parquet(geo_dir() / WEIGHTS_FILENAME)
    touched = {r for r, a, b in zip(w.region_id, np.round(w.lat, 4), np.round(w.lon, 4))
               if (a, b) in failed}
    assert touched, "the first batch must touch some district"
    got = frame.set_index("region_id")
    assert not set(got.index) & touched, "a touched district has no row at all"
    assert len(got) == len(get_aggregator().region_ids) - len(touched)
    assert np.allclose(got["t2m_c"], 25.0)


def test_soil_over_sea_cells_is_left_out(stub, monkeypatch):
    """Open-Meteo reads ERA5 soil as 0.0 over sea; ERA5's own mask leaves those cells out."""
    rid, cells = _district_with_cells()
    sea = (round(cells.lat[0], 4), round(cells.lon[0], 4))
    mask = dobs.era5_land_sea_mask().copy()
    mask["lsm"] = 1.0
    mask.loc[(mask.lat.round(4) == sea[0]) & (mask.lon.round(4) == sea[1]), "lsm"] = 0.0
    monkeypatch.setattr(dobs, "era5_land_sea_mask", lambda: mask)

    def make(i, lat, lon):
        on_sea = (round(lat, 4), round(lon, 4)) == sea
        return _block(soil_moisture_0_to_7cm=0.0 if on_sea else 0.30)
    stub(make)
    frame, _ = obs.fetch_observations(_D, _D, tier="final")
    assert frame.set_index("region_id").loc[rid, "soil_moisture_pct"] == pytest.approx(30.0)


def test_the_vendored_land_sea_mask_covers_every_weight_table_cell():
    mask = dobs.era5_land_sea_mask()
    lats, lons = obs._cells()
    assert len(mask) == len(lats) == 4902
    assert set(zip(mask.lat.round(4), mask.lon.round(4))) == \
        set(zip(np.round(lats, 4), np.round(lons, 4)))
    assert mask["lsm"].between(0, 1).all()


def test_rows_name_the_estimator_and_the_source(stub):
    stub()
    final, _ = obs.fetch_observations(_D, _D, tier="final")
    assert final["source"].str.contains("ERA5").all()
    assert final["source"].str.contains("estimator v2").all()
    prov, _ = obs.fetch_observations(_D, _D, tier="provisional")
    assert prov["source"].str.startswith("Near-real-time").all()
    assert prov["source"].str.contains("not ERA5").all()
    assert "pwat_kgm2" not in prov.columns, "the forecast API has no column water vapour"


# ------------------------------------------------------------------ what Open-Meteo cannot give

def test_surface_pressure_is_not_reported(stub):
    """Open-Meteo's surface_pressure is its MSLP reduced to its own terrain height, not
    ERA5's sp: inverting the barometric formula on 2017-11-15 returned each response's
    `elevation` to within a metre, and it read +0.8 hPa at 629 m and +4.3 at 4,835 m
    above ERA5. Missing, not wrong. It still turns the dewpoint into q, where a 1 hPa
    error moves q by ~0.1%."""
    stub()
    frame, _ = obs.fetch_observations(_D, _D, tier="final")
    assert "psfc_hpa" not in frame.columns
    assert frame["rh2m_pct"].notna().all() and frame["mslp_hpa"].notna().all()


def test_a_minutely_rate_limit_waits_out_the_minute(monkeypatch):
    class R:
        def __init__(self, code, text=""):
            self.status_code, self.text = code, text

        def json(self):
            return {"ok": True}
    answers = iter([R(429, '{"reason":"Minutely API request limit exceeded."}'), R(200)])
    monkeypatch.setattr(obs._session, "get", lambda url, params, timeout: next(answers))
    slept = []
    monkeypatch.setattr(obs.time, "sleep", slept.append)
    assert obs._get_json(obs.ARCHIVE_URL, {}) == {"ok": True}
    assert slept and slept[0] >= 60


def test_an_hourly_or_daily_limit_fails_at_once(monkeypatch):
    """Waiting a minute cannot outlast an hourly allowance; five tries would burn five."""
    class R:
        status_code = 429
        text = '{"reason":"Hourly API request limit exceeded. Please try again in the next hour."}'
    calls = []
    monkeypatch.setattr(obs._session, "get", lambda url, params, timeout: calls.append(1) or R())
    slept = []
    monkeypatch.setattr(obs.time, "sleep", slept.append)
    with pytest.raises(RuntimeError, match="Hourly"):
        obs._get_json(obs.ARCHIVE_URL, {})
    assert len(calls) == 1 and not slept


def test_a_refresh_that_lost_batches_is_recorded_as_partial(session, monkeypatch, tmp_path):
    """Measured on refresh-data.yml 2026-10-04 06:11Z: 5 of 17 batches refused (HTTP 429),
    and the run was recorded 'complete'. It is now 'partial', so the next tick retries."""
    from types import SimpleNamespace

    from app.db.models import IngestRun
    from app.live import orchestrator as orch

    frame = pd.DataFrame({"region_id": ["IN-MH-PUNE"], "date": [_D], "t2m_c": [25.0]})
    report = obs.ObsReport(tier="final", start=_D, end=_D, cells=4902, rows=1,
                           failures=["cells 600-899: HTTP 429"])
    monkeypatch.setattr(orch.observations, "fetch_observations", lambda s, e, tier: (frame, report))
    monkeypatch.setattr(orch.observations, "write_observations_csv",
                        lambda f, tier, s, e, d: tmp_path / "obs.csv")
    monkeypatch.setattr(orch, "ingest_upload", lambda *a, **k: SimpleNamespace(
        batch_id="b1", row_count_ingested=1))
    monkeypatch.setattr(orch.inference, "invalidate_caches", lambda: None)
    monkeypatch.setattr(orch, "emit", lambda *a, **k: None)

    out = orch.run_observation_refresh("final", days_back=1, retrain=False)
    assert out["status"] == "partial"
    assert out["failed_batches"] == 1
    run = session.query(IngestRun).filter(IngestRun.kind == "observations_final").one()
    assert run.status == "partial"
    assert "failed_batches" in (run.detail or "")
