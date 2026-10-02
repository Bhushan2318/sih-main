"""ERA5 per-cell daily values built the way the forecast side builds its own.

The bust label differences a GEFS forecast against an ERA5 observation. If the two sides
reduce hours and cells to one district-day differently, part of every error is method,
not forecast. Measured on Nov 2017 (666 districts): computing RH per hour and per cell on
the observation side, while the forecast side derives it from district-mean daily q, T and
p, put the observation +1.28 %RH higher on average and flipped 4.4% of humidity labels.

So the v2 observation path keeps the *components* per cell and per day - q, T, surface
pressure, u, v - sampled at the forecast's own instants, and leaves every non-linear
derivation (RH, wind speed and direction) until after the district mean, exactly as
`scripts/fetch_gefs_reforecast_sample.py` does.

Not marked importorskip("xarray"): nothing here opens a NetCDF, so these run in CI.
Inputs are small hand-built hourly frames for arithmetic only; no metric is produced.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, BACKEND / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cds = _load("fetch_era5_cds_district_observations")


def _hourly(start: str, hours: int, **cols) -> pd.DataFrame:
    t = pd.date_range(start, periods=hours, freq="h")
    base = {name: np.zeros(hours) for name in cds.CDS_VARS}
    base["2m_temperature"] = np.full(hours, 300.0)
    base["2m_dewpoint_temperature"] = np.full(hours, 290.0)
    base["surface_pressure"] = np.full(hours, 95000.0)
    base.update(cols)
    df = pd.DataFrame(base)
    df["time"] = t
    df["lat"] = 20.0
    df["lon"] = 78.0
    return df


# --------------------------------------------------------------- one humidity formula

def test_both_fetches_use_one_humidity_module():
    """The forecast side's RH and wind helpers and the observation side's are the same
    functions, not two copies that can drift."""
    from app.utils import humidity
    gefs = _load("fetch_gefs_reforecast_sample")
    assert gefs.rh_from_specific_humidity is humidity.rh_from_specific_humidity
    assert gefs.wind_speed_dir is humidity.wind_speed_dir


def test_q_from_dewpoint_round_trips_through_the_forecast_rh_formula():
    """q derived from (Td, p) and fed to the forecast side's RH formula gives exactly
    es(Td)/es(T): the two sides share one saturation formula (Bolton 1980)."""
    from app.utils import humidity
    t = np.array([301.0, 285.0, 310.0])
    td = np.array([294.0, 280.0, 290.0])
    p = np.array([95000.0, 101000.0, 70000.0])
    q = humidity.specific_humidity_from_dewpoint(td, p)
    rh = humidity.rh_from_specific_humidity(q, t, p)
    es = humidity.saturation_vapour_pressure_pa
    assert rh == pytest.approx(100.0 * es(td) / es(t), rel=1e-12)
    assert rh == pytest.approx(cds.rh_from_dewpoint(t, td), rel=1e-12)


# --------------------------------------------------------------- the forecast's instants

def test_instantaneous_fields_use_the_forecasts_eight_three_hourly_instants():
    """Day k of a forecast is the mean of its 3-hourly output at hours 3, 6, ..., 24 of
    that day. The observation day is sampled at the same eight instants, not all 24."""
    stamps = pd.date_range("2017-01-01 01:00", periods=24, freq="h")
    t2m = 280.0 + np.arange(24, dtype=float)          # a different value at every hour
    df = _hourly("2017-01-01 01:00", 24, **{"2m_temperature": t2m})
    row = cds.to_cells_daily(df).iloc[0]
    wanted = [h % 3 == 0 for h in stamps.hour]          # 03, 06, ..., 21 and 00 next day
    assert sum(wanted) == 8
    assert row["t2m_k"] == pytest.approx(t2m[wanted].mean())
    assert row["t2m_k"] != pytest.approx(t2m.mean())


def test_rain_still_sums_all_24_hourly_accumulations():
    """Rainfall is an accumulation, not an instant: every hour of the day counts, and the
    hour stamped 00:00 next day closes the day (ERA5 stamps the end of the hour)."""
    tp = np.full(24, 0.001)                              # metres per hour
    df = _hourly("2017-01-01 01:00", 24, total_precipitation=tp)
    row = cds.to_cells_daily(df).iloc[0]
    assert row["date"] == pd.Timestamp("2017-01-01").date()
    assert row["tp_mm"] == pytest.approx(24.0)


def test_specific_humidity_is_averaged_not_relative_humidity():
    """q is linear in what the air holds, so its daily mean is meaningful; RH is not
    averaged here at all. The day's q is the mean of hourly q from Td and p."""
    from app.utils import humidity
    td = np.where(np.arange(24) % 2 == 0, 295.0, 285.0)
    df = _hourly("2017-01-01 01:00", 24, **{"2m_dewpoint_temperature": td})
    row = cds.to_cells_daily(df).iloc[0]
    q = humidity.specific_humidity_from_dewpoint(td, np.full(24, 95000.0))
    stamps = pd.date_range("2017-01-01 01:00", periods=24, freq="h")
    wanted = np.array([h % 3 == 0 for h in stamps.hour])
    assert row["q2m_kgkg"] == pytest.approx(q[wanted].mean())
    assert "rh2m_pct" not in row.index and "wspd10m_ms" not in row.index


def test_wind_stays_as_components():
    """Opposing winds in two cells must cancel in the district mean, as they do on the
    forecast side, so the per-cell output keeps u and v - never a speed."""
    u = np.full(24, 5.0)
    df = _hourly("2017-01-01 01:00", 24, **{"10m_u_component_of_wind": u})
    row = cds.to_cells_daily(df).iloc[0]
    assert row["u10_ms"] == pytest.approx(5.0)
    assert row["v10_ms"] == pytest.approx(0.0)


def test_units_are_si_and_rain_is_millimetres():
    df = _hourly("2017-01-01 01:00", 24,
                 **{"mean_sea_level_pressure": np.full(24, 101000.0),
                    "volumetric_soil_water_layer_1": np.full(24, 0.3),
                    "total_column_water_vapour": np.full(24, 40.0)})
    row = cds.to_cells_daily(df).iloc[0]
    assert row["sp_pa"] == pytest.approx(95000.0)
    assert row["msl_pa"] == pytest.approx(101000.0)
    assert row["swvl1_m3m3"] == pytest.approx(0.3)
    assert row["tcwv_kgm2"] == pytest.approx(40.0)


def test_incomplete_days_are_dropped_not_averaged_short():
    df = _hourly("2017-01-01 01:00", 29)
    out = cds.to_cells_daily(df)
    assert list(out["date"]) == [pd.Timestamp("2017-01-01").date()]


def test_output_columns_are_exactly_the_cell_daily_contract():
    out = cds.to_cells_daily(_hourly("2017-01-01 01:00", 24))
    assert list(out.columns) == ["lat", "lon", "date"] + cds.CELL_DAILY_COLUMNS


# --------------------------------------------------------------- refusing a bad month

def _month(n_days: int, cells: pd.DataFrame) -> pd.DataFrame:
    dates = pd.date_range("2017-02-01", periods=n_days, freq="D").date
    rows = [(la, lo, d) for d in dates for la, lo in zip(cells.lat, cells.lon)]
    df = pd.DataFrame(rows, columns=["lat", "lon", "date"])
    for c in cds.CELL_DAILY_COLUMNS:
        df[c] = 1.0
    return df


def test_a_complete_month_passes_the_check():
    cells = pd.DataFrame({"lat": [20.0, 20.25], "lon": [78.0, 78.0]})
    cds.check_cells_daily(_month(28, cells), 2017, 2, cells)


def test_a_month_missing_a_day_is_refused():
    cells = pd.DataFrame({"lat": [20.0, 20.25], "lon": [78.0, 78.0]})
    with pytest.raises(RuntimeError, match="days"):
        cds.check_cells_daily(_month(27, cells), 2017, 2, cells)


def test_a_month_missing_a_cell_is_refused():
    cells = pd.DataFrame({"lat": [20.0, 20.25, 20.5], "lon": [78.0, 78.0, 78.0]})
    df = _month(28, cells.iloc[:2])
    with pytest.raises(RuntimeError, match="cell"):
        cds.check_cells_daily(df, 2017, 2, cells)


def test_a_nan_outside_soil_is_refused():
    """Soil can be NaN over water; nothing else may be missing - missing never becomes a
    plausible number further down."""
    cells = pd.DataFrame({"lat": [20.0, 20.25], "lon": [78.0, 78.0]})
    df = _month(28, cells)
    df.loc[3, "swvl1_m3m3"] = np.nan
    cds.check_cells_daily(df, 2017, 2, cells)            # soil NaN is allowed
    df.loc[5, "t2m_k"] = np.nan
    with pytest.raises(RuntimeError, match="t2m_k"):
        cds.check_cells_daily(df, 2017, 2, cells)


# --------------------------------------------------------------- the month loop

def _hourly_month(cells: pd.DataFrame, year: int, month: int, drop_last_hour=False):
    start = pd.Timestamp(year=year, month=month, day=1, hour=1)
    end = (start + pd.offsets.MonthBegin(1)).replace(hour=0)       # boundary hour included
    times = pd.date_range(start, end, freq="h")
    if drop_last_hour:
        times = times[:-1]
    frames = [_hourly(str(times[0]), len(times)).assign(lat=la, lon=lo)
              for la, lo in zip(cells.lat, cells.lon)]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def v2_dirs(tmp_path, monkeypatch):
    cells = pd.DataFrame({"lat": [20.0, 20.25], "lon": [78.0, 78.0]})
    monkeypatch.setattr(cds, "OUT_DIR", tmp_path)
    monkeypatch.setattr(cds, "V2_DIR", tmp_path / "_era5_cds_v2")
    monkeypatch.setattr(cds, "grid_cells", lambda: cells)
    (tmp_path / "_era5_cds").mkdir()
    return tmp_path, cells


def test_a_good_month_is_written_and_its_downloads_deleted(v2_dirs):
    root, cells = v2_dirs
    z = root / "_era5_cds" / "era5_201702_0.zip"
    z.write_bytes(b"stand-in")
    cds.build_cells_daily([2017], [2], fetch=lambda y, m: _hourly_month(cells, y, m))
    out = pd.read_parquet(cds.cells_daily_path(2017, 2))
    assert out["date"].nunique() == 28 and len(out) == 28 * len(cells)
    assert not z.exists(), "the download is deleted once the month is safely written"


def test_a_refused_month_writes_nothing_and_keeps_its_downloads(v2_dirs):
    root, cells = v2_dirs
    z = root / "_era5_cds" / "era5_201702_0.zip"
    z.write_bytes(b"stand-in")
    with pytest.raises(RuntimeError, match="days"):
        cds.build_cells_daily([2017], [2], fetch=lambda y, m: _hourly_month(
            cells, y, m, drop_last_hour=True))         # last day short of its 00:00 stamp
    assert not cds.cells_daily_path(2017, 2).exists()
    assert z.exists(), "a refused month keeps its download for the retry"


def test_a_finished_month_is_not_fetched_again(v2_dirs):
    root, cells = v2_dirs
    cds.build_cells_daily([2017], [2], fetch=lambda y, m: _hourly_month(cells, y, m))

    def boom(y, m):
        raise AssertionError("refetched a month that is already on disk")
    cds.build_cells_daily([2017], [2], fetch=boom)
