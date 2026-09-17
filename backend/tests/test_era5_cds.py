"""The CDS ERA5 fetch - the parts that are wrong *silently*.

The network half is proved by running the script against the real API. What is pinned
here is the arithmetic, because every one of these mistakes produces numbers of the right
shape, in the right range, in the right columns:

  - ERA5 `total_precipitation` at hour t is the accumulation over the hour ENDING at t.
    Grouping by floor("D") shifts a day's rainfall by one hour. Rainfall drives most
    busts, and this is the same genre of error as the valid_date offset in CLAUDE.md.
  - ERA5 has no 2 m relative humidity, so rh2m_pct is derived from dewpoint.
  - Units: K, Pa, m and m3/m3 all differ from the canonical columns.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xarray")  # the script this module loads imports xarray at collection
# time - requirements-live.txt only, not part of the core test-suite install (see the two
# individual importorskip calls below, which this makes redundant but harmless to leave).

BACKEND = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fetch_era5_cds_district_observations",
    BACKEND / "scripts" / "fetch_era5_cds_district_observations.py")
cds = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cds
_spec.loader.exec_module(cds)


# --------------------------------------------------------------- request geometry

def test_request_area_covers_every_weight_table_cell():
    """The area must contain the whole weight table or districts silently lose cells."""
    from app.utils import india_districts as idist
    w = pd.read_parquet(idist.geo_dir() / idist.WEIGHTS_FILENAME)
    north, west, south, east = cds.request_area()
    assert south <= w.lat.min() and north >= w.lat.max()
    assert west <= w.lon.min() and east >= w.lon.max()


def test_request_area_sits_on_the_quarter_degree_grid():
    """Off-grid corners make ERA5 interpolate, which is the bug we are removing."""
    for edge in cds.request_area():
        assert abs(edge / 0.25 - round(edge / 0.25)) < 1e-9, f"{edge} is off-grid"


# --------------------------------------------------------------- humidity

def test_rh_is_100_percent_when_dewpoint_equals_temperature():
    t = np.array([300.0, 280.0, 310.0])
    assert cds.rh_from_dewpoint(t, t) == pytest.approx(100.0)


def test_rh_falls_as_dewpoint_drops_and_stays_in_range():
    t = np.full(4, 303.15)
    td = np.array([303.15, 295.0, 285.0, 250.0])
    rh = cds.rh_from_dewpoint(t, td)
    assert np.all(np.diff(rh) < 0), "drier air must read lower"
    assert rh.min() >= 0.0 and rh.max() <= 100.0


def test_rh_uses_the_same_saturation_formula_as_the_forecast_side():
    """Bolton (1980), identical constants to rh_from_specific_humidity in the GEFS
    fetch. If the two sides used different es formulas the humidity bust label would
    partly record that difference."""
    src = (BACKEND / "scripts" / "fetch_gefs_reforecast_sample.py").read_text()
    assert "611.2 * np.exp(17.67 * (t_k - 273.15) / (t_k - 29.65))" in src
    t = np.array([301.0])
    td = np.array([294.0])
    es = lambda k: 611.2 * np.exp(17.67 * (k - 273.15) / (k - 29.65))  # noqa: E731
    assert cds.rh_from_dewpoint(t, td) == pytest.approx(100.0 * es(td) / es(t))


# --------------------------------------------------------------- daily collapse

def _hourly(start: str, hours: int, **cols) -> pd.DataFrame:
    t = pd.date_range(start, periods=hours, freq="h", tz=None)
    base = {name: np.zeros(hours) for name in cds.CDS_VARS}
    base.update(cols)
    df = pd.DataFrame(base)
    df["time"] = t
    df["lat"] = 20.0
    df["lon"] = 78.0
    return df


def test_precipitation_is_attributed_to_the_hour_it_fell_in():
    """1 mm at 00:00 on the 2nd belongs to the 1st: ERA5 stamps an accumulation with the
    END of its hour, so that value covers 23:00-24:00 on the 1st."""
    n = 48
    tp = np.zeros(n)
    tp[23] = 0.001            # metres, stamped 2017-01-02 00:00 = the 1st's last hour
    df = _hourly("2017-01-01 01:00", n, total_precipitation=tp)
    daily = cds.to_daily(df)
    by_date = daily.set_index("date")["precip_mm"]
    assert by_date[pd.Timestamp("2017-01-01").date()] == pytest.approx(1.0)
    assert by_date.get(pd.Timestamp("2017-01-02").date(), 0.0) == pytest.approx(0.0)


def test_a_full_day_of_rain_sums_over_exactly_24_hours():
    df = _hourly("2017-01-01 01:00", 24, total_precipitation=np.full(24, 0.002))
    daily = cds.to_daily(df)
    assert len(daily) == 1
    assert daily.iloc[0]["precip_mm"] == pytest.approx(48.0)


def test_units_are_converted_to_canonical():
    n = 24
    df = _hourly("2017-01-01 01:00", n,
                 **{"2m_temperature": np.full(n, 300.15),
                    "surface_pressure": np.full(n, 95000.0),
                    "mean_sea_level_pressure": np.full(n, 101000.0),
                    "volumetric_soil_water_layer_1": np.full(n, 0.32)})
    row = cds.to_daily(df).iloc[0]
    assert row["t2m_c"] == pytest.approx(27.0)            # K -> C
    assert row["psfc_hpa"] == pytest.approx(950.0)        # Pa -> hPa
    assert row["mslp_hpa"] == pytest.approx(1010.0)
    assert row["soil_moisture_pct"] == pytest.approx(32.0)  # m3/m3 -> %


def test_wind_is_averaged_as_vectors_not_as_angles():
    """Two opposing hours average to a light, not a 12 m/s, wind - and the direction
    convention matches (270 - atan2(v, u)) % 360 used on the forecast side."""
    n = 24
    u = np.full(n, 0.0)
    v = np.full(n, 0.0)
    u[:12] = 12.0
    u[12:] = -12.0
    df = _hourly("2017-01-01 01:00", n,
                 **{"10m_u_component_of_wind": u, "10m_v_component_of_wind": v})
    row = cds.to_daily(df).iloc[0]
    assert row["wspd10m_ms"] == pytest.approx(0.0, abs=1e-9)

    df2 = _hourly("2017-01-01 01:00", n,
                  **{"10m_u_component_of_wind": np.full(n, 5.0),
                     "10m_v_component_of_wind": np.zeros(n)})
    row2 = cds.to_daily(df2).iloc[0]
    assert row2["wspd10m_ms"] == pytest.approx(5.0)
    assert row2["wdir10m_deg"] == pytest.approx(270.0)   # wind FROM the west


def test_every_canonical_column_is_produced():
    df = _hourly("2017-01-01 01:00", 24)
    daily = cds.to_daily(df)
    for col in cds.VALUE_COLUMNS:
        assert col in daily.columns, f"{col} missing"


def test_partial_days_are_dropped_not_silently_under_summed():
    """A day with 5 hours of data must not report 5 hours of rain as a daily total."""
    df = _hourly("2017-01-01 01:00", 29, total_precipitation=np.full(29, 0.001))
    daily = cds.to_daily(df)
    assert set(daily["date"]) == {pd.Timestamp("2017-01-01").date()}, \
        "the incomplete second day must be refused, not truncated"


# --------------------------------------------------------------- one aggregator only

def test_both_fetches_share_one_aggregator():
    """CLAUDE.md: there is exactly one weight table. Both observation fetches must reach
    the districts through the same code, not two copies of it."""
    from app.utils import district_observations as shared
    _s2 = importlib.util.spec_from_file_location(
        "fetch_era5_district_observations",
        BACKEND / "scripts" / "fetch_era5_district_observations.py")
    om = importlib.util.module_from_spec(_s2)
    sys.modules[_s2.name] = om
    _s2.loader.exec_module(om)
    assert om.to_districts is shared.to_districts
    assert cds.to_districts is shared.to_districts


# --------------------------------------------------------------- archive shape

def test_a_single_timestep_keeps_valid_time_a_dimension():
    """The month-boundary pull requests exactly one hour, and `squeeze(drop=True)` would
    delete `valid_time` along with the other length-1 dimensions - the merge then fails
    with KeyError('valid_time'). Found at real volume on 2017-11 after twelve green
    tests; a two-hour probe cannot express it.

    Shape/plumbing test: the values are arbitrary, the dimensions are the point.
    """
    xr = pytest.importorskip("xarray")
    ds = xr.Dataset(
        {"t2m": (("valid_time", "latitude", "longitude"), np.zeros((1, 2, 2)))},
        coords={"valid_time": pd.to_datetime(["2017-12-01T00:00"]),
                "latitude": [21.0, 20.75], "longitude": [78.0, 78.25],
                "number": 0, "expver": "0001"},
    )
    out = cds._normalise(ds)
    assert "valid_time" in out.dims, "a one-hour request must keep its time dimension"
    assert "number" not in out.coords and "expver" not in out.coords
    df = out.to_dataframe().reset_index()
    assert "valid_time" in df.columns


def test_normalise_leaves_a_multi_hour_dataset_alone():
    xr = pytest.importorskip("xarray")
    ds = xr.Dataset(
        {"t2m": (("valid_time", "latitude", "longitude"), np.zeros((3, 2, 2)))},
        coords={"valid_time": pd.to_datetime(
            ["2017-12-01T00:00", "2017-12-01T01:00", "2017-12-01T02:00"]),
            "latitude": [21.0, 20.75], "longitude": [78.0, 78.25]},
    )
    assert cds._normalise(ds).sizes["valid_time"] == 3
