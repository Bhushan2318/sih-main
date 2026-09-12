"""IMD gridded rainfall -> district precip_mm, merged into an existing ERA5-family
observation file.

Why merge rather than write a standalone file: IMD only publishes rain (and tmin/tmax,
which this project does not use it for - CLAUDE.md's own plan is "IMD ... for
precipitation, ERA5 ... for everything else"). A precip_mm-only file would either fail
`ingest_upload`'s completeness expectations or silently drop the other eight canonical
variables. Replacing just the precip_mm column of an already-fetched CDS/ERA5 file keeps
every other variable untouched and lets `ingest_backfill.py` treat this exactly like any
other observation source.

The grid arrays below are small hand-built stand-ins for IMD's real (365/366, 135, 129)
array - shape and masking logic only, never a value that reaches a metric.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fetch_imd_district_rainfall", BACKEND / "scripts" / "fetch_imd_district_rainfall.py")
fir = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fir
_spec.loader.exec_module(fir)


class _FakeIMD:
    """Stands in for an imdlib.IMD object: same attribute names, a tiny synthetic grid."""

    def __init__(self, data, lat_array, lon_array, start_day):
        self.data = data  # shape (days, lon, lat), IMD's own axis order
        self.lat_array = lat_array
        self.lon_array = lon_array
        self.start_day = start_day


def _tiny_imd(n_days=3):
    # 2 lon x 2 lat, one cell always missing (-999), one cell real rain.
    lon = np.array([70.0, 70.25])
    lat = np.array([20.0, 20.25])
    data = np.full((n_days, 2, 2), -999.0)
    data[:, 0, 0] = [0.0, 5.5, 12.3]  # (lon=70.0, lat=20.0): real values
    # (lon=70.25, lat=20.0) and both lat=20.25 cells stay -999 (sea / no data)
    return _FakeIMD(data, lat, lon, "2018-01-01")


def test_imd_grid_to_long_masks_the_missing_flag_and_keeps_real_values():
    obj = _tiny_imd(n_days=3)
    long = fir.imd_to_long(obj)
    assert set(long.columns) >= {"lat", "lon", "date", "precip_mm"}
    real = long[(long.lat == 20.0) & (long.lon == 70.0)].sort_values("date")
    assert list(real["precip_mm"]) == [0.0, 5.5, 12.3]
    missing = long[(long.lat == 20.25) & (long.lon == 70.25)]
    assert missing["precip_mm"].isna().all()


def test_imd_grid_to_long_dates_advance_from_start_day():
    obj = _tiny_imd(n_days=3)
    long = fir.imd_to_long(obj)
    got = sorted(long[(long.lat == 20.0) & (long.lon == 70.0)]["date"].unique())
    assert list(pd.to_datetime(got)) == list(pd.date_range("2018-01-01", periods=3))


def test_merge_precip_replaces_only_that_column():
    base = pd.DataFrame({
        "region_id": ["d1", "d1", "d2"],
        "region_name": ["D1", "D1", "D2"],
        "state_id": ["s1", "s1", "s2"],
        "state_name": ["S1", "S1", "S2"],
        "latitude": [20.0, 20.0, 21.0],
        "longitude": [70.0, 70.0, 71.0],
        "date": pd.to_datetime(["2018-01-01", "2018-01-02", "2018-01-01"]).date,
        "t2m_c": [25.0, 26.0, 24.0],
        "precip_mm": [1.0, 2.0, 3.0],  # ERA5's own rainfall - must be overwritten
        "source": ["ERA5 hourly single levels via Copernicus CDS"] * 3,
    })
    imd_districts = pd.DataFrame({
        "region_id": ["d1", "d1", "d2"],
        "date": pd.to_datetime(["2018-01-01", "2018-01-02", "2018-01-01"]).date,
        "precip_mm": [10.0, 20.0, 30.0],
    })
    merged = fir.merge_precip(base, imd_districts)
    assert list(merged["precip_mm"]) == [10.0, 20.0, 30.0]
    # Every other column is untouched.
    assert list(merged["t2m_c"]) == [25.0, 26.0, 24.0]
    assert (merged["source"] != base["source"]).any(), (
        "source column should record that precip_mm came from IMD, not silently keep "
        "the ERA5 attribution on a column ERA5 no longer supplied")


def test_merge_precip_leaves_a_row_nan_when_imd_has_no_value_for_it():
    """A district-date IMD could not cover must not fall back to ERA5's own number -
    that would silently blend two rainfall products under one column with no record of
    which one produced which row."""
    base = pd.DataFrame({
        "region_id": ["d1"], "date": [pd.Timestamp("2018-01-01").date()],
        "precip_mm": [1.0], "source": ["ERA5"],
    })
    imd_districts = pd.DataFrame({
        "region_id": ["d1"], "date": [pd.Timestamp("2018-01-01").date()],
        "precip_mm": [np.nan],
    })
    merged = fir.merge_precip(base, imd_districts)
    assert pd.isna(merged["precip_mm"].iloc[0])


def test_merge_precip_refuses_a_district_date_imd_does_not_cover_at_all():
    """IMD covering fewer district-dates than the base file is a real gap, not a row to
    drop silently - refuse rather than publish a file quietly narrower than it claims."""
    base = pd.DataFrame({
        "region_id": ["d1", "d2"], "date": [pd.Timestamp("2018-01-01").date()] * 2,
        "precip_mm": [1.0, 2.0], "source": ["ERA5", "ERA5"],
    })
    imd_districts = pd.DataFrame({
        "region_id": ["d1"], "date": [pd.Timestamp("2018-01-01").date()], "precip_mm": [10.0],
    })
    with pytest.raises(ValueError, match="d2"):
        fir.merge_precip(base, imd_districts)
