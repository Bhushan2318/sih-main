"""to_districts - the aggregation half of the district observation fetch.

The network half is exercised by running the script; what is pinned here is the logic
that turns cell readings into district values, because a mistake there is silent: it
produces plausible numbers in the right shape.
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
    "fetch_era5_district_observations",
    BACKEND / "scripts" / "fetch_era5_district_observations.py")
obs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = obs
_spec.loader.exec_module(obs)

from app.utils import india_districts as idist  # noqa: E402


@pytest.fixture(scope="module")
def cells_for():
    w = pd.read_parquet(idist.geo_dir() / idist.WEIGHTS_FILENAME)

    def _get(region_id):
        return (w[w.region_id == region_id][["lat", "lon"]]
                .drop_duplicates().reset_index(drop=True))
    return _get


def _rows(cells, **values):
    """One date, a constant value per variable across every cell."""
    base = {c: 0.0 for c in obs.VALUE_COLUMNS}
    base.update(values)
    return pd.DataFrame([{**base, "date": pd.Timestamp("2019-07-17").date(),
                          "lat": c.lat, "lon": c.lon}
                         for c in cells.itertuples()])


def test_constant_field_returns_that_constant(cells_for):
    cells = cells_for("IN-MP-BHOPAL")
    out = obs.to_districts(_rows(cells, t2m_c=27.5, precip_mm=4.0), cells)
    row = out[out.region_id == "IN-MP-BHOPAL"].iloc[0]
    assert row.t2m_c == pytest.approx(27.5)
    assert row.precip_mm == pytest.approx(4.0)


def test_districts_without_a_sampled_cell_are_nan(cells_for):
    """Never a fabricated zero: a district nothing was sampled for has no value."""
    cells = cells_for("IN-MP-BHOPAL")
    out = obs.to_districts(_rows(cells, t2m_c=27.5), cells)
    assert len(out) == 666, "every district appears, whether or not it has data"
    far = out[out.region_id == "IN-AN-NICOBARISLANDS"].iloc[0]
    assert np.isnan(far.t2m_c)


def test_wind_direction_averages_circularly(cells_for):
    """350 and 10 degrees average to 0, not to 180. A plain mean would point the
    opposite way, and would do it without failing."""
    cells = cells_for("IN-RJ-JAIPUR")
    rows = _rows(cells)
    rows["wdir10m_deg"] = np.where(np.arange(len(rows)) % 2 == 0, 350.0, 10.0)
    out = obs.to_districts(rows, cells)
    got = out[out.region_id == "IN-RJ-JAIPUR"].iloc[0].wdir10m_deg
    assert min(got, 360.0 - got) < 15.0, f"expected ~0/360, got {got}"


def test_wind_direction_constant_is_preserved(cells_for):
    cells = cells_for("IN-RJ-JAIPUR")
    rows = _rows(cells)
    rows["wdir10m_deg"] = 265.0
    out = obs.to_districts(rows, cells)
    assert out[out.region_id == "IN-RJ-JAIPUR"].iloc[0].wdir10m_deg == pytest.approx(
        265.0, abs=0.5)


def test_a_nan_cell_does_not_void_the_district(cells_for):
    """One missing cell must reduce to the cells that do have data, matching the
    forecast side's behaviour over sea cells."""
    cells = cells_for("IN-RJ-JAIPUR")
    rows = _rows(cells, t2m_c=30.0)
    rows.loc[0, "t2m_c"] = np.nan
    out = obs.to_districts(rows, cells)
    assert out[out.region_id == "IN-RJ-JAIPUR"].iloc[0].t2m_c == pytest.approx(30.0)


def test_all_cells_nan_gives_nan(cells_for):
    cells = cells_for("IN-RJ-JAIPUR")
    rows = _rows(cells)
    rows["t2m_c"] = np.nan
    out = obs.to_districts(rows, cells)
    assert np.isnan(out[out.region_id == "IN-RJ-JAIPUR"].iloc[0].t2m_c)


def test_output_carries_one_row_per_district_per_date(cells_for):
    cells = cells_for("IN-MP-BHOPAL")
    a = _rows(cells, t2m_c=27.0)
    b = _rows(cells, t2m_c=28.0)
    b["date"] = pd.Timestamp("2019-07-18").date()
    out = obs.to_districts(pd.concat([a, b], ignore_index=True), cells)
    assert len(out) == 666 * 2
    assert not out.duplicated(["region_id", "date"]).any()


def test_canonical_columns_match_the_forecast_side():
    """Both halves must use the same names, or the join needs a translation layer that
    is one more place to get a variable wrong."""
    assert "t2m_c" in obs.VALUE_COLUMNS and "precip_mm" in obs.VALUE_COLUMNS
    assert "pwat_kgm2" in obs.VALUE_COLUMNS, \
        "water vapour has no daily endpoint but backs a regressor; it must survive"
