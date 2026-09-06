"""india_districts.py against the real generated geography.

These run on the committed files in data/geo/, not on fixtures - the same rule the rest of
the suite follows. If build_district_geo.py is re-run against a new boundary release,
these are what say whether the result is usable.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.utils import india_districts as idist
from app.utils.india_districts import DistrictGridAggregator, load_registry

STATES_WITH_KNOWN_COUNTS = {
    "IN-TG": 10,   # Telangana exists as its own state - the boundary set must be post-2014
    "IN-LA": 2,    # Leh and Kargil, reassigned out of Jammu and Kashmir
}


# --------------------------------------------------------------------------- registry

def test_registry_loads_every_district():
    reg = load_registry()
    assert len(reg) == 666
    assert len({d.region_id for d in reg}) == len(reg), "region_id must be unique"


def test_every_state_and_ut_is_represented():
    from app.utils.india_state_codes import STATES

    covered = {d.state_id for d in load_registry()}
    missing = {s.region_id for s in STATES} - covered
    assert not missing, f"states with no districts: {sorted(missing)}"


@pytest.mark.parametrize("state_id,count", STATES_WITH_KNOWN_COUNTS.items())
def test_reorganised_states(state_id, count):
    assert sum(1 for d in load_registry() if d.state_id == state_id) == count


def test_disputed_territory_is_kept():
    """GADM files all of Jammu and Kashmir under a disputed-area GID with no IND.*
    counterpart. Dropping those features would erase the state from the map."""
    by_state = {}
    for d in load_registry():
        by_state.setdefault(d.state_id, []).append(d)
    assert len(by_state.get("IN-JK", [])) == 20
    assert len(by_state.get("IN-AR", [])) == 18


def test_names_are_spaced_for_display():
    """The source stores names unspaced; unfixed, these reach the dashboard as-is."""
    names = {d.region_id: d.region_name for d in load_registry()}
    assert names["IN-MH-MUMBAISUBURBAN"] == "Mumbai Suburban"
    assert names["IN-AN-NORTHANDMIDDLEANDAMAN"] == "North and Middle Andaman"
    assert names["IN-WB-NORTH24PARGANAS"] == "North 24 Parganas"
    assert names["IN-LA-LEHLADAKH"] == "Leh (Ladakh)"
    for d in load_registry():
        assert "  " not in d.region_name
        assert d.region_name == d.region_name.strip()


def test_display_name_carries_the_state():
    d = idist.resolve_by_id("IN-MH-NAGPUR")
    assert d is not None and d.display_name == "Nagpur, Maharashtra"


def test_ambiguous_names_refuse_to_resolve():
    """Aurangabad is in both Bihar and Maharashtra, Bilaspur in both Chhattisgarh and
    Himachal Pradesh. Guessing between them would silently attribute one district's
    weather to another, so an unqualified name that matches several resolves to nothing."""
    assert idist.resolve_by_name("Aurangabad") is None
    assert idist.resolve_by_name("Bilaspur") is None
    assert idist.resolve_by_name("Aurangabad", state="Bihar").state_id == "IN-BR"
    assert idist.resolve_by_name("Nagpur").region_id == "IN-MH-NAGPUR"


def test_delhi_is_named_delhi():
    """The source labels the whole NCT 'West'. Delhi is the most-looked-at region on the
    map; a mislabel there reads as a broken dataset."""
    d = idist.resolve_by_id("IN-DL-DELHI")
    assert d is not None and d.region_name == "Delhi"
    assert idist.resolve_by_id("IN-DL-WEST") is None


# --------------------------------------------------------------------------- weights

def test_weights_cover_every_district_and_sum_to_one():
    w = pd.read_parquet(idist.geo_dir() / idist.WEIGHTS_FILENAME)
    sums = w.groupby("region_id")["weight"].sum()
    assert np.allclose(sums.to_numpy(), 1.0)
    assert set(sums.index) == {d.region_id for d in load_registry()}, \
        "a district with no overlapping grid cell can never be scored"


# ------------------------------------------------------------------------ aggregator

@pytest.fixture(scope="module")
def agg() -> DistrictGridAggregator:
    return DistrictGridAggregator()


def _india_grid(agg):
    """The exact cells the weight table knows, as flat lat/lon arrays."""
    keys = list(agg._cell_pos)
    lats = np.array([k[0] for k in keys], dtype=float) * idist.GRID_DEG
    lons = np.array([k[1] for k in keys], dtype=float) * idist.GRID_DEG
    return lats, lons


def test_constant_field_aggregates_to_that_constant(agg):
    lats, lons = _india_grid(agg)
    out = agg.aggregate(lats, lons, np.full(len(lats), 42.0))
    assert len(out) == 666
    assert np.allclose(out.to_numpy(), 42.0)


def test_aggregate_is_bounded_by_the_field(agg):
    lats, lons = _india_grid(agg)
    rng = np.random.default_rng(0)
    vals = rng.uniform(-5, 40, len(lats))
    out = agg.aggregate(lats, lons, vals).to_numpy()
    assert out.min() >= vals.min() - 1e-9
    assert out.max() <= vals.max() + 1e-9
    assert not np.isnan(out).any()


def test_gefs_longitude_convention_is_handled(agg):
    """GEFS publishes 0..360; the boundaries are -180..180. Same field, same answer."""
    lats, lons = _india_grid(agg)
    vals = np.linspace(0, 1, len(lats))
    a = agg.aggregate(lats, lons, vals)
    b = agg.aggregate(lats, lons % 360, vals)
    pd.testing.assert_series_equal(a, b)


def test_nan_cells_are_dropped_not_propagated(agg):
    """A land variable over sea is NaN. A coastal district must still get the mean of the
    cells that do have data, rather than becoming NaN itself."""
    lats, lons = _india_grid(agg)
    vals = np.full(len(lats), 7.0)
    vals[::3] = np.nan
    out = agg.aggregate(lats, lons, vals).to_numpy()
    finite = out[~np.isnan(out)]
    assert len(finite) > 600
    assert np.allclose(finite, 7.0)


def test_district_with_no_valid_cell_is_nan_not_zero(agg):
    lats, lons = _india_grid(agg)
    out = agg.aggregate(lats, lons, np.full(len(lats), np.nan))
    assert np.isnan(out.to_numpy()).all(), "missing data must never read as 0.0"


def test_cells_outside_india_are_ignored(agg):
    lats, lons = _india_grid(agg)
    extra_lat = np.concatenate([lats, [0.0, 60.0]])
    extra_lon = np.concatenate([lons, [0.0, 10.0]])
    vals = np.concatenate([np.full(len(lats), 3.0), [999.0, 999.0]])
    out = agg.aggregate(extra_lat, extra_lon, vals)
    assert np.allclose(out.to_numpy(), 3.0)


def test_mismatched_input_lengths_raise(agg):
    with pytest.raises(ValueError):
        agg.aggregate([10.0, 11.0], [70.0], [1.0, 2.0])


# ------------------------------------------------------------------- geojson pairing

def test_geojson_and_registry_describe_the_same_districts():
    gj = json.loads((idist.geo_dir() / idist.GEOJSON_FILENAME).read_text())
    ids = {f["properties"]["region_id"] for f in gj["features"]}
    assert ids == {d.region_id for d in load_registry()}
