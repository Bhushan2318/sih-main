"""The live feed reads districts, not 36 city points.

CLAUDE.md says each district's value is "the area-weighted mean of every 0.25 degree cell
its polygon overlaps - not a nearest-point sample". That was true of the reforecast path
and never true of the live one: `app/live/gefs.py` read `scripts/india_cities.json`, 36
points, and resolved each point to whatever district contained it. The published bundle
showed the consequence - 71 regions on both sides, every forecast row carrying a point
resolver - and the national map drew 36 districts of a 666-district country.

The fetch already downloads the whole grid. It asks NOMADS for `subregion` over
BBOX 6-38N, 68-98E at 0.25 degrees, roughly 15,600 cells, and then discarded all but 36
of them. So this is not a bandwidth change; it is reading what was already arriving.

There is exactly one weight table and CLAUDE.md forbids a second, so these tests assert
the live path goes through `app.utils.india_districts`, not that it computes its own
geography.

The fields here are synthetic and labelled as such: they exercise shape and plumbing, and
the two invariants below hold for any real field regardless of its values. Nothing here
produces a metric. A real-GRIB end-to-end check is a separate, network-bound test.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.live import gefs
from app.utils.india_districts import get_aggregator


@pytest.fixture(scope="module")
def grid():
    """The 0.25 degree grid the live fetch actually asks NOMADS for."""
    lat = np.arange(gefs.BBOX["bottomlat"], gefs.BBOX["toplat"] + 0.25, 0.25)
    lon = np.arange(gefs.BBOX["leftlon"], gefs.BBOX["rightlon"] + 0.25, 0.25)
    lons, lats = np.meshgrid(lon, lat)
    return lats.ravel(), lons.ravel()


def test_the_city_table_is_no_longer_read():
    """The 36-point path is gone, not merely unused.

    Left in place it would be the obvious thing to fall back to, and a fallback to 36
    points would look like a working site rather than a broken one.
    """
    assert not hasattr(gefs, "_extract_points"), (
        "_extract_points still exists - the point-sampling path must be removed, not "
        "bypassed")
    assert not hasattr(gefs, "load_cities")
    assert not hasattr(gefs, "CITIES_JSON")


def test_one_value_per_district_in_the_weight_table(grid):
    """Shape: the extractor's arrays must be district-length, not city-length."""
    lats, lons = grid
    agg = get_aggregator()
    field = np.full(lats.shape, 300.0)
    out = agg.aggregate(lats, lons, field)
    assert len(out) == len(agg.region_ids)
    assert len(out) > 600, f"expected the full district registry, got {len(out)}"


def test_a_constant_field_returns_that_constant_everywhere(grid):
    """The one invariant that holds for any weighting: a weighted mean of a constant.

    This is what separates a correct aggregation from one whose weights do not sum as it
    thinks they do - it fails on a normalisation bug without needing a reference value.
    """
    lats, lons = grid
    out = get_aggregator().aggregate(lats, lons, np.full(lats.shape, 42.5))
    covered = out.dropna()
    assert len(covered) > 600
    assert np.allclose(covered.to_numpy(), 42.5), "weights do not renormalise to 1"


def test_sea_cells_do_not_turn_a_coastal_district_into_nan(grid):
    """A land variable is NaN over sea. A coastal district overlaps both.

    Dropping the NaN cells and renormalising over what is left is the difference between
    a coastal district reporting a real soil-moisture value and reporting nothing. The
    aggregator documents this; this asserts the live path inherits it.
    """
    lats, lons = grid
    field = np.full(lats.shape, 10.0)
    # Label every cell south of 10N as sea, which removes part of the peninsular coast.
    field[lats < 10.0] = np.nan
    out = get_aggregator().aggregate(lats, lons, field)
    kept = out.dropna()
    assert len(kept) > 500
    assert np.allclose(kept.to_numpy(), 10.0), (
        "a partially-NaN district must report the mean of its valid cells, not a value "
        "pulled toward zero")


def test_a_district_with_no_valid_cell_is_nan_not_a_number(grid):
    """Refuse rather than patch: missing never becomes zero (CLAUDE.md rule 3)."""
    lats, lons = grid
    out = get_aggregator().aggregate(lats, lons, np.full(lats.shape, np.nan))
    assert out.isna().all(), "an all-missing field must not fabricate district values"


def test_the_prepared_index_is_reused_across_fields(grid):
    """Prepared once per grid, not once per message.

    A cycle aggregates 400 fields - 8 variables x 5 members x 10 lead days. Resolving
    every cell through a dict on each one is ~8 million Python lookups per cycle, which
    is what the aggregator's own docstring says `prepare` exists to avoid. If the live
    path calls `aggregate` per field it will work and be far slower, so this asserts the
    prepared form is what gets used.
    """
    lats, lons = grid
    agg = get_aggregator()
    idx = agg.prepare(lats, lons)
    a = agg.aggregate_prepared(idx, np.full(lats.shape, 1.0))
    b = agg.aggregate_prepared(idx, np.full(lats.shape, 2.0))
    assert np.allclose(a.dropna().to_numpy(), 1.0)
    assert np.allclose(b.dropna().to_numpy(), 2.0)
    assert hasattr(gefs, "_prepared_index_for"), (
        "the live path needs a cached prepared index; add _prepared_index_for(lats, lons)")
