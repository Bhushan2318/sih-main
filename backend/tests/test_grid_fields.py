"""grid_fields.py - the storage format behind the convolutional model's inputs.

The int16 encoding is a lossy step applied to every value the network ever sees, so what
it costs is measured here rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ingestion import grid_fields as gf
from app.ingestion.grid_fields import GridBundle, load_bundle, save_bundle


def _bundle(values=None, variables=("t2m_c", "rain_mm"), leads=(1, 2, 3)):
    lats, lons = gf.domain_coords()
    shape = (len(variables), len(leads), len(gf.STATS), len(lats), len(lons))
    if values is None:
        rng = np.random.default_rng(7)
        values = rng.uniform(-40, 45, shape).astype(np.float32)
    return GridBundle("2019-07-17", variables, leads, lats, lons,
                      values.astype(np.float32))


# ----------------------------------------------------------------------- domain

def test_domain_matches_the_declared_bounds():
    lats, lons = gf.domain_coords()
    assert (len(lats), len(lons)) == (145, 141)
    assert lats[0] == pytest.approx(gf.DOMAIN_LAT[0])
    assert lats[-1] == pytest.approx(gf.DOMAIN_LAT[1])
    assert lons[0] == pytest.approx(gf.DOMAIN_LON[0])
    assert lons[-1] == pytest.approx(gf.DOMAIN_LON[1])


def test_domain_contains_every_district():
    """A district outside the stored grid could never be predicted from it."""
    import json
    from app.utils import india_districts as idist

    reg = json.loads((idist.geo_dir() / idist.REGISTRY_FILENAME).read_text())
    lo0, lo1 = gf.DOMAIN_LON
    la0, la1 = gf.DOMAIN_LAT
    for r in reg:
        x0, y0, x1, y1 = r["bbox"]
        assert lo0 <= x0 and x1 <= lo1, f"{r['region_id']} outside domain in longitude"
        assert la0 <= y0 and y1 <= la1, f"{r['region_id']} outside domain in latitude"


def test_domain_reaches_beyond_the_coast():
    """The seas are where the depressions that cause busts form."""
    assert gf.DOMAIN_LON[0] <= 66.0 and gf.DOMAIN_LON[1] >= 97.5
    assert gf.DOMAIN_LAT[0] <= 6.0


# ---------------------------------------------------------------------- round trip

def test_round_trip_preserves_values_to_forecast_precision(tmp_path):
    b = _bundle()
    got = load_bundle(save_bundle(tmp_path / "b.npz", b))
    err = np.abs(got.values - b.values)
    assert np.nanmax(err) < 0.01, "int16 encoding must be finer than forecast accuracy"
    assert got.variables == b.variables and got.leads == b.leads
    assert got.init_date == b.init_date


def test_precision_is_half_a_quantum(tmp_path):
    """The guarantee the encoding actually makes: rounding to the nearest QUANTUM, so
    error never exceeds half of one. Over a 120 K spread that is 0.005 K, against a model
    whose 2 m temperature error is measured in whole degrees."""
    lats, lons = gf.domain_coords()
    shape = (1, 1, len(gf.STATS), len(lats), len(lons))
    rng = np.random.default_rng(0)
    vals = rng.uniform(200.0, 320.0, shape).astype(np.float32)
    b = GridBundle("2019-07-17", ("t2m_k",), (1,), lats, lons, vals)
    got = load_bundle(save_bundle(tmp_path / "t.npz", b))
    assert np.nanmax(np.abs(got.values - vals)) <= gf.QUANTUM / 2 + 1e-4


def test_a_plane_too_wide_for_the_quantum_widens_instead_of_clipping(tmp_path):
    """A 900 mm rainfall day spans more than int16 can hold at 0.01 mm. It must come back
    coarser, never truncated to the ends of the range."""
    lats, lons = gf.domain_coords()
    shape = (1, 1, len(gf.STATS), len(lats), len(lons))
    vals = np.linspace(0.0, 900.0, int(np.prod(shape))).reshape(shape).astype(np.float32)
    got = load_bundle(save_bundle(tmp_path / "w.npz",
                                  GridBundle("2019-07-17", ("rain_mm",), (1,),
                                             lats, lons, vals)))
    err = np.nanmax(np.abs(got.values - vals))
    assert err > gf.QUANTUM / 2, "this plane should have needed a wider step"
    assert err < 0.05, "but still far finer than a rainfall forecast is accurate to"
    assert got.values.max() == pytest.approx(900.0, abs=0.05)
    assert got.values.min() == pytest.approx(0.0, abs=0.05)


def test_missing_stays_missing(tmp_path):
    """A gap must never come back as 0.0 - soil moisture over sea, a short feed."""
    lats, lons = gf.domain_coords()
    vals = np.full((1, 1, len(gf.STATS), len(lats), len(lons)), 5.0, dtype=np.float32)
    vals[0, 0, 0, :10, :] = np.nan
    got = load_bundle(save_bundle(tmp_path / "n.npz",
                                  GridBundle("2019-07-17", ("v",), (1,), lats, lons, vals)))
    assert np.isnan(got.values[0, 0, 0, :10, :]).all()
    assert np.allclose(got.values[0, 0, 0, 10:], 5.0)


def test_all_nan_plane_survives(tmp_path):
    lats, lons = gf.domain_coords()
    vals = np.full((1, 1, len(gf.STATS), len(lats), len(lons)), np.nan, dtype=np.float32)
    got = load_bundle(save_bundle(tmp_path / "a.npz",
                                  GridBundle("2019-07-17", ("v",), (1,), lats, lons, vals)))
    assert np.isnan(got.values).all()


def test_constant_plane_survives(tmp_path):
    """A zero-span plane must not divide by zero."""
    lats, lons = gf.domain_coords()
    vals = np.full((1, 1, len(gf.STATS), len(lats), len(lons)), 3.5, dtype=np.float32)
    got = load_bundle(save_bundle(tmp_path / "c.npz",
                                  GridBundle("2019-07-17", ("v",), (1,), lats, lons, vals)))
    assert np.allclose(got.values, 3.5)


def test_shape_mismatch_is_rejected():
    lats, lons = gf.domain_coords()
    with pytest.raises(ValueError):
        GridBundle("2019-07-17", ("a", "b"), (1,), lats, lons,
                   np.zeros((1, 1, 2, len(lats), len(lons)), dtype=np.float32))


def test_field_accessor_indexes_correctly():
    b = _bundle()
    assert np.array_equal(b.field("rain_mm", 2, "spread"), b.values[1, 1, 1])


# ------------------------------------------------------------------- regridding

def test_subset_handles_gefs_orientation():
    """GEFS gives descending latitude on 0..360; the store ascends on -180..180."""
    src_lats = np.arange(40.0, 1.9, -gf.GRID_DEG)         # descending
    src_lons = np.arange(60.0, 110.0 + 0.1, gf.GRID_DEG)  # 0..360 convention
    field = np.tile(src_lats[:, None], (1, len(src_lons))).astype(np.float32)

    out = gf.subset_to_domain(src_lats, src_lons, field)
    want_lats, want_lons = gf.domain_coords()
    assert out.shape == (len(want_lats), len(want_lons))
    assert not np.isnan(out).any()
    # every row should carry its own latitude, proving the flip was applied
    assert np.allclose(out[:, 0], want_lats, atol=1e-4)


def test_subset_leaves_uncovered_cells_nan():
    """A source narrower than the domain must not be stretched to fill it."""
    src_lats = np.arange(20.0, 9.9, -gf.GRID_DEG)
    src_lons = np.arange(70.0, 80.0 + 0.1, gf.GRID_DEG)
    field = np.ones((len(src_lats), len(src_lons)), dtype=np.float32)
    out = gf.subset_to_domain(src_lats, src_lons, field)
    assert np.isnan(out).any()
    assert np.nansum(out) == pytest.approx(len(src_lats) * len(src_lons))


# ------------------------------------------------- pooling through the weight table

def test_grid_pools_to_districts_through_the_shared_weights():
    """The CNN pools its feature map to districts with the same area weights the tabular
    path uses. Same operator, so the two models see the same geography."""
    from app.utils.india_districts import get_aggregator

    b = _bundle(variables=("t2m_c",), leads=(1,))
    lat, lon = b.flat_coords()
    plane = b.field("t2m_c", 1)
    out = get_aggregator().aggregate(lat, lon, plane.ravel())
    assert len(out) == 666
    assert not out.isna().any(), "every district must pool from the stored domain"
    assert out.min() >= np.nanmin(plane) - 1e-6
    assert out.max() <= np.nanmax(plane) + 1e-6
