"""C4 - district descriptors: real, static, geometry-derived per-district features that
replace region_id as a raw 666-level categorical model input.

Why replace region_id: it is a categorical the model can only ever see ~34 of 666 levels
of during training (label coverage, CLAUDE.md known limitations), so it cannot generalise
to a district that never appears in the training labels. Continuous descriptors - where a
district sits, how big it is, how far from the modelled region's edge - let a district the
model never saw in training borrow strength from ones that look like it.

area_km2 and border_distance_km use a local equirectangular (tangent-plane) projection,
not a geodesic one. That is a documented approximation, not a claim of geodetic
precision: good to a fraction of a percent at district scale, at the cost of some
distortion across India's ~29 degree latitude span. The arithmetic tests below hand-check
that exact, explicit formula on small synthetic shapes, labelled as such. The real-data
tests below that check a real, verifiable geography fact (Mumbai is coastal, Jabalpur is
not) against the real GADM polygons already on disk - no synthetic geometry, no new fetch.

border_distance_km is honestly what it says: distance to the nearest edge of the
modelled landmass (coastline OR international land border - Pakistan, China, Nepal,
Bhutan, Bangladesh, Myanmar). This repo has no separate coastline reference to tell the
two apart, so it is not called "coastal_distance" - see docs/known-issues.md.
"""
from __future__ import annotations

import math

import pytest
from shapely.geometry import box

from scripts import build_district_descriptors as bdd


# --------------------------------------------------------------- formula, hand-computed

def test_the_projection_reproduces_a_hand_computed_rectangle_area():
    lat0, lon0 = 20.05, 77.05
    rect = box(77.0, 20.0, 77.1, 20.1)          # 0.1 deg square, ARITHMETIC FIXTURE
    got = bdd._polygon_area_km2(rect, lat0, lon0)
    width_km = 0.1 * bdd.KM_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(lat0))
    height_km = 0.1 * bdd.KM_PER_DEG_LAT
    assert got == pytest.approx(width_km * height_km, rel=1e-9)


def test_a_point_on_the_boundary_is_zero_km_away():
    square = box(0.0, 0.0, 1.0, 1.0).boundary   # ARITHMETIC/PLUMBING FIXTURE
    got = bdd._boundary_distance_km(0.0, 0.5, square, lat0=0.0, lon0=0.0)
    assert got == pytest.approx(0.0, abs=1e-9)


def test_the_middle_of_a_square_is_exactly_half_its_side_from_the_edge():
    square = box(0.0, 0.0, 1.0, 1.0).boundary   # ARITHMETIC FIXTURE
    edge = bdd._boundary_distance_km(0.0, 0.5, square, lat0=0.0, lon0=0.0)
    middle = bdd._boundary_distance_km(0.5, 0.5, square, lat0=0.0, lon0=0.0)
    # hand-computed: half the box's height in km, 0.5 deg * 110.574 km/deg (lat0=0, so
    # the longitude scale factor cos(0)=1 does not enter this north-south measurement).
    assert middle == pytest.approx(0.5 * bdd.KM_PER_DEG_LAT, rel=1e-9)
    assert middle > edge


# --------------------------------------------------------------- elevation, hand-computed

def test_elevation_is_the_existing_aggregators_area_weighted_mean(tmp_path):
    """elevation_mean must go through the SAME DistrictGridAggregator GEFS/ERA5 already
    use (CLAUDE.md: one weight table), not a second spatial join. A tiny, hand-crafted
    weight table proves the wiring: region X is 75% cell A (elevation 100 m) and 25% cell
    B (elevation 300 m) by weight, so the area-weighted mean is (3*100 + 1*300) / 4 = 150,
    not the plain average (200)."""
    import pandas as pd

    from app.utils.india_districts import DistrictGridAggregator

    weights = pd.DataFrame({
        "region_id": ["X", "X"], "lat": [10.0, 10.0], "lon": [77.0, 77.25],
        "weight": [3.0, 1.0],
    })
    wpath = tmp_path / "weights.parquet"
    weights.to_parquet(wpath, index=False)
    agg = DistrictGridAggregator(weights_path=wpath)

    elev = pd.DataFrame({"lat": [10.0, 10.0], "lon": [77.0, 77.25],
                        "elevation_m": [100.0, 300.0]})
    out = bdd._elevation_by_district(elev, aggregator=agg)
    assert out.loc["X"] == pytest.approx(150.0)


# --------------------------------------------------------- real geometry, real geography
#
# bdd.build() unions and queries all 666 real district polygons - a few minutes, not a
# few milliseconds. Session-scoped so the real-data tests below pay that cost once, not
# once per test; nothing here mutates the result.

@pytest.fixture(scope="session")
def real_descriptors():
    return bdd.build()


def test_mumbai_city_is_far_closer_to_the_border_than_landlocked_jabalpur(real_descriptors):
    """Real GADM polygons, not synthetic. Mumbai City is a coastal peninsula; Jabalpur
    sits in central Madhya Pradesh, several hundred km from the nearest coast or
    international border - an ordering any atlas confirms, used here as a sanity check on
    real geometry rather than a hand-computed number."""
    out = real_descriptors.set_index("region_id")
    mumbai = out.loc["IN-MH-MUMBAICITY", "border_distance_km"]
    jabalpur = out.loc["IN-MP-JABALPUR", "border_distance_km"]
    assert mumbai < 50
    assert jabalpur > 300
    assert mumbai < jabalpur


def test_every_district_gets_a_positive_unique_area(real_descriptors):
    from app.utils import india_districts as idist

    out = real_descriptors
    assert (out["area_km2"] > 0).all()
    assert (out["border_distance_km"] >= 0).all()
    assert out["region_id"].is_unique
    assert len(out) == len(idist.load_registry())
    assert set(out.columns) == {
        "region_id", "state_id", "centroid_lat", "centroid_lon",
        "area_km2", "border_distance_km", "elevation_mean",
    }


def test_lahul_and_spiti_is_far_higher_than_coastal_mumbai(real_descriptors):
    """Real fetched elevation (scripts/fetch_grid_elevation.py), not synthetic. Lahul &
    Spiti is a high-Himalayan Himachal Pradesh district, commonly cited around 3000-4000 m
    average elevation; Mumbai City is sea-level coastal - an ordering any atlas confirms."""
    out = real_descriptors.set_index("region_id")
    lahul_spiti = out.loc["IN-HP-LAHULSPITI", "elevation_mean"]
    mumbai = out.loc["IN-MH-MUMBAICITY", "elevation_mean"]
    assert lahul_spiti > 2000
    assert mumbai < 50
    assert lahul_spiti > mumbai


# --------------------------------------------------------------------- wired into loading

def test_the_descriptor_table_round_trips_through_disk(real_descriptors, tmp_path, monkeypatch):
    from app.utils import india_districts as idist

    out = real_descriptors
    path = tmp_path / bdd.OUT_FILENAME
    out.to_parquet(path, index=False)

    monkeypatch.setattr(idist, "geo_dir", lambda: tmp_path)
    idist.load_district_descriptors.cache_clear()
    loaded = idist.load_district_descriptors()
    idist.load_district_descriptors.cache_clear()
    pd_out = out.set_index("region_id").sort_index()
    pd_loaded = loaded.set_index("region_id").sort_index()
    assert list(pd_out.columns) == list(pd_loaded.columns)
    assert pd_out.equals(pd_loaded)


# --------------------------------------------------------------------- wired into frames

def test_the_training_frame_carries_the_descriptors_and_drops_region_id_as_a_feature():
    import numpy as np
    import pandas as pd

    from app.features import engineering as fe
    from app.features import pivot as pv
    from app.ml import classifier as clf_mod
    from app.ml import regressors as reg_mod

    fc = pd.DataFrame([{
        "region_id": "IN-MH-MUMBAICITY", "variable": "temperature_c",
        "value_type": "forecast", "valid_date": pd.Timestamp("2017-11-10"),
        "init_date": pd.Timestamp("2017-11-09"), "lead_time_days": 2,
        "ensemble_member_id": "m0", "value": 30.0,
    }])
    ob = pd.DataFrame([{
        "region_id": "IN-MH-MUMBAICITY", "variable": "temperature_c",
        "value_type": "observed", "valid_date": pd.Timestamp("2017-11-10"),
        "value": 31.0,
    }])
    frame = fe.build_training_frame(pd.concat([fc, ob], ignore_index=True))

    for col in ("state_id", "centroid_lat", "centroid_lon", "area_km2",
               "border_distance_km", "elevation_mean"):
        assert col in frame.columns
    assert frame.loc[0, "state_id"] == "IN-MH"
    assert frame.loc[0, "area_km2"] > 0

    reg_cols = reg_mod.feature_columns(frame)
    assert "region_id" not in reg_cols
    assert "state_id" in reg_cols
    assert "area_km2" in reg_cols
    assert "border_distance_km" in reg_cols
    assert "elevation_mean" in reg_cols

    assert clf_mod.CATEGORICAL == ["state_id", "season"]

    pred = pd.Series(1.0, index=frame.index)
    ev = pv.build_event_frame(frame, pred, {"temperature_c": 5.0}, {"temperature_c": 3.0})
    clf_cols = pv.classifier_feature_columns(ev)
    assert "region_id" not in clf_cols
    assert "state_id" in clf_cols
    assert "area_km2" in clf_cols
    assert "border_distance_km" in clf_cols
    assert "elevation_mean" in clf_cols
    # region_id itself must still be a real column - joins, SHAP grouping and the API
    # response all key on it. Only its use as a raw model FEATURE is removed.
    assert "region_id" in frame.columns
    assert "region_id" in ev.columns
