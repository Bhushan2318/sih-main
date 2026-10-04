"""Soil moisture is a land quantity, and both models have sea in coastal districts.

GEFS fills its sea cells with ~1.0 soil moisture (100%), and a district's GEFS value is an
area mean over every cell it touches, sea included. Mumbai City is 98.5% GEFS sea by
weight; Nicobar, Diu and Lakshadweep are entirely sea in both models. ERA5 reads ~0 over
water. So in coastal districts the soil "forecast" was largely a sea value and the soil
"observation" largely zero, and their difference was a bust by construction.

Feature version 2 drops soil moisture, on both sides, for districts where more than
SOIL_SEA_FRACTION_MAX of the weight is sea in either model (data/geo/soil_land_mask.parquet,
built by scripts/build_soil_land_mask.py from the GEFS grids and ERA5's land-sea mask).
Every other variable in those districts is kept.

PLUMBING FIXTURES: hand-built rows for the filtering logic; the mask file test reads the
real committed mask.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.features import engineering as fe

BACKEND = Path(__file__).resolve().parents[1]


def _rows(region, variable, value_type, value):
    r = {"region_id": region, "variable": variable, "value_type": value_type,
         "valid_date": pd.Timestamp("2017-11-03"), "value": value}
    if value_type == "forecast":
        r.update(init_date=pd.Timestamp("2017-11-02"), lead_time_days=2,
                 ensemble_member_id="m0")
    return r


def _canonical():
    rows = []
    for region in ("IN-MH-MUMBAICITY", "IN-MH-PUNE"):
        for var, fc, ob in (("soil_moisture_pct", 60.0, 30.0), ("temperature_c", 28.0, 27.0)):
            rows += [_rows(region, var, "forecast", fc), _rows(region, var, "observed", ob)]
    return pd.DataFrame(rows)


@pytest.fixture
def mumbai_excluded(monkeypatch):
    monkeypatch.setattr(fe, "soil_excluded_districts", lambda: frozenset({"IN-MH-MUMBAICITY"}))


def test_version_two_drops_soil_for_excluded_districts_on_both_sides(mumbai_excluded):
    f = fe.build_training_frame(_canonical(), feature_version=2)
    soil = f[f["variable"] == "soil_moisture_pct"]
    assert set(soil["region_id"].astype(str)) == {"IN-MH-PUNE"}


def test_other_variables_of_an_excluded_district_are_kept(mumbai_excluded):
    f = fe.build_training_frame(_canonical(), feature_version=2)
    temp = f[f["variable"] == "temperature_c"]
    assert set(temp["region_id"].astype(str)) == {"IN-MH-MUMBAICITY", "IN-MH-PUNE"}


def test_version_one_keeps_every_district_soil(mumbai_excluded):
    f = fe.build_training_frame(_canonical(), feature_version=1)
    soil = f[f["variable"] == "soil_moisture_pct"]
    assert set(soil["region_id"].astype(str)) == {"IN-MH-MUMBAICITY", "IN-MH-PUNE"}


def test_version_two_refuses_without_the_mask_file(monkeypatch, tmp_path):
    monkeypatch.setattr(fe, "SOIL_LAND_MASK_PATH", tmp_path / "missing.parquet")
    fe.soil_excluded_districts.cache_clear()
    try:
        with pytest.raises(FileNotFoundError, match="soil land mask"):
            fe.build_training_frame(_canonical(), feature_version=2)
    finally:
        fe.soil_excluded_districts.cache_clear()


# --------------------------------------------------------------- building the mask

def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_soil_land_mask", BACKEND / "scripts" / "build_soil_land_mask.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_district_sea_fractions_are_weight_sums():
    b = _load_builder()
    weights = pd.DataFrame({"region_id": ["A", "A", "B"], "lat": [10.0, 10.25, 20.0],
                            "lon": [70.0, 70.0, 80.0], "weight": [0.75, 0.25, 1.0]})
    gefs = pd.DataFrame({"lat": [10.0, 10.25, 20.0], "lon": [70.0, 70.0, 80.0],
                         "gefs_nonland": [False, True, False]})
    lsm = pd.DataFrame({"lat": [10.0, 10.25, 20.0], "lon": [70.0, 70.0, 80.0],
                        "lsm": [1.0, 0.2, 0.3]})
    t = b.district_mask(weights, gefs, lsm).set_index("region_id")
    assert t.loc["A", "gefs_nonland_frac"] == pytest.approx(0.25)
    assert t.loc["A", "era5_water_frac"] == pytest.approx(0.25)
    assert bool(t.loc["A", "soil_excluded"])


def test_era5_water_alone_does_not_exclude_soil():
    """v2 observations already use ERA5 land cells only; ERA5 water never reaches them."""
    b = _load_builder()
    weights = pd.DataFrame({"region_id": ["K"], "lat": [10.0], "lon": [76.0], "weight": [1.0]})
    gefs = pd.DataFrame({"lat": [10.0], "lon": [76.0], "gefs_nonland": [False]})
    lsm = pd.DataFrame({"lat": [10.0], "lon": [76.0], "lsm": [0.1]})
    t = b.district_mask(weights, gefs, lsm).set_index("region_id")
    assert t.loc["K", "era5_water_frac"] == pytest.approx(1.0)
    assert not bool(t.loc["K", "soil_excluded"])


def test_a_cell_is_gefs_sea_when_saturated_in_almost_every_valid_cycle():
    b = _load_builder()
    fields = np.array([[[1.004, 0.30]], [[1.000, 0.31]], [[np.nan, np.nan]], [[0.995, 0.29]]])
    got = b.gefs_nonland_cells(fields)
    assert got.tolist() == [[True, False]]


def test_the_committed_mask_covers_every_district_and_the_known_coasts():
    from app.utils import india_districts as idist
    m = pd.read_parquet(BACKEND / "data" / "geo" / "soil_land_mask.parquet")
    w = pd.read_parquet(idist.geo_dir() / idist.WEIGHTS_FILENAME)
    assert set(m["region_id"]) == set(w["region_id"])
    ex = set(m.loc[m["soil_excluded"], "region_id"])
    assert {"IN-AN-NICOBARISLANDS", "IN-LD-LAKSHADWEEP", "IN-DH-DIU",
            "IN-MH-MUMBAICITY"} <= ex
    assert "IN-MH-PUNE" not in ex and "IN-DL-DELHI" not in ex
