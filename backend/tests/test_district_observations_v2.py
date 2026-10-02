"""to_districts_v2 - district observations reduced the way the forecast side reduces GEFS.

The forecast side averages q, T, surface pressure, u and v over a district and only then
derives RH and wind (scripts/fetch_gefs_reforecast_sample.py, `_canonicalise`). v1
observations did it the other way round, and on Nov 2017 that moved RH by -2.28 %RH and
wind by -0.034 m/s on average. These tests pin the order: components first, derivation
after the mean, with the same functions as the forecast side.

Cells come from the real district weight table; values are hand-set for arithmetic only
and never reach a metric.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.utils import district_observations as obs
from app.utils import humidity
from app.utils import india_districts as idist

BACKEND = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, BACKEND / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cds = _load("fetch_era5_cds_district_observations")
ib = _load("ingest_backfill")

DAY = pd.Timestamp("2017-11-15").date()


@pytest.fixture(scope="module")
def weights():
    return pd.read_parquet(idist.geo_dir() / idist.WEIGHTS_FILENAME)


@pytest.fixture(scope="module")
def two_cell_district(weights):
    """A real district that draws on at least two cells, with its cells and weights."""
    n = weights.groupby("region_id").size()
    rid = sorted(n[n >= 2].index)[0]
    w = weights[weights.region_id == rid][["lat", "lon", "weight"]].reset_index(drop=True)
    return rid, w


def _cells_daily(cells: pd.DataFrame, **cols) -> pd.DataFrame:
    base = {"t2m_k": 300.0, "q2m_kgkg": 0.015, "sp_pa": 95000.0, "msl_pa": 101000.0,
            "u10_ms": 0.0, "v10_ms": 0.0, "tcwv_kgm2": 40.0, "swvl1_m3m3": 0.3,
            "tp_mm": 0.0}
    df = cells[["lat", "lon"]].copy().reset_index(drop=True)
    df["date"] = DAY
    for c in cds.CELL_DAILY_COLUMNS:
        v = cols.get(c, base[c])
        df[c] = np.asarray(v, dtype=float) if np.ndim(v) else float(v)
    return df


def _row(out: pd.DataFrame, rid: str) -> pd.Series:
    return out[out.region_id == rid].iloc[0]


def _wmean(w: pd.DataFrame, values) -> float:
    return float(np.sum(w.weight.to_numpy() * np.asarray(values)) / w.weight.sum())


# --------------------------------------------------------------- the contract

def test_output_columns_match_v1():
    """Same column names as v1, so a v2 file is a drop-in replacement for ingest."""
    cells = pd.DataFrame({"lat": [20.0], "lon": [78.0]})
    out = obs.to_districts_v2(_cells_daily(cells), cells)
    assert list(out.columns) == ["region_id", "date"] + obs.VALUE_COLUMNS


def test_units_are_canonical(two_cell_district):
    rid, w = two_cell_district
    out = obs.to_districts_v2(_cells_daily(w, tp_mm=12.5), w)
    r = _row(out, rid)
    assert r.t2m_c == pytest.approx(26.85)
    assert r.psfc_hpa == pytest.approx(950.0)
    assert r.mslp_hpa == pytest.approx(1010.0)
    assert r.soil_moisture_pct == pytest.approx(30.0)
    assert r.pwat_kgm2 == pytest.approx(40.0)
    assert r.precip_mm == pytest.approx(12.5)


# --------------------------------------------------------------- derive after the mean

def test_rh_is_derived_from_the_district_mean_components(two_cell_district):
    """RH = f(mean q, mean T, mean p) - the forecast side's estimator - and not the mean
    of each cell's RH, which differs whenever the cells differ."""
    rid, w = two_cell_district
    t = np.linspace(285.0, 305.0, len(w))
    q = np.linspace(0.006, 0.018, len(w))
    out = obs.to_districts_v2(_cells_daily(w, t2m_k=t, q2m_kgkg=q), w)
    got = _row(out, rid).rh2m_pct
    want = humidity.rh_from_specific_humidity(_wmean(w, q), _wmean(w, t), 95000.0)
    per_cell = _wmean(w, humidity.rh_from_specific_humidity(q, t, np.full(len(w), 95000.0)))
    assert got == pytest.approx(want, rel=1e-9)
    assert abs(got - per_cell) > 1e-3, "must not be the mean of per-cell RH"


def test_wind_is_the_speed_of_the_district_mean_vector(two_cell_district):
    """Opposing cells cancel, as they do in the forecast's district mean; the mean of
    per-cell speeds would read higher."""
    rid, w = two_cell_district
    u = np.where(np.arange(len(w)) % 2 == 0, 6.0, -6.0)
    v = np.full(len(w), 1.0)
    out = obs.to_districts_v2(_cells_daily(w, u10_ms=u, v10_ms=v), w)
    r = _row(out, rid)
    spd, drc = humidity.wind_speed_dir(np.array([_wmean(w, u)]), np.array([_wmean(w, v)]))
    assert r.wspd10m_ms == pytest.approx(spd[0], rel=1e-9)
    assert r.wdir10m_deg == pytest.approx(drc[0], rel=1e-9)
    assert r.wspd10m_ms < _wmean(w, np.hypot(u, v))


# --------------------------------------------------------------- soil over water

def test_soil_ignores_cells_that_are_mostly_water(two_cell_district):
    rid, w = two_cell_district
    lsm = w[["lat", "lon"]].assign(lsm=1.0)
    lsm.loc[0, "lsm"] = 0.2                                # first cell is mostly sea
    soil = np.full(len(w), 0.40)
    soil[0] = 0.0                                          # ERA5 reads ~0 over water
    out = obs.to_districts_v2(_cells_daily(w, swvl1_m3m3=soil), w, land_sea_mask=lsm)
    assert _row(out, rid).soil_moisture_pct == pytest.approx(40.0)


def test_an_all_water_district_has_no_soil_value_not_zero(two_cell_district):
    rid, w = two_cell_district
    lsm = w[["lat", "lon"]].assign(lsm=0.0)
    out = obs.to_districts_v2(_cells_daily(w), w, land_sea_mask=lsm)
    r = _row(out, rid)
    assert np.isnan(r.soil_moisture_pct)
    assert not np.isnan(r.t2m_c), "only soil is masked"


def test_a_mask_missing_a_cell_is_refused(two_cell_district):
    rid, w = two_cell_district
    lsm = w[["lat", "lon"]].iloc[1:].assign(lsm=1.0)
    with pytest.raises(ValueError, match="land-sea mask"):
        obs.to_districts_v2(_cells_daily(w), w, land_sea_mask=lsm)


# --------------------------------------------------------------- the year file

def _write_months(v2_dir: Path, cells: pd.DataFrame, year: int, months) -> None:
    v2_dir.mkdir(parents=True, exist_ok=True)
    for m in months:
        days = pd.date_range(f"{year}-{m:02d}-01", periods=pd.Period(f"{year}-{m:02d}").days_in_month)
        frames = [_cells_daily(cells).assign(date=d.date()) for d in days]
        pd.concat(frames).to_parquet(v2_dir / f"cells_daily_{year}{m:02d}.parquet", index=False)


@pytest.fixture
def v2_env(tmp_path, monkeypatch, two_cell_district):
    rid, w = two_cell_district
    monkeypatch.setattr(cds, "OUT_DIR", tmp_path)
    monkeypatch.setattr(cds, "V2_DIR", tmp_path / "_era5_cds_v2")
    monkeypatch.setattr(cds, "grid_cells", lambda: w[["lat", "lon"]])
    w[["lat", "lon"]].assign(lsm=1.0).to_parquet(tmp_path / "lsm.parquet")
    monkeypatch.setattr(cds, "land_sea_mask_path", lambda: tmp_path / "lsm.parquet")
    return tmp_path, rid, w


def test_a_year_needs_all_twelve_months(v2_env):
    root, rid, w = v2_env
    _write_months(root / "_era5_cds_v2", w, 2017, range(1, 12))
    with pytest.raises(RuntimeError, match="12"):
        cds.build_v2_district_year(2017)


def test_a_year_file_says_how_it_was_built(v2_env):
    root, rid, w = v2_env
    _write_months(root / "_era5_cds_v2", w, 2017, range(1, 13))
    path = cds.build_v2_district_year(2017)
    assert path.name == "era5_cds_v2_district_observations_india_2017.parquet"
    df = pd.read_parquet(path)
    assert df["date"].nunique() == 365
    assert (df["region_id"] == rid).sum() == 365
    assert df["source"].str.contains("estimator v2").all()
    v1_cols = (["region_id", "region_name", "state_id", "state_name",
                "latitude", "longitude", "date"] + obs.VALUE_COLUMNS + ["source"])
    assert list(df.columns) == v1_cols


def test_a_year_without_the_land_sea_mask_is_refused(v2_env, monkeypatch):
    root, rid, w = v2_env
    _write_months(root / "_era5_cds_v2", w, 2017, range(1, 13))
    monkeypatch.setattr(cds, "land_sea_mask_path", lambda: root / "missing.parquet")
    with pytest.raises(FileNotFoundError, match="land-sea"):
        cds.build_v2_district_year(2017)


# --------------------------------------------------------------- what ingest picks

def _touch(d: Path, name: str) -> Path:
    p = d / name
    p.write_bytes(b"")
    return p


def test_v2_wins_over_v1_cds(tmp_path):
    _touch(tmp_path, "era5_cds_district_observations_india_2017.parquet")
    want = _touch(tmp_path, "era5_cds_v2_district_observations_india_2017.parquet")
    assert ib.observation_file(2017, tmp_path) == want


def test_v2_wins_over_an_imd_file_built_on_v1(tmp_path):
    """An IMD-aligned file replaces rainfall inside a v1 CDS file, so its humidity and wind
    are v1's. Preferring it would quietly bring the v1 estimator back."""
    _touch(tmp_path, "imd_aligned_district_observations_india_2017.parquet")
    want = _touch(tmp_path, "era5_cds_v2_district_observations_india_2017.parquet")
    assert ib.observation_file(2017, tmp_path) == want
