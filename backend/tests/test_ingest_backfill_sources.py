"""Which observation file a backfill picks up.

This is the gap that kept 2017's labels at 34 districts of 666: the fetch produced
district observations, and the backfill went on reading the 35-city file because that is
the stem it looked for. Nothing errored - the store simply kept the old coverage, and a
retrain on it looked like a normal retrain.

So the preference order is pinned here: broader coverage must win, and it must win
silently-never by accident.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "ingest_backfill", BACKEND / "scripts" / "ingest_backfill.py")
ib = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ib
_spec.loader.exec_module(ib)


def _touch(d: Path, name: str) -> Path:
    p = d / name
    p.write_bytes(b"")
    return p


def test_aligned_imd_file_wins_over_plain_cds(tmp_path):
    """IMD replaces only precip_mm inside an already-fetched CDS file (see
    fetch_imd_district_rainfall.py) - when it exists it is strictly better coverage for
    the hardest variable, so it must be preferred the same way the CDS district file beat
    Open-Meteo."""
    _touch(tmp_path, "era5_cds_district_observations_india_2017.parquet")
    want = _touch(tmp_path, "imd_aligned_district_observations_india_2017.parquet")
    assert ib.observation_file(2017, tmp_path) == want


# --- Stale IMD merges -------------------------------------------------------------------
# Files named imd_merged_* were written before IMD's end-of-window date labels were
# found. They put every IMD rainfall value one model day late relative to the forecast
# it verifies. They are gitignored, so they live only on people's disks - where they
# would keep winning file selection and quietly feeding one-day-off labels to training.

def test_a_stale_imd_merge_is_refused_not_silently_used(tmp_path):
    _touch(tmp_path, "era5_cds_district_observations_india_2017.parquet")
    _touch(tmp_path, "imd_merged_district_observations_india_2017.parquet")
    with pytest.raises(RuntimeError, match="fetch_imd_district_rainfall"):
        ib.observation_file(2017, tmp_path)


def test_a_stale_imd_merge_is_refused_even_as_the_only_file(tmp_path):
    _touch(tmp_path, "imd_merged_district_observations_india_2017.csv")
    with pytest.raises(RuntimeError, match="one day"):
        ib.observation_file(2017, tmp_path)


def test_a_stale_merge_is_harmless_once_the_aligned_file_exists(tmp_path):
    _touch(tmp_path, "imd_merged_district_observations_india_2017.parquet")
    want = _touch(tmp_path, "imd_aligned_district_observations_india_2017.parquet")
    assert ib.observation_file(2017, tmp_path) == want


def test_the_imd_fetch_can_ask_for_its_era5_input_past_any_imd_file(tmp_path):
    """fetch_imd_district_rainfall.py needs the ERA5 file to merge INTO. Through the
    default lookup it would get an IMD file back - merging IMD into IMD - and a stale
    one would block the very regeneration that fixes it."""
    want = _touch(tmp_path, "era5_cds_district_observations_india_2017.parquet")
    _touch(tmp_path, "imd_merged_district_observations_india_2017.parquet")
    _touch(tmp_path, "imd_aligned_district_observations_india_2017.parquet")
    assert ib.observation_file(2017, tmp_path, include_imd=False) == want


def test_cds_district_file_wins_over_open_meteo_and_city(tmp_path):
    _touch(tmp_path, "era5_observations_india_2017.parquet")
    _touch(tmp_path, "era5_district_observations_india_2017.parquet")
    want = _touch(tmp_path, "era5_cds_district_observations_india_2017.parquet")
    assert ib.observation_file(2017, tmp_path) == want


def test_open_meteo_district_file_wins_over_city(tmp_path):
    _touch(tmp_path, "era5_observations_india_2017.parquet")
    want = _touch(tmp_path, "era5_district_observations_india_2017.parquet")
    assert ib.observation_file(2017, tmp_path) == want


def test_city_file_is_used_when_it_is_all_there_is(tmp_path):
    want = _touch(tmp_path, "era5_observations_india_2017.parquet")
    assert ib.observation_file(2017, tmp_path) == want


def test_parquet_beats_csv_for_the_same_source(tmp_path):
    _touch(tmp_path, "era5_cds_district_observations_india_2017.csv")
    want = _touch(tmp_path, "era5_cds_district_observations_india_2017.parquet")
    assert ib.observation_file(2017, tmp_path) == want


def test_a_partial_year_file_is_not_mistaken_for_the_year(tmp_path):
    """`--months 11` writes ..._2017_m11-11.parquet. Ingesting that as if it were 2017
    would train on one month while reporting a year."""
    _touch(tmp_path, "era5_cds_district_observations_india_2017_m11-11.parquet")
    got = ib.observation_file(2017, tmp_path)
    assert got is None or not got.exists(), f"partial-year file must not be picked: {got}"


def test_missing_year_reports_the_name_it_wanted(tmp_path):
    got = ib.observation_file(2017, tmp_path)
    assert got is None or not got.exists()
