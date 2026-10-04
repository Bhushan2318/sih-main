"""Retiring superseded batches: out of the store, recorded, and refused if they come back.

The retrain replaces the v1 ERA5 observations with estimator-v2 ones. The store's dedupe
keeps the newest row per key, so v2 shadows v1 wherever both have a value - but where v2 has
none (soil over an all-water district) the v1 row survives underneath. Retiring moves the
v1 batches out, writes a MOVED.md to restore from, and marks them 'retired', which
`assert_no_excluded_batches` then refuses to train beside.

PLUMBING FIXTURES: a few hand-built canonical rows per batch, for the bookkeeping only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.db.models import UploadBatch
from app.storage import parquet_store
from scripts import retire_batches as rb


def _batch(session, batch_id, filename, status="ingested", in_store=True):
    session.add(UploadBatch(id=batch_id, original_filename=filename, stored_path="-",
                            status=status))
    session.commit()
    if not in_store:
        return
    parquet_store.append_batch(batch_id, [{
        "region_id": "IN-MH-PUNE", "variable": "temperature_c", "value_type": "observed",
        "valid_date": "2016-03-01", "value": 27.0, "source_column": "t2m_c",
        "ingested_at": pd.Timestamp("2026-10-02")}])


@pytest.fixture
def store(session):
    _batch(session, "v1a", "era5_cds_district_observations_india_2016.parquet")
    _batch(session, "v1b", "era5_cds_district_observations_india_2016.parquet")
    _batch(session, "v2", "era5_cds_v2_district_observations_india_2016.parquet")
    _batch(session, "fc", "gefs_reforecast_india_2016.parquet")
    _batch(session, "imd", "imd_merged_district_observations_india_2016.parquet",
           status="quarantined", in_store=False)          # moved out on 2026-09-26
    return session


def test_selection_matches_the_pattern_and_only_ingested_batches(store):
    got = rb.select_batches(["era5_cds_district_observations_india_*.parquet"])
    assert sorted(b.id for b in got) == ["v1a", "v1b"]


def test_a_dry_run_changes_nothing(store, tmp_path):
    rb.retire(["era5_cds_district_observations_india_*.parquet"], tmp_path / "out",
              apply=False, reason="superseded")
    assert {"v1a", "v1b"} <= parquet_store.present_batch_ids()
    assert store.get(UploadBatch, "v1a").status == "ingested"
    assert not (tmp_path / "out").exists()


def test_retiring_moves_records_and_marks(store, tmp_path):
    out = tmp_path / "out"
    moved = rb.retire(["era5_cds_district_observations_india_*.parquet"], out,
                      apply=True, reason="superseded by estimator v2")
    assert sorted(moved) == ["v1a", "v1b"]
    present = parquet_store.present_batch_ids()
    assert not {"v1a", "v1b"} & present and {"v2", "fc"} <= present
    assert (out / "batch_id=v1a").is_dir()
    note = (out / "MOVED.md").read_text()
    assert "v1a" in note and "superseded by estimator v2" in note and "restore" in note.lower()
    store.expire_all()
    assert store.get(UploadBatch, "v1a").status == "retired"
    parquet_store.assert_no_excluded_batches()       # moved out: training may run


def test_a_retired_batch_put_back_stops_training(store, tmp_path):
    out = tmp_path / "out"
    rb.retire(["era5_cds_district_observations_india_*.parquet"], out, apply=True,
              reason="superseded")
    (out / "batch_id=v1a").rename(parquet_store.CANONICAL_DIR / "batch_id=v1a")
    with pytest.raises(RuntimeError, match="v1a"):
        parquet_store.assert_no_excluded_batches()


def test_an_existing_destination_is_refused_before_anything_moves(store, tmp_path):
    out = tmp_path / "out"
    (out / "batch_id=v1b").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        rb.retire(["era5_cds_district_observations_india_*.parquet"], out, apply=True,
                  reason="superseded")
    assert {"v1a", "v1b"} <= parquet_store.present_batch_ids()
    assert store.get(UploadBatch, "v1a").status == "ingested"
