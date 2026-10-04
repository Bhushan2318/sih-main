"""A quarantined batch must stop training, and a cached year must follow the store.

On 2026-09-26 four IMD batches (one-day-late rainfall) were quarantined: their rows in
upload_batch were set to 'quarantined' and their files moved out of the store. Nothing in
the code read that status, so the protection was the file move alone - restoring the
files per their MOVED.md would have put them straight back into training, where they win
the dedupe against ERA5. And `cache_year` reused any readable `paired_{year}.parquet`,
with no link to what the store held when it was built: the 2016/2017 caches kept the IMD
rainfall after the move.

PLUMBING FIXTURES: a few hand-built canonical rows per batch, for the bookkeeping only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.db.models import UploadBatch
from app.storage import parquet_store


def _rows(valid_date, variable="temperature_c"):
    return [{"region_id": "IN-MH-PUNE", "variable": variable, "value_type": "observed",
             "valid_date": valid_date, "value": 27.0, "source_column": "t2m_c",
             "ingested_at": pd.Timestamp("2026-10-02")}]


def _batch(session, batch_id, status, valid_date):
    session.add(UploadBatch(id=batch_id, original_filename=f"{batch_id}.parquet",
                            stored_path="-", status=status))
    session.commit()
    parquet_store.append_batch(batch_id, _rows(valid_date))


def test_a_quarantined_batch_in_the_store_stops_training(session):
    _batch(session, "good", "ingested", "2016-03-01")
    _batch(session, "imd2016", "quarantined", "2016-03-01")
    with pytest.raises(RuntimeError, match="imd2016"):
        parquet_store.assert_no_excluded_batches()


def test_retired_batches_are_refused_too(session):
    _batch(session, "v1obs", "retired", "2016-03-01")
    with pytest.raises(RuntimeError, match="v1obs"):
        parquet_store.assert_no_excluded_batches()


def test_a_quarantined_batch_moved_out_of_the_store_is_fine(session):
    _batch(session, "good", "ingested", "2016-03-01")
    session.add(UploadBatch(id="moved", original_filename="m.parquet", stored_path="-",
                            status="quarantined"))
    session.commit()
    parquet_store.assert_no_excluded_batches()          # in the DB, not on disk: no error


def test_building_training_frames_checks_the_store_first(session, monkeypatch):
    from app.ml import train_pipeline as tp
    called = []

    def refuse():
        called.append(1)
        raise RuntimeError("quarantined batch present")
    monkeypatch.setattr(parquet_store, "assert_no_excluded_batches", refuse)
    with pytest.raises(RuntimeError, match="quarantined"):
        tp._build_paired_in_chunks()
    assert called


# --------------------------------------------------------------- what a cached year was built from

def test_batch_spans_come_from_footers(session):
    _batch(session, "a", "ingested", "2016-03-01")
    parquet_store.append_batch("a2", _rows("2017-07-04") + _rows("2017-01-02"))
    spans = parquet_store.batch_spans()
    assert spans["a"]["min"] == pd.Timestamp("2016-03-01").date()
    assert spans["a2"]["min"] == pd.Timestamp("2017-01-02").date()
    assert spans["a2"]["max"] == pd.Timestamp("2017-07-04").date()
    assert spans["a2"]["rows"] == 2


def test_a_years_signature_covers_only_batches_that_reach_it(session):
    _batch(session, "y2016", "ingested", "2016-06-01")
    _batch(session, "y2018", "ingested", "2018-06-01")
    _batch(session, "edge", "ingested", "2017-01-05")   # inside 2016's Day-10 reach
    sig = parquet_store.year_batch_signature(2016)
    ids = {b["batch_id"] for b in sig}
    assert ids == {"y2016", "edge"}


def test_a_cache_built_from_another_store_state_is_rebuilt(session, tmp_path, monkeypatch):
    from app.ml import pooled_training as pt
    _batch(session, "y2000", "ingested", "2000-06-01")
    built = []

    def fake_build(init_date_min=None, init_date_max=None, feature_version=None):
        built.append(1)
        return pd.DataFrame({"init_date": [pd.Timestamp("2000-06-01")], "x": [1.0]}), 1
    monkeypatch.setattr(pt, "_build_paired_in_chunks", fake_build)
    pt.cache_year(2000, tmp_path)
    pt.cache_year(2000, tmp_path)
    assert len(built) == 1, "an unchanged store reuses the cache"
    parquet_store.append_batch("late", _rows("2000-07-01"))  # a new batch for that year
    pt.cache_year(2000, tmp_path)
    assert len(built) == 2, "a store change for the year must rebuild it"
    parquet_store.append_batch("other", _rows("2005-07-01"))  # another year only
    pt.cache_year(2000, tmp_path)
    assert len(built) == 2, "a batch for another year does not touch this cache"
