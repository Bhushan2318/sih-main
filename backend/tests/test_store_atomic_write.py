"""A store file must never be visible half-written.

append_batch wrote part-0.parquet straight to its final path, and read_dataset discovers
every batch_id=*/*.parquet on each call. A retrain reading the store while an ingest is
writing to it can open a file that has no footer yet and fail with "Parquet magic bytes
not found" - hours into the run. Measured risk on 2026-09-11: the 2018 ingest wrote a file
every ~18 s while the 2017 retrain read the store in 31 bursts, each of which opens every
file's footer because the init_date filter cannot prune by path.

Plumbing tests on a three-row frame built here, never a metric.
"""
from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq
import pytest

from app.storage import parquet_store


def _rows():
    return pd.DataFrame([{
        "region_id": "IN-KL-IDUKKI", "variable": "temperature_c", "value": float(i),
        "value_type": "forecast", "init_date": pd.Timestamp("2017-12-31").date(),
        "valid_date": pd.Timestamp("2017-12-31").date(), "lead_time_days": i + 1,
        "ensemble_member_id": "c00", "verification_status": None,
    } for i in range(3)])


def test_a_failed_write_leaves_nothing_discoverable(fresh_store, monkeypatch):
    real = pq.write_table

    def half_then_die(table, where, **kw):
        # write a truncated file where the writer was told to, then crash
        with open(where, "wb") as fh:
            fh.write(b"PAR1 truncated")
        raise OSError("disk went away mid-write")

    monkeypatch.setattr(parquet_store.pq, "write_table", half_then_die)
    with pytest.raises(OSError):
        parquet_store.append_batch("b-fail", _rows())
    monkeypatch.setattr(parquet_store.pq, "write_table", real)

    visible = list(parquet_store.CANONICAL_DIR.glob("batch_id=*/*.parquet"))
    assert visible == [], f"a half-written file is discoverable: {visible}"
    assert parquet_store.read_dataset().empty


def test_a_reader_that_lists_the_store_mid_write_can_still_read_it(fresh_store, monkeypatch):
    # The first fix wrote part-0.parquet.partial beside the final file. The glob could not
    # match it, but pyarrow's discovery lists every file not starting with "." or "_", so a
    # reader that listed the store mid-write held the temp path, the writer renamed it away,
    # and to_table failed with FileNotFoundError. That killed the 2017 retrain at 43 min on
    # 2026-09-11 while the 2018 ingest wrote beside it.
    parquet_store.append_batch("b-old", _rows())
    real_replace = parquet_store.os.replace
    listed = {}

    def list_then_rename(src, dst):
        listed["dataset"] = parquet_store._dataset()
        real_replace(src, dst)

    monkeypatch.setattr(parquet_store.os, "replace", list_then_rename)
    parquet_store.append_batch("b-new", _rows())
    monkeypatch.setattr(parquet_store.os, "replace", real_replace)

    assert listed["dataset"].to_table().num_rows == 3


def test_a_failed_replacement_keeps_the_previous_partition(fresh_store, monkeypatch):
    parquet_store.append_batch("b-replace", _rows())
    real = pq.write_table

    def fail_replacement(table, where, **kw):
        with open(where, "wb") as fh:
            fh.write(b"PAR1 replacement failed")
        raise OSError("replacement write failed")

    monkeypatch.setattr(parquet_store.pq, "write_table", fail_replacement)
    with pytest.raises(OSError):
        parquet_store.append_batch("b-replace", _rows())
    monkeypatch.setattr(parquet_store.pq, "write_table", real)

    visible = list(parquet_store.CANONICAL_DIR.glob("batch_id=b-replace/*.parquet"))
    assert len(visible) == 1
    assert parquet_store.read_dataset().region_id.tolist() == ["IN-KL-IDUKKI"] * 3


def test_a_successful_write_is_readable_at_its_final_name(fresh_store):
    n = parquet_store.append_batch("b-ok", _rows())
    files = list(parquet_store.CANONICAL_DIR.glob("batch_id=b-ok/*.parquet"))
    assert n == 3 and len(files) == 1
    assert pq.ParquetFile(files[0]).metadata.num_rows == 3
