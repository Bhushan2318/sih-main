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


def test_a_successful_write_is_readable_at_its_final_name(fresh_store):
    n = parquet_store.append_batch("b-ok", _rows())
    files = list(parquet_store.CANONICAL_DIR.glob("batch_id=b-ok/*.parquet"))
    assert n == 3 and len(files) == 1
    assert pq.ParquetFile(files[0]).metadata.num_rows == 3
