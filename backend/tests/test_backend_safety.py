"""Focused regression tests for ingestion safety and cycle identity.

These tests deliberately stop at the ingestion/storage boundary.  They do not train a
model or fetch weather data; a green result here means untrusted input cannot silently
become a canonical row, and a 06Z cycle cannot disappear into the 00Z dedupe group.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime

import pandas as pd
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.config import settings
from app.ingestion.canonical_schema import validate_canonical_row
from app.ingestion.pipeline import (
    _reconcile_time,
    _store_raw,
    safe_upload_filename,
)


# ------------------------------------------------------------------ upload boundaries

def test_upload_filename_is_a_contained_portable_basename():
    assert safe_upload_filename("../../outside.csv") == "outside.csv"
    assert safe_upload_filename(r"C:\\fakepath\\inside.csv") == "inside.csv"
    assert safe_upload_filename("..\\..\\outside.csv") == "outside.csv"


def test_raw_storage_cannot_escape_its_batch_directory(tmp_path, monkeypatch):
    from app.ingestion import pipeline

    source = tmp_path / "source.csv"
    source.write_text("region,date,temp_c\nBihar,2026-01-01,20\n", encoding="utf-8")
    raw_root = tmp_path / "raw"
    monkeypatch.setattr(pipeline, "RAW_DIR", raw_root)

    stored = _store_raw(source, "batch-1", "../../outside.csv")

    assert stored.parent == (raw_root / "batch-1").resolve()
    assert stored.name == "outside.csv"
    assert not (tmp_path / "outside.csv").exists()


class _ChunkedUpload:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.read_sizes = []

    async def read(self, size: int):
        self.read_sizes.append(size)
        return self.chunks.pop(0) if self.chunks else b""


def test_upload_size_is_checked_while_streaming(tmp_path, monkeypatch):
    from app.api.routers import upload

    async def run():
        monkeypatch.setattr(upload, "MAX_BYTES", 5)
        monkeypatch.setattr(upload, "_READ_CHUNK_BYTES", 2)
        source = _ChunkedUpload([b"ab", b"cd", b"ef"])
        destination = tmp_path / "upload.bin"

        with pytest.raises(HTTPException) as caught:
            await upload._stream_to_temp(source, destination)
        assert "exceeds" in str(caught.value)
        assert source.read_sizes == [2, 2, 2]

    asyncio.run(run())


def test_serving_deployment_refuses_uploads_server_side(fresh_store, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    monkeypatch.setattr(settings, "allow_local_retrain", False)
    with TestClient(app) as client:
        response = client.post(
            "/api/upload",
            files={"file": ("tiny.csv", b"region,date,temp_c\nBihar,2026-01-01,20\n", "text/csv")},
        )
    assert response.status_code == 409
    assert "disabled" in response.json()["detail"].lower()


# ---------------------------------------------------------------- canonical contract

def _canonical_row(**changes):
    row = {
        "record_id": "r",
        "upload_batch_id": "b",
        "source_file": "f.csv",
        "source_column": "temp_c",
        "variable": "temperature_c",
        "value_type": "forecast",
        "value": 20.0,
        "region_id": "IN-BR",
        "init_date": date(2026, 1, 1),
        "valid_date": date(2026, 1, 1),
        "lead_time_days": 1,
        "mapping_confidence": 1.0,
        "ingested_at": datetime(2026, 1, 1),
    }
    row.update(changes)
    return row


@pytest.mark.parametrize(
    "changes",
    [
        {"variable": "not_a_weather_variable"},
        {"value_type": "analysis"},
        {"value": float("inf")},
        {"value": 1000.0},
        {"valid_date": date(2026, 1, 2)},
        {"lat": 100.0},
    ],
)
def test_canonical_contract_rejects_invalid_enums_ranges_and_finite_values(changes):
    with pytest.raises(ValidationError):
        validate_canonical_row(_canonical_row(**changes))


# ------------------------------------------------------------------------ time rules

def test_reconcile_time_uses_day_one_as_the_initialization_date():
    assert _reconcile_time(date(2026, 1, 1), None, 1) == (
        date(2026, 1, 1), date(2026, 1, 1), 1,
    )
    assert _reconcile_time(date(2026, 1, 1), None, 2) == (
        date(2026, 1, 1), date(2026, 1, 2), 2,
    )
    assert _reconcile_time(None, date(2026, 1, 2), 2) == (
        date(2026, 1, 1), date(2026, 1, 2), 2,
    )
    assert _reconcile_time(date(2026, 1, 1), date(2026, 1, 2), None) == (
        date(2026, 1, 1), date(2026, 1, 2), 2,
    )


@pytest.mark.parametrize(
    "args",
    [
        (date(2026, 1, 1), date(2026, 1, 3), 1),  # contradictory all three
        (date(2026, 1, 1), None, 0),              # invalid lead
        (date(2026, 1, 1), None, 11),             # outside Day 1-10
        (date(2026, 1, 1), None, 1.5),            # non-integral lead
        (date(2026, 1, 1), None, 25),             # non-integral forecast hours
        (None, None, None),                        # insufficient evidence
    ],
)
def test_reconcile_time_rejects_contradictory_or_invalid_rows(args):
    assert _reconcile_time(*args) is None


def test_ingestion_drops_a_contradictory_forecast_row(session, tmp_path):
    from app.ingestion.pipeline import ingest_upload
    from app.storage import parquet_store

    source = tmp_path / "contradictory.csv"
    pd.DataFrame({
        "region": ["Bihar", "Bihar"],
        "init_date": ["2026-01-01", "2026-01-01"],
        "valid_date": ["2026-01-02", "2026-01-03"],
        "lead_time_days": [2, 1],       # second row says Day 1 but is two days later
        "temp_c": [20.0, 21.0],
        "data_type": ["forecast", "forecast"],
    }).to_csv(source, index=False)
    result = ingest_upload(
        session, source, source.name,
        confirmed_mappings=[{
            "source_column": "temp_c", "variable": "temperature_c",
            "value_type": "forecast",
        }],
    )
    assert result.row_count_ingested == 1
    assert result.skipped_rows == 1
    assert len(parquet_store.read_dataset()) == 1


# --------------------------------------------------------------------- cycle identity

def _store_row(hour: int, value: float, *, source="forecast.csv"):
    return {
        "record_id": f"r-{hour}-{value}",
        "upload_batch_id": "cycle-batch",
        "source_file": source,
        "source_column": "temp_c",
        "variable": "temperature_c",
        "value_type": "forecast",
        "value": value,
        "region_id": "IN-MH",
        "init_date": date(2026, 1, 1),
        "init_cycle": datetime(2026, 1, 1, hour),
        "cycle_hour": hour,
        "valid_date": date(2026, 1, 1),
        "lead_time_days": 1,
        "ensemble_member_id": "c00",
        "mapping_confidence": 1.0,
        "ingested_at": datetime(2026, 1, 2),
        "grain": "native",
    }


def test_cycle_hour_is_part_of_dedupe_and_exact_reads(fresh_store):
    from app.storage import parquet_store

    parquet_store.append_batch("cycle-batch", [
        _store_row(0, 10.0),
        _store_row(6, 11.0),
    ])

    both = parquet_store.read_dataset()
    assert len(both) == 2
    assert set(both["cycle_hour"].astype(int)) == {0, 6}
    assert parquet_store.distinct_forecast_cycles() == [
        (date(2026, 1, 1), 0), (date(2026, 1, 1), 6),
    ]

    six = parquet_store.read_dataset(cycle_hours=[6])
    assert six["value"].tolist() == [11.0]
    exact = parquet_store.read_dataset(init_cycles=[datetime(2026, 1, 1, 6)])
    assert exact["value"].tolist() == [11.0]


def test_a_legacy_row_without_cycle_columns_reads_as_00z(fresh_store):
    from app.storage import parquet_store

    legacy = _store_row(0, 10.0)
    legacy.pop("init_cycle")
    legacy.pop("cycle_hour")
    parquet_store.append_batch("legacy-batch", [legacy])

    got = parquet_store.read_dataset()
    assert got["cycle_hour"].tolist() == [0]
    assert got["init_cycle"].tolist() == [pd.Timestamp("2026-01-01 00:00:00")]
    assert len(parquet_store.read_dataset(cycle_hours=[0])) == 1
