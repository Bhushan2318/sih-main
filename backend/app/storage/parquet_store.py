"""The canonical dataset: an append-only, batch-partitioned parquet store.

Layout on disk (hive partitioning so pyarrow.dataset can read it back with pushdown):

    data/canonical/batch_id=<uuid>/part-0.parquet

Each upload writes exactly one partition. Re-ingesting the same batch_id overwrites only
that partition; other batches are untouched. Nothing is ever pre-joined - forecast and
observed rows sit side by side and are paired later at feature-engineering time on
(region_id, valid_date, variable).
"""

from __future__ import annotations

import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from app.db.base import resolve_path
from app.config import settings
from app.ingestion.canonical_schema import CANONICAL_COLUMNS

CANONICAL_DIR = resolve_path(settings.canonical_dir)

# Explicit Arrow schema: every partition is written with these columns, in this order and
# these types, so the whole store reads back as one consistent dataset.
ARROW_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("upload_batch_id", pa.string()),
        ("source_file", pa.string()),
        ("source_column", pa.string()),
        ("variable", pa.string()),
        ("value_type", pa.string()),
        ("value", pa.float64()),
        ("region_id", pa.string()),
        ("region_name", pa.string()),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        ("init_date", pa.date32()),
        ("valid_date", pa.date32()),
        ("lead_time_days", pa.int16()),
        ("ensemble_member_id", pa.string()),
        ("mapping_confidence", pa.float64()),
        ("ingested_at", pa.timestamp("us")),
        ("grain", pa.string()),
        ("region_resolution_method", pa.string()),
    ]
)

assert [f.name for f in ARROW_SCHEMA] == list(CANONICAL_COLUMNS), (
    "ARROW_SCHEMA drifted from CanonicalRow field order"
)


def _partition_dir(batch_id: str) -> Path:
    return CANONICAL_DIR / f"batch_id={batch_id}"


def append_batch(batch_id: str, rows: Iterable[dict] | pd.DataFrame) -> int:
    """Write one batch's canonical rows as its own partition. Returns the row count.

    Overwrites the batch's existing partition if present (idempotent re-ingest).
    """
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if df.empty:
        return 0

    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[list(CANONICAL_COLUMNS)]

    for col in ("init_date", "valid_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    df["ingested_at"] = pd.to_datetime(df["ingested_at"], errors="coerce")
    df["lead_time_days"] = df["lead_time_days"].astype("Int16")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    table = pa.Table.from_pandas(df, schema=ARROW_SCHEMA, preserve_index=False)

    part_dir = _partition_dir(batch_id)
    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, part_dir / "part-0.parquet")
    return len(df)


def drop_batch(batch_id: str) -> None:
    part_dir = _partition_dir(batch_id)
    if part_dir.exists():
        shutil.rmtree(part_dir)


def _dataset() -> ds.Dataset | None:
    if not CANONICAL_DIR.exists() or not any(CANONICAL_DIR.glob("batch_id=*/*.parquet")):
        return None
    return ds.dataset(CANONICAL_DIR, format="parquet", partitioning="hive")


def read_dataset(
    variables: Sequence[str] | None = None,
    value_types: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read the accumulated canonical dataset, optionally filtered. Empty DataFrame with
    the right columns if nothing has been ingested yet."""
    dataset = _dataset()
    if dataset is None:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))

    filt = None
    if variables:
        filt = ds.field("variable").isin(list(variables))
    if value_types:
        vt = ds.field("value_type").isin(list(value_types))
        filt = vt if filt is None else (filt & vt)

    table = dataset.to_table(filter=filt, columns=list(columns) if columns else None)
    return table.to_pandas()


def dataset_summary() -> dict:
    """Shape of the store, for /api/model/status (Phase 4) and Phase 2 smoke checks."""
    df = read_dataset()
    if df.empty:
        return {"total_rows": 0, "batches": 0, "by_variable": {}, "regions": 0,
                "valid_date_min": None, "valid_date_max": None}

    by_var = (
        df.groupby(["variable", "value_type"]).size()
        .unstack(fill_value=0).to_dict(orient="index")
    )
    return {
        "total_rows": int(len(df)),
        "batches": int(df["upload_batch_id"].nunique()),
        "by_variable": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in by_var.items()},
        "regions": int(df["region_id"].nunique(dropna=True)),
        "valid_date_min": str(df["valid_date"].min()),
        "valid_date_max": str(df["valid_date"].max()),
        "grain_counts": {k: int(v) for k, v in df["grain"].value_counts().items()},
    }
