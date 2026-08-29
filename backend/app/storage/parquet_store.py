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
        ("verification_status", pa.string()),
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


def store_fingerprint() -> str:
    """A cheap change-token for the canonical store: sorted partition-dir names plus their
    mtimes, no data read. Lets callers key a cache without a multi-hundred-ms parquet
    scan just to notice nothing changed."""
    if not CANONICAL_DIR.exists():
        return "empty"
    parts = sorted(
        f"{p.name}:{int(p.stat().st_mtime_ns)}"
        for p in CANONICAL_DIR.glob("batch_id=*")
        if p.is_dir()
    )
    return "|".join(parts) if parts else "empty"


def _dataset() -> ds.Dataset | None:
    if not CANONICAL_DIR.exists() or not any(CANONICAL_DIR.glob("batch_id=*/*.parquet")):
        return None
    # Partitions written before a schema addition simply lack the newer columns; giving
    # pyarrow the explicit schema makes those read back as null instead of failing on a
    # schema mismatch between partitions.
    schema = ARROW_SCHEMA.append(pa.field("batch_id", pa.string()))
    return ds.dataset(CANONICAL_DIR, format="parquet", partitioning="hive", schema=schema)


# Identity of a single reading. Two rows sharing this key are the same measurement, so a
# later ingest of it is a revision (provisional observation superseded by final ERA5, or
# a forecast cycle re-pulled) rather than an additional data point.
#
# source_column is part of the identity on purpose. A file can legitimately map two
# different columns onto one canonical variable - both ERA5 and GEFS carry mslp_hpa AND
# psfc_hpa - and those are two distinct measurements, not a duplicate. Without this,
# deduplication would silently keep one and discard the other. Resolving that ambiguity
# belongs to the schema mapper, which already routes such collisions to confirmation.
#
# The trade-off: the same data re-uploaded under DIFFERENT column names is not recognised
# as a repeat. That is accepted - every path that needs revision semantics (the live
# observation tiers, a re-pulled cycle) emits identical headers each time.
_DEDUPE_KEY = [
    "region_id", "valid_date", "variable", "value_type",
    "init_date", "lead_time_days", "ensemble_member_id", "source_column",
]


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse superseded rows, keeping the most authoritative version of each reading.

    Batches are separate partitions, so re-ingesting a day's observations adds rows
    rather than replacing them. Left alone, a provisional and a final observation for the
    same (region, date, variable) would BOTH survive and the forecast/observation merge in
    feature engineering would double every matching forecast row. Precedence is
    final > provisional, then most recently ingested.
    """
    if df.empty or not set(_DEDUPE_KEY).issubset(df.columns):
        return df

    rank = (
        (df["verification_status"] != "provisional").astype(int)
        if "verification_status" in df.columns
        else pd.Series(1, index=df.index)
    )
    order = pd.to_datetime(df.get("ingested_at"), errors="coerce")

    # Sort so the winner of each group is its last row, then take that row per group.
    tmp = df.assign(_rank=rank, _order=order).sort_values(
        ["_rank", "_order"], kind="mergesort"
    )

    # NaN never compares equal to itself, so a key column that is null for a whole class
    # of rows (observations carry no init_date, lead_time_days or member) would put every
    # such row in its own group and defeat deduplication entirely. Substitute a sentinel
    # for grouping only. Keys are built from `tmp` so they cannot mis-align with it.
    keys = []
    for col in _DEDUPE_KEY:
        s = tmp[col]
        if col == "region_id":
            # A null region_id means the point never resolved to a state - it does NOT
            # mean two such rows are the same reading. Falling back to the sentinel here
            # would merge genuinely different cities into one row and silently lose data,
            # so unresolved rows are identified by their coordinates instead.
            s = s.astype(object).where(
                s.notna(),
                tmp["lat"].astype(str) + "," + tmp["lon"].astype(str),
            )
        keys.append(s.astype(object).where(s.notna(), "\x00"))

    deduped = tmp.groupby(keys, sort=False, dropna=False).tail(1)
    return deduped.drop(columns=["_rank", "_order"]).sort_index()


def read_dataset(
    variables: Sequence[str] | None = None,
    value_types: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    init_dates: Sequence | None = None,
    valid_date_min: date | None = None,
    exclude_provisional: bool = False,
    dedupe: bool = True,
) -> pd.DataFrame:
    """Read the accumulated canonical dataset, optionally filtered. Empty DataFrame with
    the right columns if nothing has been ingested yet.

    `exclude_provisional` drops near-real-time observations that have not yet been
    replaced by ERA5 - training uses it so the models keep learning against the same
    baseline they were built on.
    """
    dataset = _dataset()
    if dataset is None:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))

    filt = None

    def _and(expr):
        nonlocal filt
        filt = expr if filt is None else (filt & expr)

    if variables:
        _and(ds.field("variable").isin(list(variables)))
    if value_types:
        _and(ds.field("value_type").isin(list(value_types)))
    if init_dates:
        _and(ds.field("init_date").isin([_as_date(d) for d in init_dates]))
    if valid_date_min is not None:
        _and(ds.field("valid_date") >= _as_date(valid_date_min))
    if exclude_provisional:
        # None (legacy / not applicable) must survive - only an explicit provisional goes.
        # `field != "provisional"` alone is NOT enough: in Arrow, as in SQL, comparing a
        # null yields null rather than true, so a bare inequality silently discards every
        # legacy row. The is_null() arm is what keeps them.
        _and(ds.field("verification_status").is_null()
             | (ds.field("verification_status") != "provisional"))

    # Deduplication needs the key columns even if the caller asked for a subset.
    want = list(columns) if columns else None
    read_cols = want
    if want and dedupe:
        read_cols = list(dict.fromkeys(
            # lat/lon are needed too: they stand in for region_id when a point never
            # resolved to a state (see _dedupe).
            want + _DEDUPE_KEY + ["lat", "lon", "verification_status", "ingested_at"]
        ))
        read_cols = [c for c in read_cols if c in ARROW_SCHEMA.names]

    df = dataset.to_table(filter=filt, columns=read_cols).to_pandas()
    if dedupe:
        df = _dedupe(df)
    return df[want] if want else df


def _as_date(v) -> date:
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    return pd.to_datetime(v).date()


def has_forecast_cycle(init_date: date, min_rows: int = 1) -> bool:
    """Has a forecast cycle for this init_date already been ingested?

    The idempotency guard for the scheduler: partitions are keyed by batch_id, so a
    blindly repeated pull would write a *second* partition for the same cycle rather than
    overwrite the first.
    """
    dataset = _dataset()
    if dataset is None:
        return False
    n = dataset.count_rows(
        filter=(ds.field("value_type") == "forecast")
        & (ds.field("init_date") == _as_date(init_date))
    )
    return n >= min_rows


def latest_forecast_init_date() -> date | None:
    dataset = _dataset()
    if dataset is None:
        return None
    tbl = dataset.to_table(
        filter=(ds.field("value_type") == "forecast"), columns=["init_date"]
    )
    if tbl.num_rows == 0:
        return None
    s = tbl.to_pandas()["init_date"].dropna()
    return None if s.empty else _as_date(s.max())


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
