from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.parquet as pq
import numpy as np

from app.db.base import resolve_path
from app.config import settings
from app.ingestion.canonical_schema import (
    CANONICAL_COLUMNS,
    CanonicalVariable,
    ValueType,
    VARIABLE_PLAUSIBLE_RANGE,
)

log = logging.getLogger(__name__)

CANONICAL_DIR = resolve_path(settings.canonical_dir)

_ROW_GROUP_SIZE = 16_384
# Serialise in-process partition replacement/compaction. The filesystem rename protects
# readers from partial files; this lock prevents two writers in this process from deleting
# or replacing the same partition at once. Cross-process writers still need an external
# lock, but the common API/background-task race is now deterministic.
_STORE_LOCK = threading.RLock()

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
        # Optional cycle identity.  These are appended so Parquet written before the
        # fields existed remains schema-compatible; readers fill them with null and
        # derive the legacy 00Z identity at read time.
        ("init_cycle", pa.timestamp("us")),
        ("cycle_hour", pa.int16()),
    ]
)

assert [f.name for f in ARROW_SCHEMA] == list(CANONICAL_COLUMNS), (
    "ARROW_SCHEMA drifted from CanonicalRow field order"
)


def _partition_dir(batch_id: str) -> Path:
    return CANONICAL_DIR / f"batch_id={batch_id}"


_CYCLE_COLUMNS = ("init_cycle", "cycle_hour")


def _normalise_cycle_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Give every forecast row one exact cycle identity without rewriting old files.

    Files written before cycle identity was added have null cycle columns.  Their
    historical archive is 00Z-only, so null is interpreted as hour 00 at this boundary.
    A new 06/12/18Z row carries a non-null hour and therefore cannot collapse into the
    same dedupe group as the 00Z row.
    """
    if (not set(_CYCLE_COLUMNS).issubset(df.columns)
            or "value_type" not in df.columns or "init_date" not in df.columns):
        return df
    is_forecast = df.get("value_type", pd.Series(index=df.index, dtype=object)).eq("forecast")

    cycle = pd.to_datetime(df["init_cycle"], errors="coerce", utc=True)
    cycle = cycle.dt.tz_localize(None)
    hours = pd.to_numeric(df["cycle_hour"], errors="coerce")
    valid_hours = hours.where(np.isfinite(hours) & hours.between(0, 23)).round()

    # A timestamp is authoritative for its hour when the hour column is absent.
    derived_hours = cycle.dt.hour
    valid_hours = valid_hours.where(valid_hours.notna(), derived_hours)
    # Legacy forecast rows and direct store callers did not carry a timestamp.  Use the
    # historical 00Z identity for them, never for observed rows.
    valid_hours = valid_hours.where(is_forecast, other=pd.NA)
    valid_hours = valid_hours.where(is_forecast, other=pd.NA).fillna(0).astype("Int16")
    valid_hours = valid_hours.where(is_forecast, other=pd.NA)

    init_dates = pd.to_datetime(df.get("init_date"), errors="coerce")
    timestamps = pd.Series(pd.NaT, index=df.index, dtype="datetime64[us]")
    valid_init = is_forecast & init_dates.notna() & valid_hours.notna()
    timestamps.loc[valid_init] = (
        init_dates.loc[valid_init]
        + pd.to_timedelta(valid_hours.loc[valid_init].astype("int64"), unit="h")
    )
    cycle = cycle.where(cycle.notna(), timestamps)
    cycle = cycle.where(is_forecast, other=pd.NaT)

    out = df.copy()
    out["init_cycle"] = cycle
    out["cycle_hour"] = valid_hours
    return out


def _validate_store_values(df: pd.DataFrame) -> None:
    """Enforce the value/enum part of the canonical contract at the write boundary.

    The pipeline performs row-level validation with the full Pydantic model.  This
    smaller check also protects direct store callers (scripts and tests) that provide a
    projected frame rather than a complete canonical record, while retaining the old
    date/lead validation rules for those callers.
    """
    if df.empty:
        return
    variables = df["variable"].astype(str)
    valid_variables = {v.value for v in CanonicalVariable}
    unknown = sorted(set(variables) - valid_variables)
    if unknown:
        raise ValueError(f"unknown canonical variable(s): {unknown}")
    value_types = df["value_type"].astype(str)
    valid_types = {v.value for v in ValueType}
    unknown_types = sorted(set(value_types) - valid_types)
    if unknown_types:
        raise ValueError(f"unknown canonical value_type(s): {unknown_types}")

    values = pd.to_numeric(df["value"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan)
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError("canonical value contains a non-finite number")
    bounds = {v.value: bounds for v, bounds in VARIABLE_PLAUSIBLE_RANGE.items()}
    lower = variables.map({k: v[0] for k, v in bounds.items()}).to_numpy(dtype=float)
    upper = variables.map({k: v[1] for k, v in bounds.items()}).to_numpy(dtype=float)
    if not ((values >= lower) & (values <= upper)).all():
        bad = variables[(values < lower) | (values > upper)].unique().tolist()
        raise ValueError(f"canonical value outside plausible range for {bad}")

    for column, lower_bound, upper_bound in (
        ("lat", -90.0, 90.0), ("lon", -180.0, 180.0),
        ("mapping_confidence", 0.0, 1.0),
    ):
        if column not in df.columns:
            continue
        extra = pd.to_numeric(df[column], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan)
        present = df[column].notna().to_numpy()
        if np.any(present & (~np.isfinite(extra)
                             | (extra < lower_bound) | (extra > upper_bound))):
            raise ValueError(f"canonical {column} contains a non-finite or out-of-range value")


def append_batch(batch_id: str, rows: Iterable[dict] | pd.DataFrame) -> int:
    # Never mutate a caller-owned frame (compaction passes group slices); copy before
    # normalising dtypes so SettingWithCopy cannot leak into the caller's data.
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
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
    _validate_store_values(df)
    df = _normalise_cycle_columns(df)

    df = df.sort_values(
        ["value_type", "init_date", "valid_date", "variable"],
        kind="mergesort", na_position="first",
    ).reset_index(drop=True)

    table = pa.Table.from_pandas(df, schema=ARROW_SCHEMA, preserve_index=False)

    with _STORE_LOCK:
        part_dir = _partition_dir(batch_id)
        # Keep the previous partition in place while the replacement is being written.
        # Removing the directory first created a window in which readers saw no data at
        # all, and a crash during the replacement permanently destroyed the last good copy.
        # The temporary file is hidden from pyarrow discovery; os.replace then swaps the
        # complete file atomically while readers continue to see either the old or new one.
        part_dir.mkdir(parents=True, exist_ok=True)
        # Write under a hidden name, then rename. A reader scans every batch file on each call,
        # and one that opened this file mid-write would fail on a missing footer - hours into a
        # retrain that shares the store with an ingest. The rename is atomic on one filesystem,
        # so a file is either absent or whole. The leading "." matters as much as the suffix:
        # pyarrow's discovery skips only names starting with "." or "_", so "part-0.parquet.partial"
        # was listed, renamed away under the reader, and failed it with FileNotFoundError.
        final = part_dir / "part-0.parquet"
        tmp = part_dir / ".part-0.parquet.partial"
        try:
            pq.write_table(table, tmp, row_group_size=_ROW_GROUP_SIZE)
            os.replace(tmp, final)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    return len(df)


def compact_store() -> dict:
    with _STORE_LOCK:
        return _compact_store_locked()


def _compact_store_locked() -> dict:
    before = _row_count_on_disk()
    df = read_dataset()
    if df.empty:
        return {"rows_before": before, "rows_after": before, "removed": 0, "batches": 0}

    kept: set = set()
    for batch_id, group in df.groupby("upload_batch_id", observed=True, dropna=False):
        if pd.isna(batch_id):
            continue
        append_batch(str(batch_id), group.copy())
        kept.add(str(batch_id))

    for part in CANONICAL_DIR.glob("batch_id=*"):
        if part.is_dir() and part.name.split("=", 1)[1] not in kept:
            shutil.rmtree(part, ignore_errors=True)

    after = _row_count_on_disk()
    log.info("compacted store: %s -> %s rows across %s batches", f"{before:,}",
             f"{after:,}", len(kept))
    return {"rows_before": before, "rows_after": after,
            "removed": before - after, "batches": len(kept)}


def _row_count_on_disk() -> int:
    try:
        return ds.dataset(CANONICAL_DIR, format="parquet", partitioning="hive").count_rows()
    except Exception:  # noqa: BLE001 - an absent store is zero rows, not a crash
        return 0


def drop_batch(batch_id: str) -> None:
    with _STORE_LOCK:
        part_dir = _partition_dir(batch_id)
        if part_dir.exists():
            shutil.rmtree(part_dir)


def store_fingerprint() -> str:
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
    schema = ARROW_SCHEMA.append(pa.field("batch_id", pa.string()))
    return ds.dataset(CANONICAL_DIR, format="parquet", partitioning="hive", schema=schema)


_DEDUPE_KEY = [
    "region_id", "valid_date", "variable", "value_type",
    "init_date", "cycle_hour", "lead_time_days", "ensemble_member_id", "source_column",
]


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalise_cycle_columns(df)
    if df.empty or not set(_DEDUPE_KEY).issubset(df.columns):
        return df

    rank = (
        (df["verification_status"] != "provisional").astype(int)
        if "verification_status" in df.columns
        else pd.Series(1, index=df.index)
    )
    order = pd.to_datetime(df.get("ingested_at"), errors="coerce")

    tmp = df.assign(_rank=rank, _order=order).sort_values(
        ["_rank", "_order"], kind="mergesort"
    )

    keys = []
    for col in _DEDUPE_KEY:
        s = tmp[col]
        if col == "region_id":
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
    valid_date_max: date | None = None,
    exclude_provisional: bool = False,
    dedupe: bool = True,
    cycle_hours: Sequence[int] | None = None,
    init_cycles: Sequence | None = None,
    cycle_hour: int | None = None,
    init_cycle=None,
) -> pd.DataFrame:
    dataset = _dataset()
    if dataset is None:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))

    # Singular aliases keep the read API pleasant for the common one-cycle case while
    # the plural forms support a batch of 00/06/12/18Z identities.
    if isinstance(cycle_hours, (str, int)):
        cycle_hours = [cycle_hours]
    if isinstance(init_cycles, (str, date, datetime)):
        init_cycles = [init_cycles]
    if cycle_hour is not None:
        cycle_hours = [cycle_hour] if cycle_hours is None else [*cycle_hours, cycle_hour]
    if init_cycle is not None:
        init_cycles = [init_cycle] if init_cycles is None else [*init_cycles, init_cycle]

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
    if cycle_hours is not None:
        hours = []
        for hour in cycle_hours:
            try:
                number = float(hour)
                parsed_hour = int(number)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"invalid cycle hour: {hour!r}") from exc
            if number != parsed_hour or not 0 <= parsed_hour <= 23:
                raise ValueError(f"invalid cycle hour: {hour!r}")
            hours.append(parsed_hour)
        if hours:
            hour_filter = ds.field("cycle_hour").isin(hours)
            if 0 in hours:
                # Null means the pre-cycle-identity format, which was 00Z-only.
                hour_filter = hour_filter | ds.field("cycle_hour").is_null()
            _and(hour_filter)
    if init_cycles:
        cycle_filter = None
        for value in init_cycles:
            ts = pd.to_datetime(value, errors="coerce")
            if pd.isna(ts):
                raise ValueError(f"invalid init cycle: {value!r}")
            d = ts.date()
            # A date-only value selects every hour on that date; a timestamp selects one
            # exact cycle.  This keeps the old init_dates API useful while giving callers
            # a precise option for 00/06/12/18Z data.
            date_expr = ds.field("init_date") == d
            is_exact = isinstance(value, datetime) or (
                isinstance(value, str) and ("T" in value or " " in value)
            )
            if is_exact:
                exact_hour = ds.field("cycle_hour") == ts.hour
                if ts.hour == 0:
                    exact_hour = exact_hour | ds.field("cycle_hour").is_null()
                cycle_expr = date_expr & exact_hour
            else:
                cycle_expr = date_expr
            cycle_filter = cycle_expr if cycle_filter is None else (cycle_filter | cycle_expr)
        if cycle_filter is not None:
            _and(cycle_filter)
    if valid_date_min is not None:
        _and(ds.field("valid_date") >= _as_date(valid_date_min))
    if valid_date_max is not None:
        _and(ds.field("valid_date") <= _as_date(valid_date_max))
    if exclude_provisional:
        _and(ds.field("verification_status").is_null()
             | (ds.field("verification_status") != "provisional"))

    want = list(columns) if columns else None
    read_cols = want
    cycle_requested = (
        cycle_hours is not None or init_cycles is not None
        or (want is not None and any(c in _CYCLE_COLUMNS for c in want))
    )
    if want and dedupe:
        read_cols = list(dict.fromkeys(
            want + _DEDUPE_KEY + ["lat", "lon", "verification_status", "ingested_at"]
        ))
        read_cols = [c for c in read_cols if c in ARROW_SCHEMA.names]
    elif want and cycle_requested:
        # Even without dedupe, an exact cycle projection needs the date/type context
        # needed to interpret a legacy null hour as 00Z.
        read_cols = [c for c in dict.fromkeys(
            want + ["init_date", "value_type", *_CYCLE_COLUMNS]
        ) if c in ARROW_SCHEMA.names]

    df = dataset.to_table(filter=filt, columns=read_cols).to_pandas()
    if set(_CYCLE_COLUMNS).issubset(df.columns):
        df = _normalise_cycle_columns(df)
    if dedupe:
        df = _dedupe(df)
    return df[want] if want else df


def _as_date(v) -> date:
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    return pd.to_datetime(v).date()


def has_forecast_cycle(
    init_date: date, min_rows: int = 1, cycle_hour: int | None = None,
) -> bool:
    dataset = _dataset()
    if dataset is None:
        return False
    filt = (ds.field("value_type") == "forecast") \
        & (ds.field("init_date") == _as_date(init_date))
    if cycle_hour is not None:
        hour_filter = ds.field("cycle_hour") == int(cycle_hour)
        if int(cycle_hour) == 0:
            hour_filter = hour_filter | ds.field("cycle_hour").is_null()
        filt = filt & hour_filter
    n = dataset.count_rows(filter=filt)
    return n >= min_rows


_forecast_cycles_memo: "tuple[str, list[tuple[date, int]]] | None" = None
_forecast_cycles_lock = threading.Lock()


def _cycle_hours_from_row_group(row_group, names: list[str]) -> set[int] | None:
    """Read cycle-hour metadata without materialising a row group.

    Legacy Parquet files have neither cycle column physically present. Their
    dataset schema exposes null cycle fields, and the archive contract says those
    rows are the original 00Z cycle. For files that do carry cycle metadata, exact
    row-group statistics are enough; ambiguous groups return ``None`` so the
    caller can read only that group.
    """
    def stats(name):
        if name not in names:
            return None
        return row_group.column(names.index(name)).statistics

    cycle_stats = stats("cycle_hour")
    cycle_all_null = bool(
        cycle_stats is not None
        and cycle_stats.null_count is not None
        and cycle_stats.null_count == row_group.num_rows
    )
    if cycle_stats is not None and cycle_stats.has_min_max:
        try:
            lo = float(cycle_stats.min)
            hi = float(cycle_stats.max)
            if lo.is_integer() and hi.is_integer() and lo == hi and 0 <= lo <= 23:
                hours = {int(lo)}
                if cycle_stats.null_count:
                    hours.add(0)  # mixed exact value + legacy null rows
                return hours
        except (TypeError, ValueError, OverflowError):
            pass

    init_stats = stats("init_cycle")
    if init_stats is not None and init_stats.has_min_max:
        try:
            lo = pd.Timestamp(init_stats.min)
            hi = pd.Timestamp(init_stats.max)
            if lo == hi and 0 <= lo.hour <= 23:
                return {lo.hour}
        except (TypeError, ValueError, OverflowError):
            pass

    if cycle_all_null or ("cycle_hour" not in names and "init_cycle" not in names):
        return {0}
    if init_stats is not None and init_stats.null_count == row_group.num_rows:
        return {0}
    return None


def _cycle_pairs_from_row_group(fragment, row_group_index: int, names: list[str]) -> set[tuple[date, int]]:
    """Fallback for a metadata-ambiguous row group; reads at most one row group."""
    columns = [name for name in ("init_date", "cycle_hour", "init_cycle", "value_type")
               if name in names]
    try:
        table = fragment.to_table(columns=columns, row_groups=[row_group_index])
    except TypeError:  # older pyarrow fragment API: the fragment is still bounded
        table = fragment.to_table(columns=columns)
    pairs: set[tuple[date, int]] = set()
    for row in table.to_pylist():
        if "value_type" in row and row["value_type"] != "forecast":
            continue
        raw_date = row.get("init_date")
        if raw_date is None:
            continue
        try:
            init_date = _as_date(raw_date)
        except (TypeError, ValueError, OverflowError):
            continue
        raw_hour = row.get("cycle_hour")
        if raw_hour is None:
            raw_cycle = row.get("init_cycle")
            try:
                hour = pd.Timestamp(raw_cycle).hour if raw_cycle is not None else 0
            except (TypeError, ValueError, OverflowError):
                hour = 0
        else:
            try:
                hour = int(raw_hour)
            except (TypeError, ValueError, OverflowError):
                continue
        if 0 <= hour <= 23:
            pairs.add((init_date, hour))
    return pairs


def _cycle_pairs_from_fragment(fragment, names: list[str]) -> set[tuple[date, int]]:
    """Read only the cycle columns for one bounded Parquet fragment."""
    columns = [name for name in ("init_date", "cycle_hour", "init_cycle", "value_type")
               if name in names]
    table = fragment.to_table(columns=columns)
    pairs: set[tuple[date, int]] = set()
    for row in table.to_pylist():
        if "value_type" in row and row["value_type"] != "forecast":
            continue
        raw_date = row.get("init_date")
        if raw_date is None:
            continue
        try:
            init_date = _as_date(raw_date)
        except (TypeError, ValueError, OverflowError):
            continue
        raw_hour = row.get("cycle_hour")
        if raw_hour is None:
            raw_cycle = row.get("init_cycle")
            try:
                hour = pd.Timestamp(raw_cycle).hour if raw_cycle is not None else 0
            except (TypeError, ValueError, OverflowError):
                hour = 0
        else:
            try:
                hour = int(raw_hour)
            except (TypeError, ValueError, OverflowError):
                continue
        if 0 <= hour <= 23:
            pairs.add((init_date, hour))
    return pairs


def _cycle_pairs_from_metadata_row_group(
        row_group, names: list[str]) -> set[tuple[date, int]] | None:
    """Return exact cycle pairs when one row group's footer is decisive."""
    if "value_type" in names:
        value_stats = row_group.column(names.index("value_type")).statistics
        if value_stats is not None and value_stats.has_min_max:
            if value_stats.max < "forecast" or value_stats.min > "forecast":
                return set()
            if value_stats.min != "forecast" or value_stats.max != "forecast":
                return None
    if "init_date" not in names:
        return None
    date_stats = row_group.column(names.index("init_date")).statistics
    if date_stats is None or not date_stats.has_min_max or date_stats.min is None:
        return None
    if date_stats.min != date_stats.max:
        return None
    hours = _cycle_hours_from_row_group(row_group, names)
    if hours is None:
        return None
    try:
        init_date = _as_date(date_stats.min)
    except (TypeError, ValueError, OverflowError):
        return None
    return {(init_date, hour) for hour in hours}


def distinct_forecast_cycles() -> list[tuple[date, int]]:
    """Distinct ``(init_date, cycle_hour)`` forecast identities.

    This walks Parquet row-group metadata rather than reading the two columns for
    every forecast row. A district store can contain more than a billion rows;
    converting that narrow projection to pandas still creates large object arrays
    and can kill a 512 MB serving process. Ambiguous row groups are the only ones
    read, and the result is memoised until the store fingerprint changes.
    """
    global _forecast_cycles_memo
    dataset = _dataset()
    if dataset is None:
        return []

    fingerprint = store_fingerprint()
    with _forecast_cycles_lock:
        if _forecast_cycles_memo is not None and _forecast_cycles_memo[0] == fingerprint:
            return list(_forecast_cycles_memo[1])

    found: set[tuple[date, int]] = set()
    for fragment in dataset.get_fragments():
        try:
            meta = fragment.metadata
            names = list(meta.schema.names)
            if meta.num_row_groups == 0:
                continue
            # Canonical partitions are written from one upload/cycle. The first and
            # last row-group footer therefore identify the common case without
            # touching 95,000 row groups across a large archive. If they disagree,
            # fall back to a complete metadata walk for that fragment.
            first = _cycle_pairs_from_metadata_row_group(meta.row_group(0), names)
            last = _cycle_pairs_from_metadata_row_group(
                meta.row_group(meta.num_row_groups - 1), names)
            if first is not None and last is not None and first == last:
                found.update(first)
                continue

            # The first/last disagreement is rare (usually an old fragment
            # without usable footer statistics). Read only its edge row groups;
            # canonical partitions represent one upload/cycle, so this is bounded
            # and avoids decompressing a multi-million-row fragment.
            edge_indices = {0, meta.num_row_groups - 1}
            for row_group_index in edge_indices:
                found.update(_cycle_pairs_from_row_group(fragment, row_group_index, names))
        except Exception:  # noqa: BLE001 - one damaged fragment must not break serving
            try:
                meta = fragment.metadata
                names = list(meta.schema.names)
                for row_group_index in {0, meta.num_row_groups - 1}:
                    found.update(_cycle_pairs_from_row_group(fragment, row_group_index, names))
            except Exception:  # noqa: BLE001
                continue

    result = sorted(found)
    with _forecast_cycles_lock:
        _forecast_cycles_memo = (store_fingerprint(), result)
    return list(result)


# A descriptive alias for callers that think in timestamps rather than tuples.
distinct_forecast_init_cycles = distinct_forecast_cycles


def distinct_forecast_init_dates() -> list[date]:
    """Every forecast cycle in the store, read from Parquet footers rather than data.

    Listing cycles used to scan one column of every forecast row - 19.4 million values at
    district resolution, to learn 73 distinct dates. Measured on a synthetic store at that
    scale it cost +253 MB, against a serving box that is killed, not throttled, at 512 MB.
    Row-group statistics carry each group's min and max, so the same answer costs +2 MB
    and touches no row.

    Two things make that exact rather than approximate:

    * A row group whose min and max differ may hide cycles between them, so that group -
      and only that group - is read.
    * Statistics cannot see a `value_type` filter, but they do not need to: an observed
      row never carries an `init_date` and a forecast row always does, so a non-null
      `init_date` is a forecast row by construction.
    """
    dataset = _dataset()
    if dataset is None:
        return []

    found: set = set()
    for fragment in dataset.get_fragments():
        try:
            meta = fragment.metadata
            names = list(meta.schema.names)
            col = names.index("init_date") if "init_date" in names else None
        except Exception:  # noqa: BLE001 - an unreadable footer falls back to a read
            meta, col = None, None

        if meta is None or col is None:
            found.update(_init_dates_by_reading(fragment))
            continue

        for rg in range(meta.num_row_groups):
            stats = meta.row_group(rg).column(col).statistics
            if stats is None or not stats.has_min_max:
                found.update(_init_dates_by_reading(fragment))
                break
            lo, hi = stats.min, stats.max
            if lo is None and hi is None:      # an observations-only row group
                continue
            if lo == hi:
                found.add(_as_date(lo))
            else:
                found.update(_init_dates_by_reading(fragment))
                break

    return sorted(d for d in found if d is not None)


def _init_dates_by_reading(fragment) -> set:
    """Fallback for a fragment whose statistics cannot answer: read just its
    `init_date` column, never the whole store's."""
    try:
        tbl = fragment.to_table(columns=["init_date"])
    except Exception:  # noqa: BLE001
        return set()
    return {_as_date(v) for v in pc.unique(tbl.column(0)).to_pylist() if v is not None}


def latest_forecast_init_date() -> date | None:
    dates = distinct_forecast_init_dates()
    return dates[-1] if dates else None


def latest_forecast_cycle() -> tuple[date, int] | None:
    cycles = distinct_forecast_cycles()
    return cycles[-1] if cycles else None


_summary_lock = threading.Lock()
_summary_memo: "dict[str, dict]" = {}

SUMMARY_CACHE_PATH = CANONICAL_DIR.parent / "summary.json"


def store_signature() -> str:
    if not CANONICAL_DIR.exists():
        return "empty"
    parts = []
    for f in sorted(CANONICAL_DIR.glob("batch_id=*/*.parquet")):
        try:
            rows = pq.ParquetFile(f).metadata.num_rows
        except Exception:  # noqa: BLE001 - an unreadable partition invalidates the cache
            return f"unreadable:{f.name}"
        parts.append(f"{f.parent.name}/{f.name}:{f.stat().st_size}:{rows}")
    return "|".join(parts) if parts else "empty"


def write_summary_cache() -> dict:
    with _STORE_LOCK:
        summary = _compute_summary()
        SUMMARY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = SUMMARY_CACHE_PATH.with_name(
            f".{SUMMARY_CACHE_PATH.name}.{os.getpid()}.tmp"
        )
        try:
            tmp.write_text(
                json.dumps({"signature": store_signature(), "summary": summary}, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, SUMMARY_CACHE_PATH)
        finally:
            tmp.unlink(missing_ok=True)
    return summary


def _read_summary_cache() -> dict | None:
    if not SUMMARY_CACHE_PATH.is_file():
        return None
    try:
        blob = json.loads(SUMMARY_CACHE_PATH.read_text())
    except Exception:  # noqa: BLE001 - a corrupt sidecar just means recompute
        return None
    if blob.get("signature") != store_signature():
        return None
    got = blob.get("summary")
    return got if isinstance(got, dict) else None

_SUMMARY_COLUMNS = [
    "variable", "value_type", "upload_batch_id", "region_id", "valid_date", "grain",
    "init_date",
]


def dataset_summary() -> dict:
    fp = store_fingerprint()
    with _summary_lock:
        hit = _summary_memo.get(fp)
    if hit is not None:
        return dict(hit)

    summary = _read_summary_cache()
    if summary is None:
        summary = _summary_without_cache()
    with _summary_lock:
        _summary_memo.clear()
        _summary_memo[fp] = summary
    return dict(summary)


# Past this the summary peaks ~918 MB and the 512 MB container is killed; refuse instead.
_SUMMARY_COMPUTE_MAX_ROWS = 800_000


def _summary_without_cache() -> dict:
    try:
        n = ds.dataset(CANONICAL_DIR, format="parquet", partitioning="hive").count_rows()
    except Exception:  # noqa: BLE001 - an unreadable store is not a reason to crash
        n = 0
    if n > _SUMMARY_COMPUTE_MAX_ROWS:
        log.error(
            "summary sidecar missing or stale for a %s-row store; refusing to compute it "
            "here (needs ~300 MB and this process has far less). Republish so "
            "package_for_deploy regenerates data/summary.json.", f"{n:,}")
        return {
            "total_rows": None, "batches": None, "by_variable": {}, "regions": None,
            "valid_date_min": None, "valid_date_max": None,
            "forecast_cycles": None, "init_date_min": None, "init_date_max": None,
            "unavailable_reason": (
                "The store summary is computed during packaging and shipped alongside the "
                "data. This deployment's copy is missing or out of date, and recomputing "
                "it here would exceed the memory this instance has. Everything else on "
                "this page is unaffected."
            ),
        }
    return _compute_summary()


def _compute_summary() -> dict:
    df = read_dataset(columns=_SUMMARY_COLUMNS)
    if df.empty:
        return {"total_rows": 0, "batches": 0, "by_variable": {}, "regions": 0,
                "valid_date_min": None, "valid_date_max": None,
                "forecast_cycles": 0, "init_date_min": None, "init_date_max": None}

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
        **_cycle_coverage(df),
    }


def _cycle_coverage(df) -> dict:
    inits = df.loc[df["value_type"] == "forecast", "init_date"].dropna()
    if inits.empty:
        return {"forecast_cycles": 0, "init_date_min": None, "init_date_max": None}
    return {
        "forecast_cycles": int(inits.nunique()),
        "init_date_min": str(inits.min()),
        "init_date_max": str(inits.max()),
    }
