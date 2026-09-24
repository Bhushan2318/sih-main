from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.config import settings
from app.db import crud
from app.db.base import get_session, init_db, resolve_path
from app.ingestion.canonical_schema import (
    CanonicalVariable,
    ValueType,
    validate_canonical_row,
)
from app.ingestion.parsers import ParsedTable, parse_upload
from app.ingestion import schema_mapper as sm
from app.storage import parquet_store
from app.utils.geo import get_resolver

RAW_DIR = resolve_path(settings.raw_upload_dir)

_KMH_TO_MS = 1.0 / 3.6
_INVALID = object()
_CYCLE_HOUR_RE = re.compile(r"(?:^|[_-])(?P<hour>00|06|12|18)z(?:[._-]|$)", re.IGNORECASE)
_ALLOWED_ROLES = {
    sm.ROLE_MEASUREMENT,
    sm.ROLE_DIMENSION,
    sm.ROLE_VALUE,
    sm.ROLE_VALUE_TYPE,
    sm.ROLE_VARIABLE_NAME,
    sm.ROLE_UNMAPPED,
}
_ALLOWED_CONVERSIONS = {None, "kmh_to_ms", "K_to_C", "Pa_to_hPa", "frac_to_pct"}


class MappingValidationError(ValueError):
    """A user/profile mapping cannot describe a canonical row."""


def safe_upload_filename(filename: str | Path | None, *, default: str = "upload.csv") -> str:
    """Return a single, portable basename for an uploaded/ingested file.

    Multipart clients commonly send either POSIX or Windows path components, and a
    filename is untrusted input.  Strip both separators before taking the basename so
    this behaves the same on every platform; the raw-upload destination then has one
    containment boundary rather than a platform-dependent one.  A missing/empty name
    gets the explicit default rather than a path named ``.``.
    """
    raw = str(filename or "").strip()
    if not raw:
        raw = default
    raw = raw.replace("\\", "/")
    name = PurePosixPath(raw).name
    if name in {"", ".", ".."} or "\x00" in name:
        raise ValueError("upload filename must name a file")
    # Keep the audit field bounded to the DB column as well as contained on disk.
    if len(name) > 512:
        name = name[-512:]
    return name


def cycle_hour_from_filename(filename: str | None) -> Optional[int]:
    """Read the GEFS cycle hour from the operational filename, if present."""
    if not filename:
        return None
    match = _CYCLE_HOUR_RE.search(safe_upload_filename(filename))
    return int(match.group("hour")) if match else None


def _is_missing_scalar(value) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
        return isinstance(missing, (bool, np.bool_)) and bool(missing)
    except (TypeError, ValueError):
        return False


def _coerce_datetime(value) -> Optional[datetime]:
    if _is_missing_scalar(value):
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(ts):
        return None
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_pydatetime()


def _coerce_cycle_hour(value) -> int | object:
    """Coerce a user-supplied hour, returning ``_INVALID`` rather than guessing."""
    if _is_missing_scalar(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if value.isdigit():
            value = int(value)
    if isinstance(value, (bool, np.bool_)):
        return _INVALID
    try:
        number = float(value)
        hour = int(number)
    except (TypeError, ValueError, OverflowError):
        return _INVALID
    if not np.isfinite(number) or number != hour or not 0 <= hour <= 23:
        return _INVALID
    return hour


def _coerce_lead_days(value) -> int | object:
    """Normalise day or forecast-hour lead values, rejecting non-integral values.

    Upload sources use both ``lead_day`` (1..10) and ``forecast_hour`` (24..240).
    The latter is converted only when it is an exact multiple of 24; a value such as
    25 is contradictory, not a lead of one day.
    """
    if _is_missing_scalar(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return _INVALID
    if not np.isfinite(number):
        return _INVALID
    days = number / 24.0 if number >= 24 else number
    rounded = round(days)
    if abs(days - rounded) > 1e-9 or not 1 <= rounded <= 10:
        return _INVALID
    return int(rounded)


def _cycle_timestamp(init_d: date, cycle_hour: Optional[int]) -> Optional[datetime]:
    if init_d is None or cycle_hour is None:
        return None
    return datetime.combine(init_d, time(hour=cycle_hour))


@dataclass
class IngestResult:
    batch_id: str
    status: str
    layout: Optional[str] = None
    detected_format: Optional[str] = None
    row_count_raw: int = 0
    row_count_ingested: int = 0
    skipped_rows: int = 0
    canonical_variables_found: list = field(default_factory=list)
    region_resolution_rate: float = 0.0
    profile_match: str = "none"
    mapping_proposals: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    should_retrain: bool = False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _proposal_to_dict(p: sm.ColumnProposal) -> dict:
    return {
        "source_column": p.source_column,
        "normalized": p.normalized,
        "role": p.role,
        "sample_values": p.sample_values,
        "suggested_variable": p.suggested_variable,
        "suggested_value_type": p.suggested_value_type,
        "confidence": p.confidence,
        "ambiguity_gap": p.ambiguity_gap,
        "method": p.method,
        "unit_conversion": p.unit_conversion,
        "decision": p.decision,
        "alternatives": p.alternatives,
    }


def _mapping_rows_for_db(result: sm.MappingResult) -> list[dict]:
    rows = []
    for p in result.proposals:
        rows.append(
            dict(
                source_column=p.source_column,
                source_header_normalized=p.normalized,
                role=p.role,
                mapped_variable=p.suggested_variable if p.role in ("measurement",) else None,
                mapped_value_type=p.suggested_value_type,
                decision=p.decision,
                method=p.method,
                confidence=float(p.confidence or 0.0),
                ambiguity_gap=float(p.ambiguity_gap or 0.0),
                unit_conversion=p.unit_conversion,
                sample_values_json=p.sample_values,
            )
        )
    return rows


def _dimension_columns(result: sm.MappingResult) -> dict:
    out = {}
    for p in result.proposals:
        if p.role == sm.ROLE_DIMENSION and p.suggested_variable:
            out.setdefault(p.suggested_variable, p.source_column)
    return out


def _apply_unit(value: float, conversion: Optional[str]) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if conversion == "kmh_to_ms":
        return value * _KMH_TO_MS
    if conversion == "K_to_C":
        return value - 273.15
    if conversion == "Pa_to_hPa":
        return value / 100.0
    if conversion == "frac_to_pct":
        return value * 100.0
    return value


def _coerce_date(v) -> Optional[date]:
    dt = _coerce_datetime(v)
    return None if dt is None else dt.date()


def _coerce_date_checked(v) -> date | object | None:
    """Coerce a date while distinguishing missing from present-but-invalid input."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    dt = _coerce_datetime(v)
    return _INVALID if dt is None else dt.date()


def _reconcile_time(
    init_d: Optional[date],
    valid_d: Optional[date],
    lead_days: int | object | None,
) -> Optional[tuple[date, date, int]]:
    """Reconcile the three forecast-time fields under the Day-k convention.

    ``valid_date`` is ``init_date + (lead_time_days - 1)``.  A missing member may be
    derived when the other two are present, but all three are checked when supplied:
    silently preferring a derived value would turn a contradictory upload into a
    plausible-looking forecast.  ``None`` means the row is invalid and must be refused.
    """
    if lead_days is _INVALID or init_d is _INVALID or valid_d is _INVALID:
        return None
    if init_d is not None:
        init_d = _coerce_date(init_d)
    if valid_d is not None:
        valid_d = _coerce_date(valid_d)
    lead = _coerce_lead_days(lead_days)
    if lead is _INVALID:
        return None
    if lead is not None and not 1 <= lead <= 10:
        return None

    have = sum(x is not None for x in (init_d, valid_d, lead))
    if have < 2:
        return None
    if valid_d is None:
        valid_d = init_d + timedelta(days=lead - 1)
    elif init_d is None:
        init_d = valid_d - timedelta(days=lead - 1)
    else:
        inferred = (valid_d - init_d).days + 1
        if lead is not None and inferred != lead:
            return None
        lead = inferred
    if not 1 <= lead <= 10 or valid_d != init_d + timedelta(days=lead - 1):
        return None
    return init_d, valid_d, lead


def to_canonical_rows(
    df: pd.DataFrame,
    result: sm.MappingResult,
    *,
    batch_id: str,
    source_file: str,
    grain: str,
    filename_hint: str = "",
    cycle_hour: int | str | None = None,
) -> tuple[pd.DataFrame, int, list]:
    notes: list = []
    resolver = get_resolver()
    ingested_at = datetime.now(timezone.utc)
    dims = _dimension_columns(result)

    region_col = dims.get("region")
    lat_col, lon_col = dims.get("lat"), dims.get("lon")
    valid_col, init_col, lead_col = dims.get("valid_date"), dims.get("init_date"), dims.get("lead_time_days")
    member_col = dims.get("ensemble_member_id")
    cycle_col = dims.get("cycle_hour")
    init_cycle_col = dims.get("init_cycle")
    vt_col = result.value_type_column

    supplied_cycle_hour = _coerce_cycle_hour(cycle_hour)
    if supplied_cycle_hour is _INVALID:
        raise ValueError("cycle_hour must be an integer from 0 to 23")
    if supplied_cycle_hour is None:
        supplied_cycle_hour = cycle_hour_from_filename(filename_hint)

    plan: list[tuple] = []
    if result.layout == "long":
        var_name_col = next((p.source_column for p in result.proposals
                             if p.role == sm.ROLE_VARIABLE_NAME), None)
        value_cols = [(p.source_column, p.suggested_value_type)
                      for p in result.proposals if p.role == sm.ROLE_VALUE]
        if not var_name_col or not value_cols:
            notes.append("long layout but no variable-name/value columns resolved; nothing ingested")
            return pd.DataFrame(), len(df), notes
    else:
        for scol, spec in result.measurement_map().items():
            plan.append((scol, spec["variable"], spec["value_type"], spec.get("unit_conversion")))
        if not plan:
            notes.append("no accepted measurement columns; nothing ingested")
            return pd.DataFrame(), len(df), notes

    _region_cache: dict = {}

    def _resolve_region(name, lat, lon):
        key = (name, None if lat is None else round(lat, 3), None if lon is None else round(lon, 3))
        if key not in _region_cache:
            _region_cache[key] = resolver.resolve(name=name, lat=lat, lon=lon)
        return _region_cache[key]

    def _num(row, col):
        if not col or col not in row:
            return None
        try:
            v = pd.to_numeric(row[col], errors="coerce")
            number = float(v)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if np.isfinite(number) else None

    def _norm_value_type(raw) -> Optional[str]:
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return None
        n = sm.normalize_header(str(raw))
        for vt, syns in sm.VALUE_TYPE_SYNONYMS.items():
            if any(s in n or n in s for s in syns):
                return vt.value
        return None

    out_rows: list[dict] = []
    skipped = 0
    invalid_time = 0
    invalid_mapping = 0
    resolved_regions = 0
    total_region_attempts = 0

    records = df.to_dict("records")
    for row in records:
        name = str(row[region_col]).strip() if region_col and pd.notna(row.get(region_col)) else None
        lat, lon = _num(row, lat_col), _num(row, lon_col)
        init_raw = row.get(init_col) if init_col else None
        valid_raw = row.get(valid_col) if valid_col else None
        init_checked = _coerce_date_checked(init_raw)
        valid_checked = _coerce_date_checked(valid_raw)
        date_invalid = init_checked is _INVALID or valid_checked is _INVALID
        init_dt = _coerce_datetime(init_raw)
        init_d = init_checked if isinstance(init_checked, date) else None
        valid_d = valid_checked if isinstance(valid_checked, date) else None
        lead = _coerce_lead_days(row.get(lead_col)) if lead_col else None
        member = str(row[member_col]).strip() if member_col and pd.notna(row.get(member_col)) else None
        row_vt = _norm_value_type(row.get(vt_col)) if vt_col else None

        row_cycle = _coerce_cycle_hour(row.get(cycle_col)) if cycle_col else None
        row_init_cycle = _coerce_datetime(row.get(init_cycle_col)) if init_cycle_col else None
        if row_init_cycle is None and init_dt is not None and init_dt.hour:
            row_cycle = init_dt.hour
        if row_init_cycle is not None:
            derived_row_cycle = row_init_cycle.hour
            if init_d is not None and row_init_cycle.date() != init_d:
                row_cycle = _INVALID
            else:
                row_cycle = derived_row_cycle
        if row_cycle is _INVALID:
            cycle_invalid = True
        else:
            cycle_invalid = date_invalid
        if supplied_cycle_hour is not None and row_cycle is not None \
                and row_cycle is not _INVALID and row_cycle != supplied_cycle_hour:
            cycle_invalid = True
        effective_cycle_hour = supplied_cycle_hour
        if effective_cycle_hour is None and row_cycle not in (None, _INVALID):
            effective_cycle_hour = row_cycle

        rm = _resolve_region(name, lat, lon)
        total_region_attempts += 1
        if rm.region_id is not None:
            resolved_regions += 1

        emissions: list[tuple] = []
        if result.layout == "long":
            var_label = row.get(var_name_col)
            variable = _match_variable(var_label)
            if variable is None:
                skipped += 1
                continue
            for vcol, vcol_vt in value_cols:
                raw = _num(row, vcol)
                if raw is None:
                    continue
                vt = row_vt or vcol_vt
                emissions.append((variable, vt, raw, None, vcol))
        else:
            for scol, variable, col_vt, unit in plan:
                raw = _num(row, scol)
                if raw is None:
                    continue
                vt = row_vt or col_vt
                emissions.append((variable, vt, raw, unit, scol))

        for variable, vt, raw, unit, scol in emissions:
            try:
                variable = CanonicalVariable(variable).value
                value_type = ValueType(vt)
            except (TypeError, ValueError):
                invalid_mapping += 1
                skipped += 1
                continue

            if value_type is ValueType.FORECAST:
                if cycle_invalid:
                    invalid_time += 1
                    skipped += 1
                    continue
                reconciled = _reconcile_time(init_d, valid_d, lead)
                if reconciled is None:
                    invalid_time += 1
                    skipped += 1
                    continue
                row_init_d, row_valid_d, row_lead = reconciled
                # Historical files predate cycle metadata and were 00Z-only.  Defaulting
                # those rows to 00 preserves their old identity while allowing new 06/12/18Z
                # rows to remain distinct.
                row_hour = 0 if effective_cycle_hour is None else effective_cycle_hour
                init_cycle = _cycle_timestamp(row_init_d, row_hour)
            else:
                # Observations have no forecast cycle.  They may be accompanied by stale
                # init/lead columns in a mixed file, but those fields never become canonical.
                if valid_d is None:
                    skipped += 1
                    continue
                row_init_d, row_valid_d, row_lead = None, valid_d, None
                init_cycle = None
                row_hour = None

            value = _apply_unit(raw, unit)
            confidence = _confidence_for(result, scol)
            if value is None or not np.isfinite(value) or not np.isfinite(confidence):
                invalid_mapping += 1
                skipped += 1
                continue
            candidate = dict(
                record_id=str(uuid.uuid4()),
                upload_batch_id=batch_id,
                source_file=source_file,
                source_column=scol,
                variable=variable,
                value_type=value_type.value,
                value=float(value),
                region_id=rm.region_id,
                region_name=rm.region_name or name,
                lat=lat,
                lon=lon,
                init_date=row_init_d,
                valid_date=row_valid_d,
                lead_time_days=row_lead,
                ensemble_member_id=member,
                mapping_confidence=float(confidence),
                ingested_at=ingested_at,
                grain=grain,
                region_resolution_method=rm.method,
                init_cycle=init_cycle,
                cycle_hour=row_hour,
            )
            try:
                validate_canonical_row(candidate)
            except ValidationError as exc:
                invalid_mapping += 1
                skipped += 1
                # Keep the reason in the batch note, but do not echo an entire source row
                # (and potentially a large/untrusted value) into logs or API responses.
                reason = str(exc).splitlines()[0]
                if reason not in notes:
                    notes.append(f"rejected invalid canonical value: {reason}")
                continue
            out_rows.append(candidate)

    if total_region_attempts:
        rate = resolved_regions / total_region_attempts
        notes.append(f"region resolution: {resolved_regions}/{total_region_attempts} "
                     f"({rate:.0%}) via name/point-in-polygon")
    if invalid_time:
        notes.append(f"rejected {invalid_time} forecast row-value(s) with invalid or contradictory dates/leads")
    if skipped:
        detail = "no valid_date, value_type signal, or canonical value"
        if invalid_time:
            detail += f"; {invalid_time} invalid/contradictory time rows"
        notes.append(f"skipped {skipped} row-values ({detail})")

    return pd.DataFrame(out_rows), skipped, notes


def _match_variable(label) -> Optional[str]:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    n = sm.normalize_header(str(label))
    for var, syns in sm.VARIABLE_SYNONYMS.items():
        if n in syns or any(sm._tok(s) == sm._tok(n) or sm._tok(s) <= sm._tok(n) for s in syns):
            return var.value
    return None


def _confidence_for(result: sm.MappingResult, source_column: str) -> float:
    for p in result.proposals:
        if p.source_column == source_column:
            return float(p.confidence or 0.0)
    return 0.0


def _load_profiles(session: Session):
    return crud.all_source_profiles(session)


def _store_raw(path: Path, batch_id: str, original_filename: str) -> Path:
    # ``batch_id`` is generated by the application, but resolve and check the final path
    # anyway.  This is a second boundary for callers that bypass the HTTP route.
    raw_root = Path(RAW_DIR).resolve()
    dest_dir = (raw_root / safe_upload_filename(batch_id, default="batch")).resolve()
    try:
        dest_dir.relative_to(raw_root)
    except ValueError as exc:
        raise ValueError("upload batch directory escapes the raw-upload root") from exc
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = (dest_dir / safe_upload_filename(original_filename)).resolve()
    try:
        dest.relative_to(dest_dir)
    except ValueError as exc:
        raise ValueError("upload filename escapes the raw-upload directory") from exc
    shutil.copy2(path, dest)
    return dest


def _finish_canonicalization(
    session: Session,
    batch,
    parsed: ParsedTable,
    result: sm.MappingResult,
    verification_status: Optional[str] = None,
    cycle_hour: int | str | None = None,
) -> IngestResult:
    canon_df, skipped, notes = to_canonical_rows(
        parsed.df,
        result,
        batch_id=batch.id,
        source_file=batch.original_filename,
        grain=parsed.grain,
        filename_hint=batch.original_filename,
        cycle_hour=cycle_hour,
    )
    if verification_status and not canon_df.empty:
        canon_df["verification_status"] = canon_df["value_type"].map(
            lambda vt: verification_status if vt == "observed" else None
        )
    n = parquet_store.append_batch(batch.id, canon_df)
    variables = sorted(canon_df["variable"].unique().tolist()) if not canon_df.empty else []

    crud.replace_column_mappings(session, batch.id, _mapping_rows_for_db(result))
    confirmed_map = {
        p.source_column: {
            "variable": p.suggested_variable,
            "value_type": p.suggested_value_type,
            "role": p.role,
            "unit_conversion": p.unit_conversion,
        }
        for p in result.proposals
        if p.decision in ("auto_accept", "confirmed") and p.role != sm.ROLE_UNMAPPED
    }
    profile = crud.upsert_source_profile(
        session,
        fingerprint=result.fingerprint,
        headers=parsed.headers,
        confirmed_mapping=confirmed_map,
        file_format=parsed.detected_format,
        layout=result.layout,
    )

    rate = 0.0
    if not canon_df.empty:
        rate = float((canon_df["region_id"].notna()).mean())

    all_notes = parsed.parse_notes + result.notes + notes
    crud.update_batch(
        session,
        batch,
        status="ingested",
        detected_format=parsed.detected_format,
        layout=result.layout,
        grain=parsed.grain,
        canonical_row_count=n,
        skipped_row_count=skipped,
        source_profile_id=profile.id,
        notes_json=all_notes,
    )
    return IngestResult(
        batch_id=batch.id,
        status="ingested",
        layout=result.layout,
        detected_format=parsed.detected_format,
        row_count_raw=batch.row_count_raw or 0,
        row_count_ingested=n,
        skipped_rows=skipped,
        canonical_variables_found=variables,
        region_resolution_rate=rate,
        profile_match=result.profile_match,
        notes=all_notes,
        should_retrain=n > 0,
    )


def ingest_upload(
    session: Session,
    path: Path | str,
    original_filename: Optional[str] = None,
    confirmed_mappings: Optional[list] = None,
    verification_status: Optional[str] = None,
    cycle_hour: int | str | None = None,
) -> IngestResult:
    path = Path(path)
    original_filename = safe_upload_filename(original_filename or path.name)
    if cycle_hour is not None:
        parsed_cycle_hour = _coerce_cycle_hour(cycle_hour)
        if parsed_cycle_hour is _INVALID:
            raise ValueError("cycle_hour must be an integer from 0 to 23")
        cycle_hour = parsed_cycle_hour

    batch = crud.create_upload_batch(
        session,
        original_filename=original_filename,
        stored_path="",
        content_sha256=_sha256(path),
    )
    try:
        stored = _store_raw(path, batch.id, original_filename)
        crud.update_batch(session, batch, stored_path=str(stored))

        parsed = parse_upload(stored, original_filename)
        crud.update_batch(session, batch, row_count_raw=int(len(parsed.df)),
                          detected_format=parsed.detected_format, grain=parsed.grain)

        mapper = sm.SchemaMapper(filename_hint=original_filename)
        result = mapper.map_table(parsed.df, existing_profiles=_load_profiles(session))

        if confirmed_mappings is not None:
            _apply_confirmations(result, confirmed_mappings)

        if confirmed_mappings is None and not result.auto_accepted and result.profile_match != "exact":
            crud.replace_column_mappings(session, batch.id, _mapping_rows_for_db(result))
            crud.update_batch(session, batch, status="pending_confirmation",
                              layout=result.layout,
                              notes_json=parsed.parse_notes + result.notes)
            return IngestResult(
                batch_id=batch.id,
                status="pending_confirmation",
                layout=result.layout,
                detected_format=parsed.detected_format,
                row_count_raw=int(len(parsed.df)),
                profile_match=result.profile_match,
                mapping_proposals=[_proposal_to_dict(p) for p in result.proposals],
                notes=parsed.parse_notes + result.notes,
            )

        return _finish_canonicalization(
            session, batch, parsed, result, verification_status, cycle_hour=cycle_hour
        )

    except Exception as exc:  # noqa: BLE001 - record then re-raise
        crud.update_batch(session, batch, status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


def confirm_mapping(session: Session, batch_id: str, mappings: list) -> IngestResult:
    batch = crud.get_upload_batch(session, batch_id)
    if batch is None:
        raise ValueError(f"unknown batch {batch_id}")

    parsed = parse_upload(Path(batch.stored_path), batch.original_filename)
    result = sm.SchemaMapper(filename_hint=batch.original_filename).map_table(
        parsed.df, existing_profiles=_load_profiles(session)
    )
    _apply_confirmations(result, mappings)
    crud.update_batch(session, batch, status="canonicalizing")
    return _finish_canonicalization(
        session, batch, parsed, result,
        cycle_hour=cycle_hour_from_filename(batch.original_filename),
    )


def _apply_confirmations(result: sm.MappingResult, mappings: list) -> None:
    """Apply a user mapping after validating it as a canonical mapping."""
    proposal_columns = {p.source_column for p in result.proposals}
    by_col: dict[str, dict] = {}
    for raw in mappings or []:
        if not isinstance(raw, dict):
            raise MappingValidationError("each mapping must be an object")
        source = raw.get("source_column")
        if not isinstance(source, str) or not source or source not in proposal_columns:
            raise MappingValidationError(f"mapping refers to an unknown source column: {source!r}")
        if source in by_col:
            raise MappingValidationError(f"duplicate mapping for source column {source!r}")
        role = raw.get("role") or sm.ROLE_MEASUREMENT
        if role not in _ALLOWED_ROLES:
            raise MappingValidationError(f"unsupported mapping role: {role!r}")
        conversion = raw.get("unit_conversion")
        if conversion not in _ALLOWED_CONVERSIONS:
            raise MappingValidationError(f"unsupported unit conversion: {conversion!r}")
        variable = raw.get("variable")
        value_type = raw.get("value_type")
        if role == sm.ROLE_MEASUREMENT and variable not in (None, "", "unmapped"):
            try:
                variable = CanonicalVariable(variable).value
            except (TypeError, ValueError) as exc:
                raise MappingValidationError(f"unknown canonical variable: {variable!r}") from exc
        elif role != sm.ROLE_MEASUREMENT:
            # Dimension/value-type columns are allowed to carry no measurement mapping,
            # but a non-null enum value must still be real if the client supplied one.
            variable = variable or None
        if value_type not in (None, "", "unmapped"):
            try:
                value_type = ValueType(value_type).value
            except (TypeError, ValueError) as exc:
                raise MappingValidationError(f"unknown canonical value_type: {value_type!r}") from exc
        else:
            value_type = None
        by_col[source] = {
            **raw,
            "role": role,
            "variable": variable,
            "value_type": value_type,
            "unit_conversion": conversion,
        }

    vt_col = result.value_type_column
    if vt_col is not None:
        m = by_col.get(vt_col)
        if m is not None and (m.get("role") == sm.ROLE_UNMAPPED
                              or m.get("variable") in (None, "", "unmapped")):
            result.value_type_column = None
            result.notes.append(
                f"value_type column '{vt_col}' was excluded by confirmation; "
                "value_type now comes from each column's confirmed mapping"
            )

    for p in result.proposals:
        m = by_col.get(p.source_column)
        if not m:
            if p.decision == "needs_confirmation":
                p.decision = "unmapped"
                p.role = sm.ROLE_UNMAPPED
            continue
        if m["role"] == sm.ROLE_UNMAPPED or m["variable"] in (None, "", "unmapped"):
            p.decision, p.role = "unmapped", sm.ROLE_UNMAPPED
            continue
        p.role = m["role"]
        p.suggested_variable = m["variable"] or p.suggested_variable
        p.suggested_value_type = m["value_type"] or p.suggested_value_type
        p.unit_conversion = m["unit_conversion"]
        if p.role == sm.ROLE_MEASUREMENT and not p.suggested_variable:
            raise MappingValidationError(
                f"measurement column {p.source_column!r} needs a canonical variable"
            )
        p.decision, p.method = "confirmed", "manual"

    accepted: dict = {}
    for p in result.proposals:
        if p.role == sm.ROLE_MEASUREMENT and p.decision in ("auto_accept", "confirmed") \
                and p.suggested_variable:
            accepted.setdefault(
                (p.suggested_variable, p.suggested_value_type), []
            ).append(p.source_column)
    for (variable, _vt), cols in accepted.items():
        if len(cols) > 1:
            result.notes.append(
                f"{len(cols)} columns ingested as {variable} ({', '.join(cols)}) - "
                "kept as distinct series (dedupe keys on source column)"
            )


def _main() -> None:
    ap = argparse.ArgumentParser(description="ingest one file into the canonical store")
    ap.add_argument("--file", required=True)
    ap.add_argument("--confirm-all", action="store_true",
                    help="accept every needs_confirmation proposal as-is (demo shortcut)")
    ap.add_argument("--reset", action="store_true", help="wipe metadata DB + canonical store first")
    args = ap.parse_args()

    if args.reset:
        from app.db.base import engine
        from app.db.models import Base
        Base.metadata.drop_all(engine)
        if parquet_store.CANONICAL_DIR.exists():
            shutil.rmtree(parquet_store.CANONICAL_DIR)

    init_db()
    src = Path(args.file)

    with get_session() as session:
        res = ingest_upload(session, src, src.name)
        if res.status == "pending_confirmation" and args.confirm_all:
            seen: dict = {}
            confirmations = []
            cands = sorted(
                (p for p in res.mapping_proposals
                 if p["role"] == "measurement" and p["decision"] == "needs_confirmation"
                 and p["suggested_variable"]),
                key=lambda p: p["confidence"], reverse=True,
            )
            for p in cands:
                key = (p["suggested_variable"], p["suggested_value_type"])
                if key in seen:
                    continue
                seen[key] = p["source_column"]
                confirmations.append({
                    "source_column": p["source_column"],
                    "variable": p["suggested_variable"],
                    "value_type": p["suggested_value_type"],
                    "unit_conversion": p["unit_conversion"],
                })
            res = confirm_mapping(session, res.batch_id, confirmations)

    print(f"\nbatch {res.batch_id}  status={res.status}  layout={res.layout}  format={res.detected_format}")
    print(f"raw rows={res.row_count_raw}  canonical rows={res.row_count_ingested}  skipped={res.skipped_rows}")
    print(f"variables: {res.canonical_variables_found}")
    print(f"region resolution rate: {res.region_resolution_rate:.0%}   profile match: {res.profile_match}")
    for n in res.notes:
        print("  -", n)
    if res.status == "pending_confirmation":
        print("\nPROPOSALS NEEDING CONFIRMATION:")
        for p in res.mapping_proposals:
            if p["decision"] == "needs_confirmation":
                print(f"  {p['source_column']:22} -> {p['suggested_variable']} "
                      f"(vt={p['suggested_value_type']}, conf={p['confidence']:.2f})  "
                      f"samples={p['sample_values'][:3]}")
    else:
        print("\ncanonical store summary:")
        for k, v in parquet_store.dataset_summary().items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    sys.exit(_main())
