"""ORM models for the metadata DB.

Six tables:
  * UploadBatch    - one row per uploaded file, its parse result and ingest status
  * ColumnMapping  - one row per source column per batch: how it was interpreted
  * SourceProfile  - a reusable confirmed mapping, keyed by a header fingerprint
  * TrainingRun    - one row per retrain (populated in Phase 3)
  * Alert          - one row per generated bust alert (populated in Phase 4)
  * IngestRun      - one row per automated live-ingestion attempt (Phase 6)

TrainingRun and Alert are defined now, with minimal columns, so later phases never have
to alter the schema - `init_db()` creates everything up front.

Annotations use ``typing.Optional`` (not ``X | None``): SQLAlchemy evaluates mapped
annotation strings, and the local Python is 3.9 where ``str | None`` is not runtime-valid.
"""

from datetime import datetime, timezone
from typing import Optional
import uuid

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class UploadBatch(Base):
    __tablename__ = "upload_batch"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    original_filename: Mapped[str] = mapped_column(String(512))
    stored_path: Mapped[str] = mapped_column(String(1024))
    detected_format: Mapped[Optional[str]] = mapped_column(String(32), default=None)
    content_sha256: Mapped[Optional[str]] = mapped_column(String(64), default=None, index=True)

    row_count_raw: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    # received | pending_confirmation | canonicalizing | ingested | failed
    status: Mapped[str] = mapped_column(String(32), default="received", index=True)
    layout: Mapped[Optional[str]] = mapped_column(String(8), default=None)  # wide | long
    grain: Mapped[str] = mapped_column(String(16), default="native")

    canonical_row_count: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    skipped_row_count: Mapped[int] = mapped_column(Integer, default=0)

    source_profile_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("source_profile.id"), default=None
    )
    error: Mapped[Optional[str]] = mapped_column(Text, default=None)
    notes_json: Mapped[Optional[list]] = mapped_column(JSON, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    column_mappings: Mapped[list["ColumnMapping"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan"
    )
    source_profile: Mapped[Optional["SourceProfile"]] = relationship(back_populates="batches")


class ColumnMapping(Base):
    __tablename__ = "column_mapping"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    upload_batch_id: Mapped[str] = mapped_column(
        ForeignKey("upload_batch.id", ondelete="CASCADE"), index=True
    )

    source_column: Mapped[str] = mapped_column(String(512))
    source_header_normalized: Mapped[str] = mapped_column(String(512))
    # measurement | dimension | value_type | unmapped
    role: Mapped[str] = mapped_column(String(16), default="unmapped")

    mapped_variable: Mapped[Optional[str]] = mapped_column(String(48), default=None)
    mapped_value_type: Mapped[Optional[str]] = mapped_column(String(16), default=None)
    # auto_accept | needs_confirmation | confirmed | unmapped
    decision: Mapped[str] = mapped_column(String(24), default="unmapped")
    method: Mapped[Optional[str]] = mapped_column(String(24), default=None)

    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    ambiguity_gap: Mapped[float] = mapped_column(Float, default=0.0)
    unit_conversion: Mapped[Optional[str]] = mapped_column(String(24), default=None)

    sample_values_json: Mapped[Optional[list]] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    batch: Mapped["UploadBatch"] = relationship(back_populates="column_mappings")


class SourceProfile(Base):
    __tablename__ = "source_profile"
    __table_args__ = (UniqueConstraint("fingerprint", name="uq_source_profile_fingerprint"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    header_list_json: Mapped[list] = mapped_column(JSON)
    # {source_column: {"variable": ..., "value_type": ..., "role": ..., "unit_conversion": ...}}
    confirmed_mapping_json: Mapped[dict] = mapped_column(JSON)

    file_format: Mapped[Optional[str]] = mapped_column(String(32), default=None)
    layout: Mapped[Optional[str]] = mapped_column(String(8), default=None)

    times_seen: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    batches: Mapped[list["UploadBatch"]] = relationship(back_populates="source_profile")


class TrainingRun(Base):
    __tablename__ = "training_run"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, default=_uuid, index=True)
    status: Mapped[str] = mapped_column(String(16), default="running")  # running|complete|failed
    triggered_by_batch_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("upload_batch.id"), default=None
    )
    validation_metrics_json: Mapped[Optional[dict]] = mapped_column(JSON, default=None)
    error: Mapped[Optional[str]] = mapped_column(Text, default=None)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)


class Alert(Base):
    __tablename__ = "alert"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    training_run_id: Mapped[Optional[str]] = mapped_column(String(64), default=None, index=True)
    region_id: Mapped[str] = mapped_column(String(8), index=True)
    region_name: Mapped[Optional[str]] = mapped_column(String(128), default=None)
    lead_time_days: Mapped[int] = mapped_column(Integer)
    bust_probability: Mapped[float] = mapped_column(Float)
    risk_band: Mapped[str] = mapped_column(String(16))
    dominant_variable: Mapped[Optional[str]] = mapped_column(String(48), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)


class IngestRun(Base):
    """One row per automated live-ingestion attempt (Phase 6).

    Kept separate from UploadBatch: a single run may produce several batches (a forecast
    cycle plus an observation refresh), skip entirely because the cycle was already
    present, or fail before any file exists. This is also what /api/ingest/status reads,
    so the dashboard can show real feed freshness instead of implying the data is current.
    """

    __tablename__ = "ingest_run"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(24), index=True)  # forecast|observations_provisional|observations_final
    # For forecast runs: the cycle, as "YYYY-MM-DD HH" UTC. For observation runs: the window end.
    target: Mapped[Optional[str]] = mapped_column(String(32), default=None, index=True)
    status: Mapped[str] = mapped_column(String(16), default="running")  # running|complete|skipped|failed
    trigger: Mapped[str] = mapped_column(String(16), default="schedule")  # schedule|manual|backfill

    upload_batch_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("upload_batch.id"), default=None
    )
    rows_ingested: Mapped[int] = mapped_column(Integer, default=0)
    detail: Mapped[Optional[str]] = mapped_column(Text, default=None)
    error: Mapped[Optional[str]] = mapped_column(Text, default=None)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
