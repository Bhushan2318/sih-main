"""Canonical internal data schema for Sanket.

This module defines the target shape every upload gets mapped to, regardless of its
original column names or format. It intentionally contains no ingestion/mapping logic
(see schema_mapper.py, built in Phase 2) — only the vocabulary and row shape that logic
maps onto.

See the approved plan (canonical schema section) for the reasoning behind each choice,
most notably: forecast and observed values are kept as separate long-format rows rather
than pre-paired at ingestion, and atmospheric moisture / soil moisture are two distinct
canonical variables rather than one "moisture" field.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CanonicalVariable(str, Enum):
    """The meteorological variables the pipeline understands. Not every upload will
    contain all of these — the ML pipeline (Phase 3) discovers which are present rather
    than assuming this full list."""

    RAINFALL_MM = "rainfall_mm"
    TEMPERATURE_C = "temperature_c"
    HUMIDITY_PCT = "humidity_pct"
    PRESSURE_HPA = "pressure_hpa"
    ATMOSPHERIC_MOISTURE_KGM2 = "atmospheric_moisture_kgm2"
    SOIL_MOISTURE_PCT = "soil_moisture_pct"
    WIND_SPEED_MS = "wind_speed_ms"
    WIND_DIRECTION_DEG = "wind_direction_deg"


class ValueType(str, Enum):
    """Whether a canonical row is a model forecast or a ground-truth observation."""

    FORECAST = "forecast"
    OBSERVED = "observed"


# Plausible physical ranges per canonical variable, used by the schema-mapper's
# value-range heuristic (Phase 2) as one signal among several — never the sole
# disambiguator, since several variables' ranges overlap (e.g. humidity_pct and
# soil_moisture_pct are both roughly 0-100).
VARIABLE_UNITS: dict[CanonicalVariable, str] = {
    CanonicalVariable.RAINFALL_MM: "mm",
    CanonicalVariable.TEMPERATURE_C: "°C",
    CanonicalVariable.HUMIDITY_PCT: "% RH",
    CanonicalVariable.PRESSURE_HPA: "hPa",
    CanonicalVariable.ATMOSPHERIC_MOISTURE_KGM2: "kg/m² (TCWV)",
    CanonicalVariable.SOIL_MOISTURE_PCT: "% volumetric",
    CanonicalVariable.WIND_SPEED_MS: "m/s",
    CanonicalVariable.WIND_DIRECTION_DEG: "degrees (meteorological, from-direction)",
}

VARIABLE_PLAUSIBLE_RANGE: dict[CanonicalVariable, tuple[float, float]] = {
    CanonicalVariable.RAINFALL_MM: (0.0, 2000.0),
    CanonicalVariable.TEMPERATURE_C: (-50.0, 60.0),
    CanonicalVariable.HUMIDITY_PCT: (0.0, 100.0),
    CanonicalVariable.PRESSURE_HPA: (800.0, 1100.0),
    CanonicalVariable.ATMOSPHERIC_MOISTURE_KGM2: (0.0, 90.0),
    CanonicalVariable.SOIL_MOISTURE_PCT: (0.0, 100.0),
    CanonicalVariable.WIND_SPEED_MS: (0.0, 100.0),
    CanonicalVariable.WIND_DIRECTION_DEG: (0.0, 360.0),
}


class CanonicalRow(BaseModel):
    """One row of the canonical long-format fact table (mirrors the parquet columns
    the accumulated dataset is stored as). One row = one (variable, value_type) reading
    for one region at one valid_date, optionally tied to a specific forecast issuance
    (init_date/lead_time_days) and ensemble member.
    """

    record_id: str
    upload_batch_id: str
    source_file: str
    source_column: str

    variable: CanonicalVariable
    value_type: ValueType
    value: float

    region_id: Optional[str] = None  # ISO 3166-2:IN, e.g. "IN-MH"
    region_name: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

    init_date: Optional[date] = None
    valid_date: date
    lead_time_days: Optional[int] = Field(default=None, ge=1, le=10)

    ensemble_member_id: Optional[str] = None

    mapping_confidence: float = Field(ge=0.0, le=1.0)
    ingested_at: datetime

    # Provenance / quality flags carried alongside every row so downstream code and the
    # UI can be honest about how a value was placed, without re-deriving it.
    grain: str = "native"  # "native" (daily/sub-daily) or "monthly" (coarse aggregate)
    region_resolution_method: Optional[str] = None  # name | point_in_polygon | nearest_polygon | unresolved

    # How settled an observed value is (Phase 6 two-tier verification).
    #   None          - not applicable (forecast rows) or ingested before Phase 6, in
    #                   which case it came from the ERA5 archive and is already final.
    #   "provisional" - near-real-time analysis, available within hours but subject to
    #                   revision. Shown badged in the UI and EXCLUDED from training.
    #   "final"       - ERA5/ERA5T reanalysis; the baseline the models were trained on.
    # Downstream code must test `!= "provisional"` rather than `== "final"`, so legacy
    # rows carrying None are not silently dropped from training.
    verification_status: Optional[str] = None


# Column order for the canonical parquet partitions. parquet_store writes exactly these,
# in this order, so every batch's partition is schema-compatible for pyarrow.dataset.
CANONICAL_COLUMNS: tuple[str, ...] = tuple(CanonicalRow.model_fields.keys())
