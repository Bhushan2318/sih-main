"""Pydantic response models for every endpoint.

Design rule from the plan: when there is no trained model or no data, these shapes carry
an explicit `model_trained: false` / `available: false` and EMPTY collections. They never
carry a placeholder or example number.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Schema(BaseModel):
    """Base for every response model.

    `protected_namespaces=()` because several fields legitimately start with `model_`
    (`model_trained`, `model_mae`, ...) - they describe the ML model, and pydantic would
    otherwise warn about its own reserved `model_` prefix on every import.
    """

    model_config = ConfigDict(protected_namespaces=())


# --------------------------------------------------------------------------- upload

class MappingProposal(Schema):
    source_column: str
    normalized: str
    role: str                       # measurement | dimension | value_type | variable_name | value | unmapped
    sample_values: list = Field(default_factory=list)
    suggested_variable: Optional[str] = None
    suggested_value_type: Optional[str] = None
    confidence: float = 0.0
    ambiguity_gap: float = 0.0
    method: Optional[str] = None
    unit_conversion: Optional[str] = None
    decision: str = "unmapped"
    alternatives: list = Field(default_factory=list)


class UploadResponse(Schema):
    batch_id: str
    status: str                     # pending_confirmation | training_started | failed
    detected_format: Optional[str] = None
    layout: Optional[str] = None
    row_count_raw: int = 0
    row_count_ingested: int = 0
    skipped_rows: int = 0
    canonical_variables_found: list = Field(default_factory=list)
    region_resolution_rate: float = 0.0
    source_profile_match: str = "none"
    mapping_proposals: list = Field(default_factory=list)
    notes: list = Field(default_factory=list)


class ConfirmMappingItem(Schema):
    source_column: str
    variable: Optional[str] = None
    value_type: Optional[str] = None
    role: Optional[str] = None
    unit_conversion: Optional[str] = None


class ConfirmMappingRequest(Schema):
    mappings: list


# --------------------------------------------------------------------------- regions

class RegionSummary(Schema):
    region_id: str
    region_name: Optional[str] = None
    bust_probability: Optional[float] = None
    risk_band: Optional[str] = None
    confidence: Optional[float] = None
    dominant_variable: Optional[str] = None
    data_available: bool = True


class RegionsResponse(Schema):
    lead_time_days: int
    model_trained: bool
    last_trained_at: Optional[datetime] = None
    current_run_id: Optional[str] = None
    init_date: Optional[date] = None
    valid_date: Optional[date] = None
    risk_band_definitions: dict = Field(default_factory=dict)
    regions: list = Field(default_factory=list)
    message: Optional[str] = None
    # Which lead days the current cycle actually covers. A 00Z run covers 1-10, but a
    # 06/12/18Z run cannot produce a whole-calendar-day forecast for its own init day, so
    # it starts at day 2. The dashboard reads this instead of assuming day 1 exists.
    available_lead_days: list = Field(default_factory=list)


class VariablePoint(Schema):
    lead_time_days: int
    valid_date: Optional[date] = None
    predicted_value: Optional[float] = None
    observed_value: Optional[float] = None
    # "final" (ERA5, what the models were trained against), "provisional" (near-real-time,
    # subject to revision) or None when nothing has verified this lead yet.
    observed_status: Optional[str] = None
    predicted_error: Optional[float] = None
    confidence: Optional[float] = None
    ensemble_spread: Optional[float] = None
    ensemble_member_count: Optional[int] = None


class VariableSeries(Schema):
    variable: str
    available: bool = True
    unit: Optional[str] = None
    bust_threshold: Optional[float] = None
    model_mae: Optional[float] = None
    model_rmse: Optional[float] = None
    model_r2: Optional[float] = None
    metrics_split: Optional[str] = None
    points: list = Field(default_factory=list)


class BustProbabilityPoint(Schema):
    lead_time_days: int
    valid_date: Optional[date] = None
    bust_probability: float
    risk_band: str
    dominant_variable: Optional[str] = None


class TopFactor(Schema):
    feature: str
    importance: float
    method: str


class RegionDetailResponse(Schema):
    region_id: str
    region_name: Optional[str] = None
    model_trained: bool
    current_run_id: Optional[str] = None
    init_date: Optional[date] = None
    variables: list = Field(default_factory=list)
    bust_probability_curve: list = Field(default_factory=list)
    top_factors: list = Field(default_factory=list)
    top_factors_method: Optional[str] = None
    analog_cases: list = Field(default_factory=list)
    message: Optional[str] = None


# --------------------------------------------------------------------------- alerts

class Alert(Schema):
    alert_id: str
    region_id: str
    region_name: Optional[str] = None
    lead_time_days: int
    valid_date: Optional[date] = None
    bust_probability: float
    risk_band: str
    dominant_variable: Optional[str] = None
    created_at: datetime
    training_run_id: Optional[str] = None


class AlertsResponse(Schema):
    generated_at: datetime
    model_trained: bool
    risk_band_definitions: dict = Field(default_factory=dict)
    alerts: list = Field(default_factory=list)
    message: Optional[str] = None


# --------------------------------------------------------------------- model status

class ModelStatusResponse(Schema):
    model_trained: bool
    current_run_id: Optional[str] = None
    last_trained_at: Optional[datetime] = None
    training_in_progress: bool = False
    last_training_error: Optional[str] = None
    data_volume: dict = Field(default_factory=dict)
    modelled_variables: list = Field(default_factory=list)
    skipped_variables: dict = Field(default_factory=dict)
    validation_metrics: dict = Field(default_factory=dict)
    thresholds: dict = Field(default_factory=dict)
    explanation_method: Optional[str] = None
    websocket_clients: int = 0
    message: Optional[str] = None
