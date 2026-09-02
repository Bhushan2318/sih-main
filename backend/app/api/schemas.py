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


class AllRegionsResponse(Schema):
    """Every lead day's map payload in one response. The cycle is scored once regardless,
    so this costs no more to compute than a single day - it just saves the dashboard a
    round-trip (and a loading flash) every time the lead-day selector moves."""

    model_trained: bool
    last_trained_at: Optional[datetime] = None
    current_run_id: Optional[str] = None
    init_date: Optional[date] = None
    risk_band_definitions: dict = Field(default_factory=dict)
    available_lead_days: list = Field(default_factory=list)
    days: list = Field(default_factory=list)   # one RegionsResponse per lead day 1..10
    message: Optional[str] = None


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
    # What the MODEL learned from, which is not the same thing as what this box holds.
    # Training happens on a 16 GB runner and can read years of reforecast archive; the
    # serving box gets only what it needs to answer a request. Reporting the store as if
    # it were the training base would understate the evidence behind the model - so both
    # are reported, from their own sources, and neither stands in for the other.
    training_data: dict = Field(default_factory=dict)
    # "compared to what?" - the baseline ladder this run was scored against, on the same
    # rows. Empty when the run published none, never filled in with numbers from another.
    baselines: dict = Field(default_factory=dict)
    modelled_variables: list = Field(default_factory=list)
    skipped_variables: dict = Field(default_factory=dict)
    validation_metrics: dict = Field(default_factory=dict)
    thresholds: dict = Field(default_factory=dict)
    explanation_method: Optional[str] = None
    websocket_clients: int = 0
    message: Optional[str] = None


# --------------------------------------------------------------------------- replay
# Guided replay: step through one real historical forecast cycle lead day by lead day
# and show how a bust developed. Every field is a scored number from that cycle or a
# sentence generated from those numbers - nothing here is scripted or synthetic.


class ReplayCycleSummary(Schema):
    init_date: date
    lead_days: list = Field(default_factory=list)
    n_regions: int = 0
    peak_bust_probability: Optional[float] = None
    peak_lead_day: Optional[int] = None
    peak_region_id: Optional[str] = None
    peak_region_name: Optional[str] = None
    n_high_regions_peak: int = 0
    verified: bool = False                 # at least one lead day has an observed value
    verified_lead_days: int = 0
    peak_region_abs_error: Optional[float] = None   # realised |forecast - observed| on the dominant var
    medium_range_growth: float = 0.0       # mean P(bust) days 4-10 minus days 1-3


class ReplayRegionStep(Schema):
    region_id: str
    region_name: Optional[str] = None
    bust_probability: float
    risk_band: str
    confidence: Optional[float] = None
    dominant_variable: Optional[str] = None


class ReplayLeadStep(Schema):
    lead_time_days: int
    valid_date: Optional[date] = None
    regions: list = Field(default_factory=list)     # ReplayRegionStep, desc by bust probability
    n_high: int = 0
    n_medium: int = 0
    mean_bust_probability: Optional[float] = None
    narration: str = ""                             # generated from this step's real numbers


class ReplayFocusPoint(Schema):
    lead_time_days: int
    valid_date: Optional[date] = None
    predicted_value: Optional[float] = None
    observed_value: Optional[float] = None
    observed_status: Optional[str] = None
    ensemble_spread: Optional[float] = None


class ReplayFocusSeries(Schema):
    region_id: str
    region_name: Optional[str] = None
    variable: str
    unit: Optional[str] = None
    bust_threshold: Optional[float] = None
    points: list = Field(default_factory=list)


class ReplayResponse(Schema):
    model_trained: bool
    current_run_id: Optional[str] = None
    init_date: Optional[date] = None
    available_cycles: list = Field(default_factory=list)    # ReplayCycleSummary, most demo-worthy first
    steps: list = Field(default_factory=list)               # ReplayLeadStep, ascending lead day
    focus: Optional[ReplayFocusSeries] = None               # charted region (peak region by default)
    focus_options: list = Field(default_factory=list)       # ReplayFocusSeries for each worst-list region, so the UI can swap the chart
    risk_band_definitions: dict = Field(default_factory=dict)
    summary_narration: Optional[str] = None
    message: Optional[str] = None


# --------------------------------------------------------------------------- ensemble
# The hero divergence view: the moment a cycle's ensemble and reality come apart, drawn
# from the five real GEFS members rather than a summary statistic.

class EnsemblePoint(Schema):
    lead_time_days: int
    valid_date: Optional[date] = None
    value: Optional[float] = None
    observed_status: Optional[str] = None      # final | provisional | None (not verified)


class EnsembleMemberTrace(Schema):
    member_id: str
    is_control: bool = False
    points: list = Field(default_factory=list)   # EnsemblePoint


class NationalRiskPoint(Schema):
    """Risk across every scored region for one lead day."""
    lead_time_days: int
    valid_date: Optional[date] = None
    mean_bust_probability: float
    min_bust_probability: float
    max_bust_probability: float
    n_regions: int
    n_high_regions: int


class CalibrationBin(Schema):
    bin_lo: float
    bin_hi: float
    predicted_mean: float
    observed_rate: float
    n: int


class ModelSkill(Schema):
    """How accurate the guesses actually were, on data the model never trained on."""
    split: Optional[str] = None
    n: int = 0
    roc_auc: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    brier: Optional[float] = None
    bust_rate: Optional[float] = None
    calibration: list = Field(default_factory=list)   # CalibrationBin
    note: Optional[str] = None


class EnsembleDivergenceResponse(Schema):
    model_trained: bool
    current_run_id: Optional[str] = None
    init_date: Optional[date] = None

    region_id: Optional[str] = None
    region_name: Optional[str] = None
    variable: Optional[str] = None
    unit: Optional[str] = None

    members: list = Field(default_factory=list)   # EnsembleMemberTrace, one per GEFS member
    ensemble_mean: list = Field(default_factory=list)   # EnsemblePoint
    observed: list = Field(default_factory=list)       # EnsemblePoint - only verified leads

    # First lead day this region's P(bust) reaches the top band: where the cycle turns.
    crossover_lead: Optional[int] = None
    peak_bust_probability: Optional[float] = None

    mean_bust_probability: Optional[float] = None
    prior_mean_bust_probability: Optional[float] = None
    prior_init_date: Optional[date] = None
    # Why no prior-cycle comparison is shown, when there isn't one.
    prior_note: Optional[str] = None

    n_high_regions: int = 0
    n_scored_regions: int = 0
    high_by_lead: Optional[int] = None

    # How much the member spread grows across the horizon for the charted series, and the
    # plain-language reason this subject was chosen - so the picture is never arbitrary.
    spread_growth: Optional[float] = None
    subject_reason: Optional[str] = None

    # Provenance, stated on screen rather than assumed.
    source: Optional[str] = None
    member_count: int = 0

    # National view, shown when no single region is pinned: risk across every scored
    # region for each lead day, plus how accurate the model's past guesses actually were.
    national: list = Field(default_factory=list)      # NationalRiskPoint
    skill: Optional[ModelSkill] = None
    national_note: Optional[str] = None

    eyebrow: Optional[str] = None
    headline_note: Optional[str] = None
    message: Optional[str] = None
