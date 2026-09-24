from __future__ import annotations

from datetime import date, datetime, timedelta
from enum import Enum
from math import isfinite
from typing import Any, Mapping, Optional

from pydantic import BaseModel, Field, FiniteFloat, model_validator


class CanonicalVariable(str, Enum):

    RAINFALL_MM = "rainfall_mm"
    TEMPERATURE_C = "temperature_c"
    HUMIDITY_PCT = "humidity_pct"
    PRESSURE_HPA = "pressure_hpa"
    ATMOSPHERIC_MOISTURE_KGM2 = "atmospheric_moisture_kgm2"
    SOIL_MOISTURE_PCT = "soil_moisture_pct"
    WIND_SPEED_MS = "wind_speed_ms"
    WIND_DIRECTION_DEG = "wind_direction_deg"


class ValueType(str, Enum):

    FORECAST = "forecast"
    OBSERVED = "observed"


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

CIRCULAR_VARIABLES: frozenset[CanonicalVariable] = frozenset({CanonicalVariable.WIND_DIRECTION_DEG})
"""Variables where 0 and 360 are the same value, so a naive abs(predicted - observed)
overstates a small miss near the wraparound (350 vs 5 reads as 345, not the true 15).
Excluded from plain abs-error comparisons wherever they'd otherwise be computed."""

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
    """The validation contract for a row entering the canonical store.

    The mapper's confidence score is deliberately not a substitute for this model.  A
    stored profile or a hand-written confirmation can be edited independently of the
    mapper, and a pandas/numpy float can carry NaN or infinity through a conversion
    without raising.  Keeping the final checks here makes the parquet writer's input a
    genuine runtime boundary rather than a collection of dictionaries.
    """

    record_id: str
    upload_batch_id: str
    source_file: str
    source_column: str

    variable: CanonicalVariable
    value_type: ValueType
    value: FiniteFloat

    region_id: Optional[str] = None
    region_name: Optional[str] = None
    lat: Optional[FiniteFloat] = None
    lon: Optional[FiniteFloat] = None

    init_date: Optional[date] = None
    valid_date: date
    lead_time_days: Optional[int] = Field(default=None, ge=1, le=10)

    ensemble_member_id: Optional[str] = None

    mapping_confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    ingested_at: datetime

    grain: str = "native"
    region_resolution_method: Optional[str] = None

    verification_status: Optional[str] = None

    # A date is not a cycle: GEFS publishes 00/06/12/18Z.  These fields are optional so
    # Parquet files written before cycle identity was added remain readable without a
    # migration.  New forecast rows populate both; readers derive the 00Z fallback for
    # legacy rows at read time.
    init_cycle: Optional[datetime] = None
    cycle_hour: Optional[int] = Field(default=None, ge=0, le=23)

    @model_validator(mode="after")
    def _validate_canonical_contract(self) -> "CanonicalRow":
        variable = CanonicalVariable(self.variable)
        value_type = ValueType(self.value_type)
        lo, hi = VARIABLE_PLAUSIBLE_RANGE[variable]
        value = float(self.value)
        if not isfinite(value) or not lo <= value <= hi:
            raise ValueError(
                f"{variable.value} value {value!r} is outside the canonical range "
                f"[{lo}, {hi}] or is not finite"
            )

        for name, lower, upper in (("lat", -90.0, 90.0), ("lon", -180.0, 180.0)):
            coordinate = getattr(self, name)
            if coordinate is not None and not lower <= float(coordinate) <= upper:
                raise ValueError(f"{name} {coordinate!r} is outside [{lower}, {upper}]")

        if self.init_cycle is not None and self.init_date is not None:
            if self.init_cycle.date() != self.init_date:
                raise ValueError("init_cycle and init_date identify different dates")
            if self.cycle_hour is not None and self.init_cycle.hour != self.cycle_hour:
                raise ValueError("init_cycle and cycle_hour identify different hours")

        if value_type is ValueType.FORECAST:
            if self.init_date is None or self.lead_time_days is None:
                raise ValueError("forecast rows require init_date and lead_time_days")
            expected_valid = self.init_date + timedelta(days=self.lead_time_days - 1)
            if self.valid_date != expected_valid:
                raise ValueError(
                    "forecast dates contradict lead_time_days: expected "
                    f"{expected_valid.isoformat()}, got {self.valid_date.isoformat()}"
                )
        elif self.init_date is not None or self.lead_time_days is not None \
                or self.init_cycle is not None or self.cycle_hour is not None:
            raise ValueError("observed rows cannot carry forecast cycle fields")

        return self


def validate_canonical_row(row: Mapping[str, Any]) -> CanonicalRow:
    """Validate and return one row at the ingestion boundary.

    Keeping this tiny wrapper makes call sites independent of Pydantic's validation
    exception type while retaining the detailed error in ``ValidationError``.
    """
    return CanonicalRow.model_validate(dict(row))


CANONICAL_COLUMNS: tuple[str, ...] = tuple(CanonicalRow.model_fields.keys())
