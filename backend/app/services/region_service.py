"""Region-facing read logic: turn a ScoredCycle into the map and detail-panel payloads.

Every branch that cannot produce a real number returns an explicit empty/`model_trained:
false` shape with a human-readable `message`, never a placeholder.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from app.api import schemas
from app.ingestion.canonical_schema import VARIABLE_UNITS, CanonicalVariable
from app.ml import inference, registry
from app.ml.explain import top_factors_for
from app.utils import india_state_codes

NOT_TRAINED_MSG = (
    "No model has been trained yet. Upload a forecast dataset with matching observations "
    "to train one - nothing is shown until a real model exists."
)
NO_SCORE_MSG = (
    "A model exists but the canonical store has no forecast cycle that can be scored yet."
)


def _region_name(region_id: str) -> Optional[str]:
    rec = india_state_codes.resolve_by_region_id(region_id)
    return rec.region_name if rec else None


def _f(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else round(f, 6)


def _last_trained_at():
    rid = registry.current_run_id()
    if not rid:
        return None
    p = registry.run_dir(rid) / "manifest.json"
    if p.exists():
        import datetime as _dt
        return _dt.datetime.fromtimestamp(p.stat().st_mtime, tz=_dt.timezone.utc)
    return None


def _risk_band_definitions(state) -> dict:
    cuts = state.thresholds.risk_band_cuts
    return {
        "low": f"P(bust) < {cuts['medium']:.3f}",
        "medium": f"{cuts['medium']:.3f} <= P(bust) < {cuts['high']:.3f}",
        "high": f"P(bust) >= {cuts['high']:.3f}",
        "basis": "percentile bands of the classifier's predicted probabilities on the "
                 "validation split of the current training run",
    }


# --------------------------------------------------------------------------- map view

def get_regions(lead_time_days: int) -> schemas.RegionsResponse:
    state = inference.load_model_state()
    if state is None:
        return schemas.RegionsResponse(
            lead_time_days=lead_time_days, model_trained=False,
            regions=[], message=NOT_TRAINED_MSG,
        )

    scored = inference.score_latest_cycle(state)
    if scored is None or scored.events.empty:
        return schemas.RegionsResponse(
            lead_time_days=lead_time_days, model_trained=True,
            current_run_id=state.run_id, last_trained_at=_last_trained_at(),
            regions=[], message=NO_SCORE_MSG,
        )

    available = sorted(int(d) for d in scored.events["lead_time_days"].dropna().unique())

    ev = scored.events[scored.events["lead_time_days"] == lead_time_days]
    if ev.empty:
        # Say which days this cycle does cover rather than only what it lacks: a 06/12/18Z
        # run legitimately has no day-1, and the dashboard uses this to land on a lead day
        # that exists instead of an empty map.
        covered = (f" This cycle covers day {available[0]}-{available[-1]}."
                   if available else "")
        return schemas.RegionsResponse(
            lead_time_days=lead_time_days, model_trained=True,
            current_run_id=state.run_id, last_trained_at=_last_trained_at(),
            init_date=scored.init_date.date(),
            risk_band_definitions=_risk_band_definitions(state), regions=[],
            available_lead_days=available,
            message=f"The current forecast cycle has no day-{lead_time_days} data.{covered}",
        )

    conf_cols = [c for c in ev.columns if c.startswith("conf_")]
    regions = []
    for row in ev.itertuples():
        rid = str(row.region_id)
        conf = np.nanmean([getattr(row, c, np.nan) for c in conf_cols]) if conf_cols else np.nan
        regions.append(schemas.RegionSummary(
            region_id=rid,
            region_name=_region_name(rid),
            bust_probability=_f(row.bust_probability),
            risk_band=row.risk_band,
            confidence=_f(conf),
            dominant_variable=getattr(row, "dominant_variable", None),
            data_available=True,
        ))
    regions.sort(key=lambda r: (r.bust_probability is None, -(r.bust_probability or 0)))

    valid = ev["valid_date"].max()
    return schemas.RegionsResponse(
        lead_time_days=lead_time_days,
        model_trained=True,
        current_run_id=state.run_id,
        last_trained_at=_last_trained_at(),
        init_date=scored.init_date.date(),
        valid_date=pd.to_datetime(valid).date() if pd.notna(valid) else None,
        risk_band_definitions=_risk_band_definitions(state),
        regions=regions,
        available_lead_days=available,
    )


# ------------------------------------------------------------------------ detail view

def get_region_detail(region_id: str) -> schemas.RegionDetailResponse:
    state = inference.load_model_state()
    if state is None:
        return schemas.RegionDetailResponse(
            region_id=region_id, region_name=_region_name(region_id),
            model_trained=False, message=NOT_TRAINED_MSG,
        )

    scored = inference.score_latest_cycle(state)
    if scored is None:
        return schemas.RegionDetailResponse(
            region_id=region_id, region_name=_region_name(region_id),
            model_trained=True, current_run_id=state.run_id, message=NO_SCORE_MSG,
        )

    pv = scored.per_variable
    pv = pv[pv["region_id"].astype(str) == region_id]
    ev = scored.events[scored.events["region_id"].astype(str) == region_id]
    if pv.empty and ev.empty:
        return schemas.RegionDetailResponse(
            region_id=region_id, region_name=_region_name(region_id),
            model_trained=True, current_run_id=state.run_id,
            init_date=scored.init_date.date(),
            message=f"No forecast data for {region_id} in the current cycle.",
        )

    metrics = inference.model_validation_metrics(state).get("regressors", {})

    variables = []
    for var in state.variables:
        sub = pv[pv["variable"] == var].sort_values("lead_time_days")
        m = metrics.get(var, {})
        if sub.empty:
            variables.append(schemas.VariableSeries(
                variable=var, available=False,
                unit=_unit(var), bust_threshold=_f(state.thresholds.bust_threshold.get(var)),
            ))
            continue
        points = [
            schemas.VariablePoint(
                lead_time_days=int(r.lead_time_days),
                valid_date=pd.to_datetime(r.valid_date).date() if pd.notna(r.valid_date) else None,
                predicted_value=_f(r.predicted_value),
                observed_value=_f(r.observed_value),
                observed_status=(getattr(r, "verification_status", None)
                                 if pd.notna(getattr(r, "observed_value", None)) else None),
                predicted_error=_f(r.predicted_error),
                confidence=_f(r.confidence),
                ensemble_spread=_f(r.ensemble_spread),
                ensemble_member_count=(int(r.ensemble_member_count)
                                       if pd.notna(r.ensemble_member_count) else None),
            )
            for r in sub.itertuples()
        ]
        variables.append(schemas.VariableSeries(
            variable=var, available=True, unit=_unit(var),
            bust_threshold=_f(state.thresholds.bust_threshold.get(var)),
            model_mae=_f(m.get("mae")), model_rmse=_f(m.get("rmse")),
            model_r2=_f(m.get("r2")), metrics_split=m.get("split"),
            points=points,
        ))

    curve = [
        schemas.BustProbabilityPoint(
            lead_time_days=int(r.lead_time_days),
            valid_date=pd.to_datetime(r.valid_date).date() if pd.notna(r.valid_date) else None,
            bust_probability=_f(r.bust_probability) or 0.0,
            risk_band=r.risk_band,
            dominant_variable=getattr(r, "dominant_variable", None),
        )
        for r in ev.sort_values("lead_time_days").itertuples()
    ]

    factors, method = _top_factors(state, region_id, ev)
    return schemas.RegionDetailResponse(
        region_id=region_id,
        region_name=_region_name(region_id),
        model_trained=True,
        current_run_id=state.run_id,
        init_date=scored.init_date.date(),
        variables=variables,
        bust_probability_curve=curve,
        top_factors=factors,
        top_factors_method=method,
        analog_cases=[],   # similarity search not implemented; empty rather than faked
    )


def _unit(var: str) -> Optional[str]:
    try:
        return VARIABLE_UNITS[CanonicalVariable(var)]
    except (ValueError, KeyError):
        return None


def _top_factors(state, region_id: str, ev: pd.DataFrame):
    if state.shap_summary.empty or ev.empty:
        return [], None
    # explain the riskiest lead day for this region
    lead = int(ev.sort_values("bust_probability", ascending=False)["lead_time_days"].iloc[0])
    raw = top_factors_for(state.shap_summary, region_id, lead, model="classifier", k=6)
    factors = [schemas.TopFactor(**f) for f in raw]
    method = factors[0].method if factors else None
    return factors, method
