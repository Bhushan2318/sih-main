from __future__ import annotations

from typing import Optional

import pandas as pd

from app.api import schemas
from app.ingestion.canonical_schema import CanonicalVariable
from app.ingestion.canonical_schema import CIRCULAR_VARIABLES
from app.ml import inference
from app.services import replay_cases
from app.services.region_service import (
    NOT_TRAINED_MSG,
    _f,
    _region_name,
    _risk_band_definitions,
    _unit,
)

_MAX_CYCLES = 10

_cycles_memo: "tuple[str, list[schemas.ReplayCycleSummary]] | None" = None
_case_memo: "tuple[tuple, inference.ScoredCycle] | None" = None


def _cycle_summary(state, init) -> Optional[schemas.ReplayCycleSummary]:
    try:
        sc = inference.score_cycle(state, init)
    except inference.CycleNotPrecomputed:
        # On the serving box a cycle CI skipped is left out of the list rather than offered
        # and then refused.
        return None
    return summary_from_scored(state, sc, kind="forecast")


def summary_from_scored(state, sc, **labels) -> Optional[schemas.ReplayCycleSummary]:
    """The list entry for one scored cycle. `labels` are the fields that say what kind of
    cycle it is (kind, title, ...); everything else is read off the scores."""
    if sc is None or sc.events.empty:
        return None
    ev = sc.events
    leads = sorted(int(d) for d in ev["lead_time_days"].dropna().unique())
    peak = ev.loc[ev["bust_probability"].idxmax()]
    peak_lead = int(peak["lead_time_days"])
    at_peak = ev[ev["lead_time_days"] == peak_lead]

    verified_leads = 0
    peak_abs_err = None
    peak_rel_err = None
    peak_dom_var = None
    growth = 0.0
    pv = sc.per_variable
    if not pv.empty and "observed_value" in pv.columns:
        vmask = pv["observed_value"].notna() & pv["predicted_value"].notna()
        verified_leads = int(pv.loc[vmask, "lead_time_days"].nunique())
        dom = peak.get("dominant_variable")
        prz = pv[
            vmask
            & (pv["region_id"].astype(str) == str(peak["region_id"]))
            & (pv["variable"] == dom)
        ]
        # wind_direction_deg is circular (0 and 360 are the same bearing); a naive abs
        # difference can read as a ~345 degree "miss" for an actual 15 degree one, so it's
        # excluded here the same way ensemble_service and _focus_variable_for exclude it.
        if not prz.empty and dom not in CIRCULAR_VARIABLES:
            peak_abs_err = float((prz["predicted_value"] - prz["observed_value"]).abs().mean())
            peak_dom_var = str(dom)
            thr = state.thresholds.bust_threshold.get(peak_dom_var)
            # Divided by that variable's own bust threshold so a rainfall miss (mm) and a
            # pressure miss (hPa) land on the same scale: 1.0 means "missed by exactly one
            # bust's worth". Without this, order_cycles has confidence and coverage to sort
            # on but nothing for how wrong a verified cycle actually turned out to be.
            if thr:
                peak_rel_err = peak_abs_err / thr

    near = ev.loc[ev["lead_time_days"] <= 3, "bust_probability"].mean()
    far = ev.loc[ev["lead_time_days"] >= 4, "bust_probability"].mean()
    if pd.notna(near) and pd.notna(far):
        growth = float(far - near)

    return schemas.ReplayCycleSummary(
        init_date=pd.Timestamp(sc.init_date).date(),
        lead_days=leads,
        n_regions=int(ev["region_id"].nunique()),
        peak_bust_probability=_f(peak["bust_probability"]),
        peak_lead_day=peak_lead,
        peak_region_id=str(peak["region_id"]),
        peak_region_name=_region_name(str(peak["region_id"])),
        n_high_regions_peak=int((at_peak["risk_band"] == "high").sum()),
        verified=verified_leads > 0,
        verified_lead_days=verified_leads,
        peak_region_abs_error=_f(peak_abs_err),
        peak_region_relative_error=_f(peak_rel_err),
        peak_region_variable=peak_dom_var,
        peak_region_unit=_unit(peak_dom_var),
        medium_range_growth=round(growth, 4),
        **labels,
    )


def _event_cycles(state) -> list[schemas.ReplayCycleSummary]:
    """The catalogued past events this run has artifacts for, in catalogue order. Read
    from each case's summary.json; nothing is scored and no Parquet is opened. A run
    without cases (a CI-trained model) simply lists none."""
    out = []
    for case in replay_cases.CASES:
        s = replay_cases.read_case_summary(state.run_id, case)
        if s is None:
            continue
        s = {k: v for k, v in s.items() if k in schemas.ReplayCycleSummary.model_fields}
        s.update(kind="event", title=case.title, focus_region_id=case.focus_region_id,
                 focus_variable=case.focus_variable)
        try:
            out.append(schemas.ReplayCycleSummary(**s))
        except ValueError:  # a malformed summary is a missing case, never a broken list
            continue
    return out


def list_cycles() -> list[schemas.ReplayCycleSummary]:
    global _cycles_memo
    state = inference.load_model_state()
    if state is None:
        return []
    if _cycles_memo is not None and _cycles_memo[0] == state.run_id:
        return _cycles_memo[1]

    events = _event_cycles(state)
    event_dates = {e.init_date for e in events}
    out: list[schemas.ReplayCycleSummary] = []
    for init in inference.available_cycles()[:_MAX_CYCLES]:
        if pd.Timestamp(init).date() in event_dates:
            continue
        s = _cycle_summary(state, init)
        if s is not None:
            out.append(s)
    # Events first, in catalogue order: they are the cycles whose outcome is known, and
    # where Replay opens. The recent forecasts keep their own ordering below them.
    out = events + order_cycles(out)
    _cycles_memo = (state.run_id, out)
    return out


# Cycles within this fraction of the best coverage are treated as equally complete, so a
# cycle missing a handful of districts is not demoted below one with six more.
_COVERAGE_TIER = 0.1


def order_cycles(cycles: list) -> list:
    """Best cycle first, coverage before everything else.

    The store is cumulative, so after the live feed moved from 36 city points to all 666
    districts it holds both kinds at once. Sorting on `verified` first put the old sparse
    cycles on top - they are the ones whose outcome is known, because the full-coverage
    ones are recent - and Replay opened on a map of India with 36 districts drawn on it.
    Measured on the live site 2026-09-23: /api/replay defaulted to 2026-09-15, 36 districts
    per lead day.

    A cycle that cannot show the country cannot show a bust. "Outcome not yet known" is
    already explained in the UI; a near-empty map is not, and reads as broken rather than
    as young. As newer cycles age into verification this ordering converges on what the
    old one wanted anyway.

    Coverage is compared in tiers rather than exactly, so within a tier the previous
    ordering still decides. Nothing is filtered out: the sparse cycles are real runs, and
    right now they are the only ones whose outcome is known.
    """
    if not cycles:
        return []
    best = max((c.n_regions or 0) for c in cycles) or 1
    return sorted(
        cycles,
        key=lambda c: (
            round((c.n_regions or 0) / best / _COVERAGE_TIER),
            c.verified,
            # How badly a verified cycle actually missed, not just how confident the model
            # was. peak_region_relative_error is the peak miss as a multiple of that
            # variable's own bust threshold, so it is comparable across variables/units.
            # Unverified cycles carry None here and sort as 0.0 - below any real miss,
            # which is correct: there is nothing yet to call badly missed.
            round(c.peak_region_relative_error or 0.0, 3),
            round(max(c.medium_range_growth, 0.0), 3),
            round(c.peak_bust_probability or 0.0, 3),
            c.verified_lead_days,
        ),
        reverse=True,
    )


def get_replay(
    init_date: Optional[str] = None, focus_region: Optional[str] = None
) -> schemas.ReplayResponse:
    state = inference.load_model_state()
    if state is None:
        return schemas.ReplayResponse(model_trained=False, message=NOT_TRAINED_MSG)

    cycles = list_cycles()
    if not cycles:
        return schemas.ReplayResponse(
            model_trained=True, current_run_id=state.run_id, available_cycles=[],
            message="No forecast cycle in the store can be scored yet.",
        )

    target = init_date or str(cycles[0].init_date)
    case = replay_cases.case_for(target)
    sc = _read_event(state, case) if case is not None else None
    if sc is None:
        case = None
        sc = inference.score_cycle(state, target)
    if sc is None or sc.events.empty:
        return schemas.ReplayResponse(
            model_trained=True, current_run_id=state.run_id, available_cycles=cycles,
            message=f"Cycle {target} is not in the store or has nothing to score.",
        )

    cuts = state.thresholds.risk_band_cuts
    ev = sc.events.sort_values(["lead_time_days", "bust_probability"], ascending=[True, False])

    steps: list[schemas.ReplayLeadStep] = []
    prev: dict[str, float] = {}
    prev_dom = ""
    for lead, g in ev.groupby("lead_time_days", sort=True):
        cur = {str(r.region_id): float(r.bust_probability) for r in g.itertuples()}
        cur_dom = _pretty(g.iloc[0].get("dominant_variable"))
        regs = [
            schemas.ReplayRegionStep(
                region_id=str(r.region_id),
                region_name=_region_name(str(r.region_id)),
                bust_probability=_f(r.bust_probability) or 0.0,
                risk_band=r.risk_band,
                dominant_variable=getattr(r, "dominant_variable", None),
            )
            for r in g.itertuples()
        ]
        vd = g["valid_date"].iloc[0]
        steps.append(
            schemas.ReplayLeadStep(
                lead_time_days=int(lead),
                valid_date=pd.to_datetime(vd).date() if pd.notna(vd) else None,
                regions=regs,
                n_high=int((g["risk_band"] == "high").sum()),
                n_medium=int((g["risk_band"] == "medium").sum()),
                mean_bust_probability=_f(g["bust_probability"].mean()),
                narration=_narrate(int(lead), g, prev, cur, cuts["high"], prev_dom),
            )
        )
        prev = cur
        prev_dom = cur_dom

    if case is not None:
        # An event opens on the district it is about, charting the variable it is about.
        default_focus, focus_options = _build_focus(
            sc, state, focus_region or case.focus_region_id,
            prefer_variable={case.focus_region_id: case.focus_variable})
    else:
        default_focus, focus_options = _build_focus(sc, state, focus_region)
    return schemas.ReplayResponse(
        model_trained=True,
        current_run_id=state.run_id,
        init_date=pd.Timestamp(target).date(),
        available_cycles=cycles,
        steps=steps,
        focus=default_focus,
        focus_options=focus_options,
        risk_band_definitions=_risk_band_definitions(state),
        summary_narration=_summarise(sc, steps),
    )


# Canonical names end in their unit (rainfall_mm, atmospheric_moisture_kgm2). The
# narration reads as a sentence, so the unit token is dropped rather than spoken as
# "kgm2"; the suffixes come from the enum itself, not a second list to keep in step.
_UNIT_SUFFIXES = {v.value.rpartition("_")[2] for v in CanonicalVariable}


def _pretty(var) -> str:
    if not (isinstance(var, str) and var):
        return ""
    head, _, tail = var.rpartition("_")
    return (head if head and tail in _UNIT_SUFFIXES else var).replace("_", " ")


def _narrate(lead: int, g: pd.DataFrame, prev: dict, cur: dict, hi: float, prev_dom: str) -> str:
    top = g.iloc[0]
    tname = _region_name(str(top["region_id"])) or str(top["region_id"])
    bits: list[str] = []

    if prev:
        deltas = {rid: cur[rid] - prev[rid] for rid in cur if rid in prev}
        if deltas:
            rid_up, dlt = max(deltas.items(), key=lambda kv: kv[1])
            if dlt >= 0.05:
                nm = _region_name(rid_up) or rid_up
                bits.append(f"bust risk over {nm} climbs {prev[rid_up]:.2f}→{cur[rid_up]:.2f}")
    if not bits:
        verb = "opens with" if not prev else "still carries"
        bits.append(f"{tname} {verb} the highest bust risk at {float(top['bust_probability']):.2f}")

    dom = _pretty(top.get("dominant_variable"))
    if dom and dom != prev_dom:
        bits.append(f"now driven by {dom} error" if prev_dom else f"driven by {dom} error")

    n_high = int((g["bust_probability"] >= hi).sum())
    if n_high:
        verb = "" if not prev else "now "
        bits.append(f"{n_high} region{'s' if n_high != 1 else ''} {verb}high-risk".replace("  ", " "))

    return f"Day {lead} — " + "; ".join(bits) + "."


def _summarise(sc: inference.ScoredCycle, steps: list[schemas.ReplayLeadStep]) -> str:
    ev = sc.events
    peak = ev.loc[ev["bust_probability"].idxmax()]
    pr = _region_name(str(peak["region_id"])) or str(peak["region_id"])
    pl = int(peak["lead_time_days"])
    first = steps[0].mean_bust_probability or 0.0
    last = steps[-1].mean_bust_probability or 0.0
    trend = "climbs" if last > first + 0.02 else ("eases" if last < first - 0.02 else "holds")
    dom = _pretty(peak.get("dominant_variable"))
    dtxt = f", led by {dom} error." if dom else "."

    verified = (
        not sc.per_variable.empty
        and "observed_value" in sc.per_variable.columns
        and sc.per_variable["observed_value"].notna().any()
    )
    tail = (
        " Its forecast-vs-observed track is charted below."
        if verified
        else " This cycle has not verified yet, so no observed track is drawn."
    )
    return (
        f"Init {pd.Timestamp(sc.init_date).date()}: mean bust risk {trend} from "
        f"{first:.2f} on Day {steps[0].lead_time_days} to {last:.2f} on Day {steps[-1].lead_time_days}. "
        f"It peaks at {float(peak['bust_probability']):.2f} over {pr} on Day {pl}{dtxt}{tail}"
    )


def _focus_variable_for(v_region: pd.DataFrame, dominant: Optional[str], state) -> Optional[str]:
    if v_region.empty:
        return None
    if dominant and dominant not in CIRCULAR_VARIABLES and (v_region["variable"] == dominant).any():
        return dominant
    v = v_region.copy()
    v["abs_err"] = (v["predicted_value"] - v["observed_value"]).abs()
    thr = v["variable"].map(lambda x: state.thresholds.bust_threshold.get(x) or float("nan"))
    ranked = v.assign(_r=v["abs_err"] / thr).groupby("variable", observed=True)["_r"].mean()
    if ranked.dropna().empty:
        ranked = v.groupby("variable", observed=True)["abs_err"].mean()
    return None if ranked.empty else str(ranked.idxmax())


def _verified_by_region(sc: inference.ScoredCycle) -> "dict[str, pd.DataFrame]":
    """The chartable rows of every district, filtered once and split once.

    This used to be a filter over the whole per-variable table inside a loop over all 666
    districts, converting every region id to a string each time. Measured 2026-09-26: 4.98
    of get_replay's 5.25 s on the live cycle - which has nothing verified, so all 666
    passes found nothing - and 6.2 of 6.4 s on a past event. tests/test_replay_focus_speed.py.
    """
    pv = sc.per_variable
    if pv.empty or "observed_value" not in pv.columns:
        return {}
    v = pv[pv["observed_value"].notna() & pv["predicted_value"].notna()
           & ~pv["variable"].isin(CIRCULAR_VARIABLES)]
    if v.empty:
        return {}
    return {str(rid): g for rid, g in v.groupby(v["region_id"].astype(str), sort=False)}


def _focus_for_region(
    v: pd.DataFrame, state, region_id: str, dominant: Optional[str]
) -> Optional[schemas.ReplayFocusSeries]:
    """`v` is this district's chartable rows, from _verified_by_region."""
    var = _focus_variable_for(v, dominant, state)
    if var is None:
        return None
    sub = v[v["variable"] == var].sort_values("lead_time_days")
    points = [
        schemas.ReplayFocusPoint(
            lead_time_days=int(r.lead_time_days),
            valid_date=pd.to_datetime(r.valid_date).date() if pd.notna(r.valid_date) else None,
            predicted_value=_f(r.predicted_value),
            observed_value=_f(r.observed_value),
            observed_status=(getattr(r, "verification_status", None) or "final"),
            ensemble_spread=_f(getattr(r, "ensemble_spread", None)),
        )
        for r in sub.itertuples()
    ]
    return schemas.ReplayFocusSeries(
        region_id=region_id,
        region_name=_region_name(region_id),
        variable=var,
        unit=_unit(var),
        bust_threshold=_f(state.thresholds.bust_threshold.get(var)),
        points=points,
    )


def _build_focus(
    sc: inference.ScoredCycle, state, want_region: Optional[str],
    prefer_variable: Optional[dict] = None,
) -> "tuple[Optional[schemas.ReplayFocusSeries], list[schemas.ReplayFocusSeries]]":
    """`prefer_variable` maps a region to the variable to chart for it, in place of the
    one that drove its bust risk; it still falls back if that variable has no verified
    points there."""
    ev = sc.events
    if ev.empty:
        return None, []
    rid_col = ev["region_id"].astype(str)
    peak_rid = str(ev.loc[ev["bust_probability"].idxmax(), "region_id"])

    worst_row = ev.loc[ev.groupby(rid_col)["bust_probability"].idxmax()]
    dom_by_region = {
        str(r.region_id): getattr(r, "dominant_variable", None) for r in worst_row.itertuples()
    }
    rest = (
        worst_row.assign(_rid=worst_row["region_id"].astype(str))
        .sort_values("bust_probability", ascending=False)["_rid"]
        .tolist()
    )
    order = [peak_rid] + [r for r in rest if r != peak_rid]

    verified = _verified_by_region(sc)
    if not verified:
        return None, []
    options: list[schemas.ReplayFocusSeries] = []
    prefer_variable = prefer_variable or {}
    for rid in order:
        v = verified.get(rid)
        if v is None:
            continue
        fs = _focus_for_region(v, state, rid,
                               prefer_variable.get(rid) or dom_by_region.get(rid))
        if fs is not None:
            options.append(fs)
    if not options:
        return None, []
    default = next((o for o in options if o.region_id == want_region), options[0])
    return default, options


def _read_event(state, case) -> Optional[inference.ScoredCycle]:
    """A past event's trimmed artifact, holding at most one in memory at a time."""
    global _case_memo
    key = (state.run_id, case.init_date)
    if _case_memo is not None and _case_memo[0] == key:
        return _case_memo[1]
    sc = replay_cases.read_case(state.run_id, case)
    _case_memo = (key, sc) if sc is not None else None
    return sc


def invalidate() -> None:
    global _cycles_memo, _case_memo
    _cycles_memo = None
    _case_memo = None
