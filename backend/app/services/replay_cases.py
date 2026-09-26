"""Past events Replay offers alongside today's forecasts, with their outcome known.

Replay's recent cycles are mostly too young to have verified, so on their own they cannot
show the thing the product exists for: a forecast that went badly wrong, and whether the
model saw it coming. These are cycles from the reforecast archive where it did go wrong,
over a district and a day everyone remembers.

HOW THEY WERE CHOSEN, AND WHY THAT MATTERS
- By what happened, never by how the model did. The list was fixed before any of them was
  scored, and each is shown whatever the model said; a miss stays in.
- Out of sample only. `sample_note` reads the served run's own split from its manifest and
  refuses a case from a year the model trained on - a replay of training data is a recital,
  not evidence. Each case is labelled with which kind of unseen year it is.
- Only events ERA5 actually records. The truth Replay checks against is ERA5, and several
  famous Indian rain events barely appear in it. Those are left out and written up in
  docs/known-issues.md rather than shown against a truth that missed them.
- Scored as of 00 UTC on the init date (`inference.score_cycle(..., as_of_init=True)`).

Only the titles here are written by hand. Every number and every sentence Replay shows for
a case comes from the scored cycle, built by scripts/build_replay_cases.py into the run
directory it was scored with, so a case can never be served by a model that did not score
it.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from app.ml import registry
from app.ml.inference import ScoredCycle

DIR_NAME = "replay_cases"
_EVENTS = "events.parquet"
_PER_VARIABLE = "per_variable.parquet"
_SUMMARY = "summary.json"
_META = "meta.json"

# What Replay reads, and nothing more. A full scored cycle carries every pred_err_*,
# conf_*, spread_* and descriptor column the classifier needed; none of it is shown.
EVENT_COLUMNS = ("region_id", "init_date", "lead_time_days", "valid_date",
                 "bust_probability", "risk_band", "dominant_variable")
PER_VARIABLE_COLUMNS = ("region_id", "lead_time_days", "variable", "valid_date",
                        "predicted_value", "observed_value", "ensemble_spread",
                        "ensemble_member_count")


class CaseRefused(ValueError):
    """A case that cannot be shown honestly, so is not built at all."""


@dataclass(frozen=True)
class ReplayCase:
    id: str
    title: str
    init_date: dt.date
    # The day the event peaked, which the builder requires an observation for.
    peak_valid_date: dt.date
    focus_region_id: str
    # What Replay charts for the focus district. Chosen by what happened (these are rain
    # events), not by whichever variable the model happened to rank highest there.
    focus_variable: str


# Display order: the first is where Replay opens.
CASES: "tuple[ReplayCase, ...]" = (
    ReplayCase("kerala-2018", "Kerala floods · Aug 2018", dt.date(2018, 8, 13),
               dt.date(2018, 8, 15), "IN-KL-IDUKKI", "rainfall_mm"),
    ReplayCase("mumbai-2017", "Mumbai rain · Aug 2017", dt.date(2017, 8, 27),
               dt.date(2017, 8, 29), "IN-MH-MUMBAISUBURBAN", "rainfall_mm"),
    ReplayCase("ockhi-2017", "Cyclone Ockhi · Nov 2017", dt.date(2017, 11, 28),
               dt.date(2017, 11, 30), "IN-TN-KANNIYAKUMARI", "rainfall_mm"),
    ReplayCase("fani-2019", "Cyclone Fani · May 2019", dt.date(2019, 5, 1),
               dt.date(2019, 5, 3), "IN-OR-KHORDHA", "rainfall_mm"),
)


def case_for(init_date) -> Optional[ReplayCase]:
    """The catalogued case for this init date, if it is one."""
    if init_date is None:
        return None
    try:
        d = pd.Timestamp(init_date).date()
    except (ValueError, TypeError):
        return None
    return next((c for c in CASES if c.init_date == d), None)


def sample_note(year: int, manifest: dict) -> str:
    """How a case's year relates to the run that scored it, read from that run's manifest.

    Refuses a training year, and refuses when the manifest does not say - a case that
    cannot say whether the model saw its year cannot say whether it is evidence.
    """
    split = manifest.get("split_cycles") or {}
    train = manifest.get("pooled_train_years") or split.get("train_years") or []
    test = manifest.get("test_year", split.get("test_year"))
    if not train or test is None:
        raise CaseRefused("the run's manifest does not record its training years and test "
                          "year, so a case cannot be labelled in or out of sample")
    if year in set(int(y) for y in train):
        raise CaseRefused(f"{year} is one of this model's training years; a replay of it "
                          f"would show the model data it learned from")
    if year == int(test):
        return (f"Held-out test year: {year} was used only to score this model, "
                f"never to train it.")
    return f"Out of sample: {year} was never used to train or test this model."


def case_dir(run_id: str, init_date, base: Optional[Path] = None) -> Path:
    root = Path(base) if base is not None else registry.run_dir(run_id)
    return root / DIR_NAME / pd.Timestamp(init_date).strftime("%Y%m%d")


def write_case(run_id: str, case: ReplayCase, scored: ScoredCycle, summary: dict,
               base: Optional[Path] = None) -> Path:
    """Write one case, trimmed to what Replay reads. `meta.json` goes LAST, as in
    app/ml/precomputed.py: a build killed part-way leaves a directory the reader rejects,
    never one that reads as whole with districts missing."""
    missing = ([c for c in EVENT_COLUMNS if c not in scored.events.columns]
               + [c for c in PER_VARIABLE_COLUMNS if c not in scored.per_variable.columns])
    if missing:
        raise CaseRefused(f"scored cycle lacks {missing}; Replay could not show it")

    d = case_dir(run_id, case.init_date, base)
    d.mkdir(parents=True, exist_ok=True)
    (d / _META).unlink(missing_ok=True)
    scored.events[list(EVENT_COLUMNS)].to_parquet(
        d / _EVENTS, index=False, compression="zstd")
    scored.per_variable[list(PER_VARIABLE_COLUMNS)].to_parquet(
        d / _PER_VARIABLE, index=False, compression="zstd")
    (d / _SUMMARY).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (d / _META).write_text(json.dumps({
        "run_id": run_id,
        "case_id": case.id,
        "init_date": case.init_date.isoformat(),
        "n_events": int(len(scored.events)),
    }), encoding="utf-8")
    return d


def _meta(run_id: str, case: ReplayCase, base: Optional[Path]) -> "tuple[Path, dict] | None":
    d = case_dir(run_id, case.init_date, base)
    try:
        meta = json.loads((d / _META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if meta.get("run_id") != run_id or meta.get("init_date") != case.init_date.isoformat():
        return None
    return d, meta


def read_case_summary(run_id: str, case: ReplayCase,
                      base: Optional[Path] = None) -> Optional[dict]:
    """The case's summary, for the cycle list. Two small JSON reads, no Parquet."""
    found = _meta(run_id, case, base)
    if found is None:
        return None
    try:
        return json.loads((found[0] / _SUMMARY).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_case(run_id: str, case: ReplayCase,
              base: Optional[Path] = None) -> Optional[ScoredCycle]:
    """The case as a ScoredCycle, or None if there is not a complete one for this run."""
    found = _meta(run_id, case, base)
    if found is None:
        return None
    d, meta = found
    try:
        events = pd.read_parquet(d / _EVENTS)
        per_variable = pd.read_parquet(d / _PER_VARIABLE)
    except Exception:  # noqa: BLE001 - an unreadable artifact is a miss, never a crash
        return None
    if len(events) != meta.get("n_events"):
        return None
    return ScoredCycle(
        run_id=run_id,
        init_date=pd.Timestamp(case.init_date),
        events=events,
        per_variable=per_variable,
        n_rows_scored=len(events),
    )
