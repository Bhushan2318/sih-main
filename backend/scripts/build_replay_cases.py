"""Build Replay's past events for one run, from the reforecast archive.

    python -m scripts.build_replay_cases --run-id run_20260922T043925Z [--only kerala-2018]

Workstation only, against the training store. Each case is a whole 666-district cycle
scored as of its init date, which is the 1,406 MB scoring path the serving box refuses
(app/ml/precomputed.py). What it writes is trimmed to what Replay reads and lands inside
the run directory, `data/models/<run_id>/replay_cases/`, so the cases ship with the model
that scored them: scripts/publish_serving_model.py packs the whole run directory.

For each case in app/services/replay_cases.CASES it refuses, rather than writes:
- a case from one of the run's training years (read from the run's own manifest);
- a cycle short of any district or any of lead days 1-10 - a map quietly missing
  districts says nothing about itself;
- a case with no observation of its variable over its district on its peak day, because
  then there is nothing to check the forecast against.

Then it prints what the case shows, whatever that is - the forecast against ERA5 over the
focus district on the peak day, and the bust probability the model gave it. None of those
numbers are written anywhere but the artifact; Replay reads them from there.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

import pandas as pd  # noqa: E402

from app.services.replay_cases import CaseRefused, ReplayCase  # noqa: E402

LEAD_DAYS = range(1, 11)


def check_complete(sc, case: ReplayCase, n_regions: "int | None" = None,
                   lead_days=LEAD_DAYS) -> None:
    """Raise CaseRefused unless this scored cycle can show the whole case."""
    if sc is None or sc.events.empty:
        raise CaseRefused(f"{case.id}: nothing scored for {case.init_date}")
    if n_regions is None:
        from app.utils import india_districts
        n_regions = len(india_districts.load_registry())

    ev = sc.events
    per_lead = ev.groupby("lead_time_days")["region_id"].nunique()
    missing_leads = sorted(set(int(d) for d in lead_days) - set(int(d) for d in per_lead.index))
    if missing_leads:
        raise CaseRefused(f"{case.id}: lead days {missing_leads} were not scored")
    short = per_lead[per_lead < n_regions]
    if not short.empty:
        raise CaseRefused(
            f"{case.id}: {int(short.min())} districts scored on Day {int(short.idxmin())}, "
            f"{n_regions} expected - refusing a map short of districts")

    pv = sc.per_variable
    seen = pv[(pv["region_id"].astype(str) == case.focus_region_id)
              & (pv["variable"].astype(str) == case.focus_variable)
              & (pd.to_datetime(pv["valid_date"]).dt.date == case.peak_valid_date)
              & pv["observed_value"].notna()]
    if seen.empty:
        raise CaseRefused(
            f"{case.id}: no observed {case.focus_variable} over {case.focus_region_id} on "
            f"{case.peak_valid_date}; nothing to check the forecast against")


def case_summary(state, sc, case: ReplayCase, note: str) -> dict:
    """The case's cycle-list entry, built by the same code that builds a live cycle's."""
    from app.services import replay_service

    s = replay_service.summary_from_scored(
        state, sc, kind="event", title=case.title, sample_note=note,
        focus_region_id=case.focus_region_id, focus_variable=case.focus_variable)
    if s is None:
        raise CaseRefused(f"{case.id}: nothing to summarise")
    return s.model_dump(mode="json")


def report(state, sc, case: ReplayCase) -> None:
    """What the case shows, printed from the scores."""
    ev, pv = sc.events, sc.per_variable
    lead = (case.peak_valid_date - case.init_date).days + 1
    thr = state.thresholds.bust_threshold.get(case.focus_variable)
    print(f"  districts {ev['region_id'].nunique()}, lead days "
          f"{sorted(int(d) for d in ev['lead_time_days'].unique())}, events {len(ev)}")

    row = pv[(pv["region_id"].astype(str) == case.focus_region_id)
             & (pv["variable"].astype(str) == case.focus_variable)
             & (pd.to_datetime(pv["valid_date"]).dt.date == case.peak_valid_date)]
    if not row.empty:
        r = row.iloc[0]
        err = abs(float(r["predicted_value"]) - float(r["observed_value"]))
        against = f" against a bust threshold of {thr:.1f}" if thr else ""
        print(f"  {case.focus_region_id} {case.focus_variable} on {case.peak_valid_date} "
              f"(Day {lead}): forecast {float(r['predicted_value']):.1f}, ERA5 "
              f"{float(r['observed_value']):.1f}, error {err:.1f}{against}")

    day = ev[ev["lead_time_days"] == lead].sort_values("bust_probability", ascending=False)
    day = day.reset_index(drop=True)
    hit = day.index[day["region_id"].astype(str) == case.focus_region_id]
    if len(hit):
        i = int(hit[0])
        r = day.iloc[i]
        print(f"  P(bust) there on Day {lead}: {float(r['bust_probability']):.3f} "
              f"({r['risk_band']}, rank {i + 1} of {len(day)}, driven by "
              f"{r['dominant_variable']})")
    own = ev[ev["region_id"].astype(str) == case.focus_region_id].sort_values("lead_time_days")
    print("  its P(bust) by lead day: " + ", ".join(
        f"D{int(x.lead_time_days)} {float(x.bust_probability):.2f}" for x in own.itertuples()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--only", action="append", help="case id to build (repeatable)")
    args = ap.parse_args()

    from app.ml import inference
    from app.services import replay_cases

    state = inference.load_model_state(args.run_id)
    if state is None or state.run_id != args.run_id:
        print(f"cannot load {args.run_id}", file=sys.stderr)
        return 1

    cases = [c for c in replay_cases.CASES if not args.only or c.id in args.only]
    refused = []
    for case in cases:
        print(f"{case.id}: {case.title}, init {case.init_date} 00 UTC")
        t0 = time.time()
        try:
            # The sample check comes first: a training-year case is not worth scoring.
            note = replay_cases.sample_note(case.init_date.year, state.manifest)
            sc = inference.score_cycle(state, case.init_date, as_of_init=True)
            check_complete(sc, case)
            summary = case_summary(state, sc, case, note)
            out = replay_cases.write_case(state.run_id, case, sc, summary)
        except CaseRefused as exc:
            print(f"  REFUSED: {exc}")
            refused.append(case.id)
            continue
        print(f"  {note}")
        report(state, sc, case)
        size = sum(p.stat().st_size for p in out.iterdir()) / 1e6
        print(f"  wrote {out} ({size:.2f} MB) in {time.time() - t0:.0f}s")
        inference.invalidate_caches()  # one full cycle in memory at a time

    if refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
