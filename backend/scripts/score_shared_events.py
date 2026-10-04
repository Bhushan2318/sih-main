"""Score a new run and the served run on the same events, against the new run's label.

The promotion gate used to compare two ROC-AUCs measured on different things: each run on
its own events, labelled with its own thresholds, and nothing checked that the events or
labels matched. With bias-corrected busts (label version 2) the two runs' labels differ by
design. This scores both runs as of issue time on one calendar year
(`score_run_on_year --as-of-issue`, each with its own regressors, features, thresholds and
bias), joins them on the event keys, and keeps the new run's label for both - "could each
model see this bust coming?" - so the gate compares like with like:

    python -m scripts.score_shared_events --new-run <RUN> --incumbent run_20260922T043925Z --year 2017 \\
        [--incumbent-reference data/analysis/cross_year_scores/run_20260922T043925Z_on_2017_as_of_issue.parquet]

`--incumbent-reference` is that run's scoring by the code it was trained with (the
`pre-overhaul` tag): its probabilities must agree here within 1e-5, which checks that
the incumbent is being scored exactly as before (feature and label version 1).
Then: `publish_serving_model --run-id <RUN> --shared-events <the file written here>`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EVENT_KEYS = ["region_id", "init_date", "valid_date", "lead_time_days"]
REFERENCE_TOLERANCE = 1e-5


def _keyed(df: pd.DataFrame, who: str) -> pd.DataFrame:
    k = df[EVENT_KEYS].copy()
    k["region_id"] = k["region_id"].astype(str)
    for c in ("init_date", "valid_date"):
        k[c] = pd.to_datetime(k[c])
    k["lead_time_days"] = k["lead_time_days"].astype(int)
    if k.duplicated().any():
        raise ValueError(f"{int(k.duplicated().sum())} duplicate event keys in the {who} "
                         "events - an event must appear once")
    return k


def shared_events(new: pd.DataFrame, incumbent: pd.DataFrame) -> pd.DataFrame:
    """The new run's events with both runs' probabilities and the new run's label.

    Every new-run event must have an incumbent probability, or the two scores would not
    describe the same events. Incumbent events the new label drops (no corrected error
    for any label variable) are left out."""
    a = _keyed(new, "new run").assign(y_bust=new["y_bust"].astype(int).to_numpy(),
                                      proba_new=new["model_proba"].to_numpy(dtype=float))
    b = _keyed(incumbent, "incumbent").assign(
        proba_incumbent=incumbent["model_proba"].to_numpy(dtype=float))
    out = a.merge(b, on=EVENT_KEYS, how="left")
    missing = int(out["proba_incumbent"].isna().sum())
    if missing:
        raise ValueError(f"{missing} of {len(out)} new-run events have no incumbent "
                         "probability - the two runs were not scored on the same events")
    return out.reset_index(drop=True)


def check_incumbent_reference(incumbent: pd.DataFrame, reference: pd.DataFrame) -> float:
    """Refuse unless the incumbent's probabilities equal its reference scoring (by the code
    it was trained with) on every shared event. Returns the largest difference."""
    a = _keyed(incumbent, "incumbent").assign(p=incumbent["model_proba"].to_numpy(dtype=float))
    b = _keyed(reference, "reference").assign(r=reference["model_proba"].to_numpy(dtype=float))
    m = a.merge(b, on=EVENT_KEYS, how="inner")
    if m.empty:
        raise ValueError("the incumbent and its reference scoring share no events")
    worst = float(np.max(np.abs(m["p"] - m["r"])))
    if worst > REFERENCE_TOLERANCE:
        raise ValueError(f"the incumbent's probabilities differ from its reference scoring by "
                         f"up to {worst:.2e} (> {REFERENCE_TOLERANCE:g}) - it is not being "
                         "scored as it was before; refusing to compare against it")
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-run", required=True)
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--incumbent-reference", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from app.config import settings
    from app.db.base import resolve_path
    from app.ml import classifier as clf_mod
    from app.ml import registry
    from scripts.score_run_on_year import score_run_on_year

    new_ev = score_run_on_year(args.new_run, args.year, as_of_issue_time=True)
    inc_ev = score_run_on_year(args.incumbent, args.year, as_of_issue_time=True)
    if args.incumbent_reference is not None:
        worst = check_incumbent_reference(inc_ev, pd.read_parquet(args.incumbent_reference))
        print(f"incumbent matches its reference scoring: max |difference| {worst:.2e}")
    out = shared_events(new_ev, inc_ev)
    manifest = json.loads((registry.run_dir(args.new_run) / "manifest.json").read_text())
    out = out.assign(new_run_id=args.new_run, incumbent_run_id=args.incumbent, year=args.year,
                     label_version=int(manifest.get("label_version", 1)))
    for who, col in (("new run", "proba_new"), ("incumbent", "proba_incumbent")):
        m = clf_mod._evaluate(out["y_bust"], out[col])
        print(f"{who}: n={m['n']:,} roc_auc={m['roc_auc']:.4f} brier={m['brier']:.4f}")
    path = args.out or (resolve_path(settings.data_dir) / "analysis" / "shared_events"
                        / f"{args.new_run}_vs_{args.incumbent}_on_{args.year}.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
