"""Score a saved run's classifier against a full calendar year, independent of whatever
that run's own train/val/test split was.

Why this exists: two runs trained on different years cannot be compared by their own
held-out ROC-AUC, because the chronological within-year split always tests on Nov-Dec -
the model's easiest season (see docs/known-issues.md). A model trained on 2016, one on
2018, one on 2019 each report a number from a *different* test set. To compare them, every
run needs to be scored on the *same* year, so the difference measured is the model, not
the season it happened to be tested on.

Deliberately bounded to one calendar year per call (`--init-date-min`/`--init-date-max`
inside `_build_paired_in_chunks`, same as a normal retrain) - one year is ~76M paired rows,
~7.6 GB, the scale already proven to fit. Scoring several runs on the same year means
paying that build cost once per run, not once per comparison.

    python -m scripts.score_run_on_year --run-id run_20260911T163128Z --year 2018
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def score_run_on_year(run_id: str, year: int) -> pd.DataFrame:
    """The saved run's own regressors + classifier + thresholds, applied to one full
    calendar year of paired data the run may or may not have trained on.

    Reuses exactly what live serving already does (`app.ml.inference`): load the run's
    artifacts once, predict each variable's error with the run's own regressors (not
    out-of-fold - this data is genuinely held out from that run, whether or not the run
    ever saw the year), build the event frame with the run's own p90/bust thresholds and
    historical bust frequency (computed on *that* run's training split, not recomputed
    here - recomputing it on the scoring year would leak that year's own statistics into
    its own label), then score with the run's own classifier. Returns a frame with
    `model_proba` and `y_bust`, the same two columns `_emit_eval_events` and
    `scripts/run_baselines.py` key off, plus `scored_run_id`/`scored_year` for traceability
    when several runs' outputs get compared side by side.
    """
    from app.ml import inference
    from app.ml.train_pipeline import _build_paired_in_chunks
    from app.features import engineering as fe
    from app.features import pivot as pv

    state = inference.load_model_state(run_id)
    if state is None:
        raise ValueError(f"no complete saved model state for {run_id} - need regressors, "
                         f"a classifier and thresholds all present in data/models/{run_id}")

    lo = pd.Timestamp(f"{year}-01-01")
    hi = pd.Timestamp(f"{year}-12-31")
    paired, _ = _build_paired_in_chunks(init_date_min=lo, init_date_max=hi)
    if paired.empty:
        raise ValueError(f"no paired forecast+observation data for {year} in the store")

    # The run's own train-split historical bust frequency, not one recomputed on the
    # scoring year - see the docstring above.
    hbf = state.historical_bust_freq
    key = list(zip(paired["region_id"].astype(str), paired["season"].astype(str)))
    paired["historical_bust_frequency_region_season"] = [hbf.get(k, np.nan) for k in key]
    # Same reasoning for the jump climatology (C1): the run's own training split, never
    # the scoring year's.
    fe.attach_jump_climatology(paired, state.jump_climatology)

    pred = pd.Series(np.nan, index=paired.index, dtype=float)
    for var, (model, cols) in state.regressors.items():
        mask = paired["variable"] == var
        if not mask.any():
            continue
        X = inference._prep(paired.loc[mask], cols,
                            inference.categorical_features(model))
        pred.loc[mask] = model.predict(X)

    events = pv.build_event_frame(
        paired, pred, state.thresholds.p90_error, state.thresholds.bust_threshold, hbf)
    if events.empty or "y_bust" not in events:
        raise ValueError(f"{year} produced no scoreable events against {run_id}'s thresholds")

    X_evt = inference._prep(events, state.classifier_columns,
                            inference.categorical_features(state.classifier))
    events["model_proba"] = state.classifier.predict_proba(X_evt)[:, 1]
    events["split"] = "test"
    events["scored_run_id"] = run_id
    events["scored_year"] = year
    return events


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write the scored events (default: "
                         "data/analysis/cross_year_scores/<run_id>_on_<year>.parquet)")
    args = ap.parse_args()

    from app.config import settings
    from app.db.base import resolve_path
    from app.ml import classifier as clf_mod

    events = score_run_on_year(args.run_id, args.year)
    metrics = clf_mod._evaluate(events["y_bust"], events["model_proba"])
    print(f"{args.run_id} scored on {args.year}: n={metrics['n']:,} "
          f"bust_rate={metrics['bust_rate']:.3f} roc_auc={metrics['roc_auc']:.4f} "
          f"brier={metrics['brier']:.4f}")

    out = args.out or (resolve_path(settings.data_dir) / "analysis" / "cross_year_scores"
                       / f"{args.run_id}_on_{args.year}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    for col in events.columns:
        if str(events[col].dtype) == "category":
            events[col] = events[col].astype(str)
    events.to_parquet(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
