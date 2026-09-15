"""Worker invoked for the held-out test year AND the validation slice by
`app.ml.pooled_training._run_final_model_events_subprocess`.

Same rationale as `_build_pooled_year_events_worker.py`, generalized to accept
multiple cached years (validation can span more than one pooled year) instead of one:
each year is read in full then reduced to event grain, never held alongside another
year, and never held in the PARENT at all.

Real incident 2026-09-16 (v18): with `[MEM]` checkpoints finally in place, the parent
process's own memory was traced directly (not guessed) - `va` alone, loaded once and
held resident for the whole run, was 39.65M rows and cost 11.7 GB RSS the moment it was
built, before training even started. That single un-isolated load explains the
"parent grows to ~35 GB" mystery from v15/v16 far better than the va/event_va-isolation
guesses tried first: those isolated the *build* of event_va, but the parent still had
to hold the full `va` frame for the entire training loop to slice per variable. This
worker removes that need entirely - the parent now passes only `cached_paths` +
`cycles` + `artifacts` (small), and reads/predicts/reduces to event grain here,
returning only the small event-level frame and each variable's metric.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with open(args.job, "rb") as f:
        job = pickle.load(f)

    result = {"event_frame": None, "metrics": {}, "error": None}
    try:
        import numpy as np
        import pandas as pd

        from app.features import pivot as pv
        from app.ml import regressors as reg_mod
        from app.ml.pooled_training import (
            attach_hbf_column, categorical_to_str, read_parquet_retrying,
        )

        columns = job["columns"]
        cols = sorted(columns) if columns else None
        parts = []
        for path in job["cached_paths"]:
            part = read_parquet_retrying(path, columns=cols)
            part = part[part["init_date"].isin(job["cycles"])]
            if part.empty:
                continue
            # Cast per year before concatenating - categories are not guaranteed to
            # align across cached years (see pooled_stats for the real crash this
            # caused once already).
            part["region_id"] = categorical_to_str(part["region_id"])
            part["season"] = categorical_to_str(part["season"])
            parts.append(part)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        if df.empty:
            result["event_frame"] = pd.DataFrame()
        else:
            hbf = job["hbf"]
            df = attach_hbf_column(df, hbf)
            pred = pd.Series(np.nan, index=df.index, dtype=float)
            for var, art in job["artifacts"].items():
                tmask = df["variable"] == var
                if not tmask.any():
                    continue
                p = reg_mod.predict_variable_error(art, df[tmask])
                pred.loc[tmask] = p
                if tmask.sum() >= 5:
                    result["metrics"][var] = reg_mod._evaluate(df.loc[tmask, "abs_error"], p)
            result["event_frame"] = pv.build_event_frame(
                df, pred, job["p90_error"], job["bust_threshold"], hbf)
    except Exception:
        result["error"] = traceback.format_exc()

    with open(args.out, "wb") as f:
        pickle.dump(result, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
