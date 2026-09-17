"""Worker invoked once for the held-out test year by
`app.ml.pooled_training._run_test_events_subprocess`.

Same rationale as `_build_pooled_year_events_worker.py`: the held-out test year is
read in full (it is deliberately never filtered down, per `full_retrain_pooled`'s own
comment - holding a year out entirely is the point), so it carries exactly the same
row-count risk that made a training year's event-frame build crash on 2026-09-15 (v9).
`test_year` was never actually reached in any attempt before that fix, so this is a
preemptive application of the same pattern rather than a fix for an observed crash -
cheaper than losing another multi-hour run to find out the hard way.
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

    result = {"event_frame": None, "test_metrics": {}, "error": None}
    try:
        import numpy as np
        import pandas as pd

        from app.features import pivot as pv
        from app.ml import regressors as reg_mod
        from app.ml.pooled_training import attach_hbf_column

        columns = job["columns"]
        df = pd.read_parquet(job["cached_path"], columns=sorted(columns) if columns else None)
        df = df[df["init_date"].isin(job["test_cycles"])]
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
                    result["test_metrics"][var] = reg_mod._evaluate(
                        df.loc[tmask, "abs_error"], p)
            result["event_frame"] = pv.build_event_frame(
                df, pred, job["p90_error"], job["bust_threshold"], hbf)
    except Exception:
        result["error"] = traceback.format_exc()

    with open(args.out, "wb") as f:
        pickle.dump(result, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
