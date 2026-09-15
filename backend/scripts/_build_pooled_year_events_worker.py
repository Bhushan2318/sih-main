"""Worker invoked once per training year by `app.ml.pooled_training._run_year_events_subprocess`.

Why this is a separate process
--------------------------------
`build_pooled_train_events` streams one cached year at a time so no more than one
year's full-width frame is ever resident - but the per-year body itself (load, attach
OOF predictions, then `pivot.build_event_frame`'s groupby/pivot_table) still ran inside
`full_retrain_pooled`'s own long-lived process. That process had already dispatched
every variable's training to a subprocess (see `_train_pooled_variable_worker.py`) by
the time this loop runs, but a real crash 2026-09-15 (v9) showed this loop is its own
source of the same fragmentation failure: pandas' groupby internals need a large int64
codes array (`safe_sort`/`take_nd`) sized to one year's row count - 76.7M rows for a
densely-covered year - and `ArrayMemoryError`'d trying to allocate ~586 MiB for it
despite gigabytes of nominally free memory, the same "free but not contiguous"
signature as every earlier crash in this file's history.

So each year's raw-frame work (the part with real memory cost) now happens in a fresh
process that exits right after, returning only the small, already event-reduced
DataFrame to the parent - mirroring the fix already applied to per-variable training.
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

    result = {"event_frame": None, "error": None}
    try:
        import numpy as np
        import pandas as pd

        from app.features import pivot as pv
        from app.ml import regressors as reg_mod
        from app.ml.pooled_training import attach_hbf_column

        columns = job["columns"]
        df = pd.read_parquet(job["cached_path"], columns=sorted(columns) if columns else None)
        df = df[df["init_date"].isin(job["train_cycles"])]
        if df.empty:
            result["event_frame"] = pd.DataFrame()
        else:
            hbf = job["hbf"]
            df = attach_hbf_column(df, hbf)
            df["_fold"] = df["init_date"].map(job["fold_of"])
            oof = pd.Series(np.nan, index=df.index, dtype=float)
            fold_models = job["fold_models"]
            for variable in sorted(df["variable"].unique()):
                models_for_var = fold_models.get(variable, {})
                if not models_for_var:
                    continue
                vmask = df["variable"] == variable
                for fold, (model, cols) in models_for_var.items():
                    fmask = vmask & (df["_fold"] == fold)
                    if not fmask.any():
                        continue
                    oof.loc[fmask] = model.predict(reg_mod._prep_X(df.loc[fmask], cols))
            result["event_frame"] = pv.build_event_frame(
                df, oof, job["p90_error"], job["bust_threshold"], hbf)
    except Exception:
        result["error"] = traceback.format_exc()

    with open(args.out, "wb") as f:
        pickle.dump(result, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
