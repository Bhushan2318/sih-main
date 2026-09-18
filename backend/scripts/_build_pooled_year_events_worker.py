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

Real crash 2026-09-17/18: process isolation alone was not enough - the SAME 584-586 MiB
groupby allocation failed again, twice, each ~1.8 hours into a real 3-year pool, inside
this already-isolated worker. `fold_models` (real XGBoost Booster/regressor objects, one
per variable per fold - up to 24 for 8 variables x 3 folds) and the `job` dict holding
them stay resident through the OOF-prediction loop and are never freed before
`build_event_frame`'s own groupby needs its contiguous block - `full_retrain_pooled`
itself already does exactly this cleanup ("del fold_models ... gc.collect()") after this
worker's subprocess returns, but that never helped the worker's OWN peak, only the
parent's. Freed explicitly below, right before the call that needs the headroom.

That cleanup alone was still not enough - the identical crash recurred through both
retry attempts on the next run. The actual remaining cost was `build_event_frame`'s own
defensive `paired.copy()`, doubling this frame's footprint right before the groupby that
needed the room the copy had just consumed. `df` here is freshly loaded by this worker
and read by nothing else afterward, so it is passed with `copy_input=False`.
"""
from __future__ import annotations

import argparse
import gc
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

            # fold_models (real Booster objects, up to 24 of them) and job (which holds
            # them, plus p90_error/bust_threshold we've already pulled out below) are
            # done being useful the moment oof is filled in - freed here, not left
            # resident through build_event_frame's own big allocation.
            p90_error, bust_threshold = job["p90_error"], job["bust_threshold"]
            del fold_models, job
            gc.collect()

            result["event_frame"] = pv.build_event_frame(
                df, oof, p90_error, bust_threshold, hbf, copy_input=False)
    except Exception:
        result["error"] = traceback.format_exc()

    with open(args.out, "wb") as f:
        pickle.dump(result, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
