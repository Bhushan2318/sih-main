"""Worker invoked once per variable by `app.ml.pooled_training._run_variable_subprocess`.

Why this is a separate process, not a function call
-----------------------------------------------------
`full_retrain_pooled` used to call `train_variable_regressor_pooled` and
`oof_fold_models` directly, once per variable, inside one long-lived process shared
across every variable and both the GPU and CPU thread. That process accumulated
dozens of short-lived DataIter/QuantileDMatrix/Booster objects (each wrapping native
pyarrow/XGBoost C++ allocations) across the whole run, and repeatedly crashed with
`ArrayMemoryError` after 1-3 hours - not because total memory ran out, but because
Windows' heap allocator could not find one contiguous block big enough, even with
free space available (fragmentation). `gc.collect()` at every fit boundary (see
pooled_training.py's history) shrank individual allocations and delayed the crash,
but never fixed the underlying cause, because Python's GC cannot defragment a
process's native heap - only the OS reclaiming the whole process can.

So each variable's regressor + its out-of-fold models (~4 XGBoost fits: one main
fit, `n_splits` fold fits) now run in a fresh process that exits right after,
unconditionally handing every native allocation it made back to the OS. The parent
(`_train_group` in pooled_training.py) never accumulates fragmentation across
variables because it never holds the native memory in the first place - only this
worker does, for one variable at a time.

Job/result contract: both are pickled dicts (local-only, not a security boundary) -
see `_run_variable_subprocess` for the exact keys.
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

    result = {"artifact": None, "skipped": None, "val_pred": None,
              "fold_models": {}, "error": None}
    try:
        from app.ml import pooled_training as pt
        from app.ml import regressors as reg_mod

        art = pt.train_variable_regressor_pooled(
            job["cached"], job["train_years"], job["variable"], job["train_cycles"],
            job["va_var"], job["hbf"], job["cache_dir"], device=job["device"])
        if art is None:
            result["skipped"] = "regressor training returned None or too few rows"
        else:
            result["artifact"] = art
            if len(job["va_var"]):
                import pandas as pd
                preds = reg_mod.predict_variable_error(art, job["va_var"])
                result["val_pred"] = pd.Series(preds, index=job["va_var"].index)
        result["fold_models"] = pt.oof_fold_models(
            job["cached"], job["train_years"], job["variable"], job["train_cycles"],
            job["hbf"], job["fold_of"], job["cache_dir"], device=job["device"])
    except Exception as exc:  # noqa: BLE001 - reported to the parent, not swallowed
        result["error"] = traceback.format_exc()
        result["skipped"] = f"worker exception: {exc}"

    with open(args.out, "wb") as f:
        pickle.dump(result, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
