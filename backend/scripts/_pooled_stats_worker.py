"""Worker invoked once per training run by `app.ml.pooled_training._run_pooled_stats_subprocess`.

Same rationale as the other three `_build_pooled_*_worker.py` / `_train_pooled_variable_
worker.py` scripts: `pooled_stats` concatenates every training year's thin frame (189M
rows for a real 3-year pool of densely-covered years) and then runs
`compute_historical_bust_frequency`'s own internal groupby over the whole thing - real
crash 2026-09-16 (v13), 3 minutes in, `ArrayMemoryError` allocating 1.41 GiB for the
groupby's sort-index array, in the PARENT process, before any of the already-isolated
per-variable/per-year stages even start. Isolating this call too means the parent never
holds the 189M-row concatenated frame at all - only the three small result dicts
(hbf, p90_error, bust_threshold) come back.
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

    result = {"hbf": None, "p90_error": None, "bust_threshold": None, "error": None}
    try:
        from app.ml.pooled_training import pooled_stats

        hbf, p90_error, bust_threshold = pooled_stats(
            job["cached"], job["train_years"], job["train_cycles"])
        result["hbf"], result["p90_error"], result["bust_threshold"] = (
            hbf, p90_error, bust_threshold)
    except Exception:
        result["error"] = traceback.format_exc()

    with open(args.out, "wb") as f:
        pickle.dump(result, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
