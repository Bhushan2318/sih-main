"""Worker invoked by `app.ml.pooled_training.full_retrain_pooled` once a run is saved.

Runs `pooled_training.finalize_for_serving` in a fresh process: it reads the validation
year again, and by then the parent has held a seventeen-year job's worth of pyarrow
arena. Returns only a short summary.

The parent passes its model directory explicitly and it is set before any app module is
imported, so this reads exactly the run the parent wrote. Inheriting MODEL_DIR through
the environment is not a guarantee: the test suite's conftest is imported twice and
rewrites it, and the worker then looked in a different directory than the parent had
written to.
"""
from __future__ import annotations

import argparse
import os
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

    result = {"summary": None, "error": None}
    try:
        if job.get("model_dir"):
            os.environ["MODEL_DIR"] = job["model_dir"]
        from app.ml.pooled_training import finalize_for_serving

        result["summary"] = finalize_for_serving(job["run_id"], Path(job["cache_dir"]))
    except Exception:
        result["error"] = traceback.format_exc()

    with open(args.out, "wb") as f:
        pickle.dump(result, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
