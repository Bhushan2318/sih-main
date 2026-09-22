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
        from app.ml.pooled_training import test_event_frame

        # One batch of forecast dates at a time - see test_event_frame. The whole-year
        # read is what failed for a training year on 2026-09-22 (41.6 GB peak commit).
        result["event_frame"], result["test_metrics"] = test_event_frame(
            job["cached_path"], job["test_cycles"], job["hbf"], job["p90_error"],
            job["bust_threshold"], job["artifacts"], job["columns"])
    except Exception:
        result["error"] = traceback.format_exc()

    with open(args.out, "wb") as f:
        pickle.dump(result, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
