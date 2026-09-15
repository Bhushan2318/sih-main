"""Worker invoked once for the validation event frame by
`app.ml.pooled_training._run_event_frame_subprocess`.

Why this exists
-----------------
Unlike the training-year and test-year event builds (already isolated -
_build_pooled_year_events_worker.py, _build_pooled_test_events_worker.py),
`event_va = pv.build_event_frame(va, val_pred, ...)` ran directly in the parent
process, on `va` - the pool's validation "tail" slice, loaded once at the start of
`full_retrain_pooled` and held resident for the entire run. Real incident 2026-09-16
(v15): the parent process grew to ~35 GB private memory within ~20-25 minutes, still
inside the per-variable training loop (killed live, no traceback, so the exact
mechanism is not pinned down as precisely as the crashes that produced a stack trace).
`va` itself, plus this un-isolated `build_event_frame` call at the end, are both real,
plausible contributors and both fit the same "large frame processed in a long-lived
process" pattern every other fix this week addressed - so this closes that gap
regardless of which exact line was responsible.

`va` and `val_pred` are pickled in by the caller (already in the parent's memory at
call time; nothing new is read from disk here) - this only isolates the `build_event_
frame` computation itself, not a fresh read.
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
        from app.features import pivot as pv

        result["event_frame"] = pv.build_event_frame(
            job["paired"], job["pred_err"], job["p90_error"], job["bust_threshold"],
            job["hbf"])
    except Exception:
        result["error"] = traceback.format_exc()

    with open(args.out, "wb") as f:
        pickle.dump(result, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
