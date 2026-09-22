"""Worker invoked once by `app.ml.pooled_training._run_val_events_subprocess`.

Builds the validation event frame from the spilled validation rows, one batch of
forecast dates at a time (`pooled_training.val_event_frame`), and returns only the
event-grain result.

Why a subprocess: real crash 2026-09-22 (04:26 UTC). The parent was holding every
training cycle's events for seventeen years when it read the whole validation spill back
to build these events, and a 306 MB malloc failed. Built here, neither the frame nor the
pyarrow read arena ever touches the parent.
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
        from app.ml.pooled_training import val_event_frame

        result["event_frame"] = val_event_frame(
            job["spill_dir"], job["val_pred"], job["hbf"], job["p90_error"],
            job["bust_threshold"])
    except Exception:
        result["error"] = traceback.format_exc()

    with open(args.out, "wb") as f:
        pickle.dump(result, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
