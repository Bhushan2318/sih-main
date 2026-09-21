"""Worker invoked once by `app.ml.pooled_training._run_val_frame_subprocess`.

Builds the pooled validation frame and writes it to disk partitioned by variable,
instead of returning it. Nothing comes back but a row count.

Why a subprocess at all: the parent has to spawn a per-variable training worker that
needs a single ~9.19 GB allocation for XGBoost's quantised matrix over seventeen years.
Building the validation frame in the parent left it resident at 16.8 GB of a 23.7 GB
machine - pyarrow's read arena is not returned to the OS when the frame is freed - so
every one of the eight variables failed that malloc and the run produced a model with
zero variables (2026-09-21). Built here instead, the whole cost dies with this process
and the parent never allocates it.

Partitioned by variable because that is exactly how it is consumed: each training worker
wants one variable's rows and nothing else.
"""
from __future__ import annotations

import argparse
import pickle
import shutil
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

    result = {"n_rows": 0, "error": None}
    try:
        import pandas as pd

        from app.ml.pooled_training import attach_hbf_column

        out_dir = Path(job["out_dir"])
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        columns = sorted(job["columns"])
        val_cycles = job["val_cycles"]
        hbf = job["hbf"]
        next_row = 0
        for year in job["val_years"]:
            part = pd.read_parquet(job["cached"][year], columns=columns)
            part = part[part["init_date"].isin(val_cycles)]
            if part.empty:
                del part
                continue
            # One shared row numbering across years, so the parent can hold a plain
            # RangeIndex for val_pred rather than a 77-million-element index array.
            part = part.reset_index(drop=True)
            part["_va_row"] = range(next_row, next_row + len(part))
            next_row += len(part)
            part = attach_hbf_column(part, hbf)
            # variable= directories, appended to across years: a training worker then
            # reads one variable's rows without touching any other variable's.
            part.to_parquet(out_dir, partition_cols=["variable"], index=False)
            del part
        result["n_rows"] = next_row
    except Exception:  # noqa: BLE001 - reported to the parent, never raised across it
        result["error"] = traceback.format_exc()

    with open(args.out, "wb") as f:
        pickle.dump(result, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
