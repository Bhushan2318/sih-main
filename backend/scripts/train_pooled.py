"""CLI for app.ml.pooled_training.full_retrain_pooled - train on any number of years
without materialising them all at once (see that module's docstring for why this exists
and app.ml.train_pipeline.full_retrain for the single-frame path this is not a
replacement for).

Always make_current=False; this never touches the promotion gate.

    python -m scripts.train_pooled --train-years 2016-2018 --test-year 2019 \
        --cache-dir data/_pooled_cache --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))


def _parse_years(spec: str) -> list:
    out: set = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(chunk))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-years", required=True, help="2016-2018, or 2016,2017,2018")
    ap.add_argument("--test-year", type=int, required=True)
    ap.add_argument("--cache-dir", type=Path, default=BACKEND_DIR / "data" / "_pooled_cache")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from app.config import settings
    if not settings.allow_local_retrain:
        print("ALLOW_LOCAL_RETRAIN is not set to true - refusing, same guard "
              "full_retrain uses.", file=sys.stderr)
        return 1

    from app.ml.pooled_training import full_retrain_pooled

    train_years = _parse_years(args.train_years)
    report = full_retrain_pooled(train_years, args.test_year, args.cache_dir)

    if args.json:
        print(json.dumps({
            "run_id": report.run_id, "status": report.status, "error": report.error,
            "split_cycles": report.split_cycles, "modelled_variables": report.modelled_variables,
            "skipped_variables": report.skipped_variables,
            "classifier_metrics": report.classifier_metrics, "seconds": report.seconds,
        }, indent=2, default=str))
    else:
        print(f"run_id={report.run_id} status={report.status} seconds={report.seconds:.0f}")
        if report.error:
            print(report.error, file=sys.stderr)
        print(f"modelled: {report.modelled_variables}")
        held = report.classifier_metrics.get("test") or report.classifier_metrics.get("val") or {}
        if held:
            print(f"held-out roc_auc={held.get('roc_auc')} n={held.get('n')}")

    return 0 if report.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
