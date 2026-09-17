"""Run a queue of pooled-training jobs unattended, without one job's failure losing the
rest of the night.

Why this exists: every pooled-training crash on record (docs/known-issues.md - the OOM
crashes, the humidity_pct native crash) was found because someone was watching the
process, or found a job dead at some run count the next morning with no idea how far it
got. `full_retrain_pooled` itself is already crash-resistant per variable (each variable's
regressor runs in its own subprocess, with a retry + CPU fallback since 2026-09-16 -
app/ml/pooled_training.py). What was missing is the layer above that: nothing caught an
exception ESCAPING `full_retrain_pooled` itself (a bad job spec, an out-of-memory in the
parent, a KeyboardInterrupt from a flaky terminal) and nothing wrote progress anywhere
durable until the whole process exited. This script is that layer - one job's exception
is caught, logged with a full traceback, and the queue moves on; progress is written to
a JSON file after every job, not just at the end, so killing this script mid-run still
leaves a readable record of what finished.

    python -m scripts.run_pooled_overnight --queue queue.json --out overnight_report.json

`queue.json` is a list of {"train_years": [2016,2017,2018], "test_year": 2019} objects.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(log_path: Path, msg: str) -> None:
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_queue(queue: list[dict], cache_dir: Path, out_path: Path, log_path: Path) -> list[dict]:
    from app.config import settings
    if not settings.allow_local_retrain:
        raise RuntimeError("ALLOW_LOCAL_RETRAIN is not set to true - refusing, same guard "
                           "full_retrain_pooled's own caller uses.")

    from app.ml.pooled_training import full_retrain_pooled

    results: list[dict] = []
    if out_path.exists():
        # Resume: only a job that actually SUCCEEDED is skipped. Re-running a finished
        # multi-hour job on a re-dispatch is the casual re-run CLAUDE.md says not to do -
        # but a job that crashed produced no model and nothing to lose by trying again.
        # Real bug 2026-09-17: the first version of this skipped ANY recorded status,
        # including "exception" - a fold that crashed 32 minutes in would have been
        # silently skipped forever on every future resume, never producing a model, with
        # no error and no sign anything was wrong short of reading the full report.
        results = json.loads(out_path.read_text())
    done_keys = {(tuple(sorted(r["train_years"])), r["test_year"])
                for r in results if r.get("status") == "success"}
    # Superseded failed attempts stay in the log for history, but are dropped from the
    # report that gets rewritten below - a stale "exception" record next to this run's
    # fresh "success" for the identical job would just be confusing.
    results = [r for r in results
              if r.get("status") == "success"
              or (tuple(sorted(r["train_years"])), r["test_year"]) not in
                 {(tuple(sorted(j["train_years"])), j["test_year"]) for j in queue}]

    for i, job in enumerate(queue):
        key = (tuple(sorted(job["train_years"])), job["test_year"])
        if key in done_keys:
            _log(log_path, f"[{i+1}/{len(queue)}] already succeeded: "
                            f"train={job['train_years']} test={job['test_year']} - skipping")
            continue

        _log(log_path, f"[{i+1}/{len(queue)}] starting: train={job['train_years']} "
                        f"test={job['test_year']}")
        t0 = time.time()
        record = {"train_years": job["train_years"], "test_year": job["test_year"],
                  "started_at": _now()}
        try:
            report = full_retrain_pooled(job["train_years"], job["test_year"], cache_dir)
            record.update({
                "status": report.status,
                "run_id": report.run_id,
                "error": report.error,
                "modelled_variables": report.modelled_variables,
                "skipped_variables": report.skipped_variables,
                "classifier_test_roc_auc": (report.classifier_metrics.get("test", {})
                                            .get("roc_auc")),
                "seconds": report.seconds,
            })
            _log(log_path, f"[{i+1}/{len(queue)}] done: status={report.status} "
                            f"run_id={report.run_id} "
                            f"roc_auc={record['classifier_test_roc_auc']} "
                            f"skipped={list(report.skipped_variables)} "
                            f"({time.time()-t0:.0f}s)")
        except Exception as exc:  # noqa: BLE001 - the whole point: never let one job's
                                  # exception take the rest of the queue down with it.
            record.update({
                "status": "exception",
                "error": f"{exc}",
                "traceback": traceback.format_exc(),
                "seconds": time.time() - t0,
            })
            _log(log_path, f"[{i+1}/{len(queue)}] CRASHED: {exc!r} "
                            f"({time.time()-t0:.0f}s) - see traceback in {out_path}, "
                            f"continuing to next job")

        record["finished_at"] = _now()
        results.append(record)
        # Written after EVERY job, not just at the end - a report that only exists once
        # the whole queue finishes is no report at all if this process is killed at 3am.
        out_path.write_text(json.dumps(results, indent=2, default=str))

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", type=Path, required=True,
                    help="JSON file: list of {train_years, test_year}")
    ap.add_argument("--cache-dir", type=Path, default=BACKEND_DIR / "data" / "_pooled_cache")
    ap.add_argument("--out", type=Path, default=BACKEND_DIR / "data" / "overnight_report.json")
    ap.add_argument("--log", type=Path, default=BACKEND_DIR / "data" / "overnight.log")
    args = ap.parse_args()

    queue = json.loads(args.queue.read_text())
    _log(args.log, f"queue loaded: {len(queue)} jobs from {args.queue}")

    results = run_queue(queue, args.cache_dir, args.out, args.log)

    n_ok = sum(1 for r in results if r.get("status") == "success")
    n_bad = len(results) - n_ok
    _log(args.log, f"queue finished: {n_ok} succeeded, {n_bad} did not "
                    f"(see {args.out} for details)")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
