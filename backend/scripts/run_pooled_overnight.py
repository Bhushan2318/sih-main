"""Run a queue of pooled-training jobs unattended, without one job's failure losing the
rest of the night.

Why this exists: every pooled-training crash on record (docs/known-issues.md - the OOM
crashes, the humidity_pct native crash) was found because someone was watching the
process, or found a job dead at some run count the next morning with no idea how far it
got. `full_retrain_pooled` itself is already crash-resistant per variable (each variable's
regressor runs in its own subprocess, with a retry + CPU fallback since 2026-09-16 -
app/ml/pooled_training.py). What was missing is the layer above that: nothing caught an
exception ESCAPING `full_retrain_pooled` itself, and nothing wrote progress anywhere
durable until the whole process exited.

Real crash 2026-09-17, running two jobs back to back: the FIRST version of this script
called `full_retrain_pooled` directly, in-process, for every queued job - so two jobs ran
sequentially inside one long-lived Python process. Job 1 (test_year=2016) itself lost 5 of
its 8 variables to plain `malloc()` failures (soil_moisture_pct, temperature_c,
wind_direction_deg, wind_speed_ms, atmospheric_moisture_kgm2) despite "succeeding" overall
- and job 2 (test_year=2017) crashed 123 seconds in with a MemoryError on an allocation an
order of magnitude smaller than what job 1 had just been juggling. This machine has ~24 GB
RAM (matches the ceiling already documented elsewhere in known-issues.md); every
fragmentation crash this codebase has ever actually fixed was fixed by process-level
isolation, never by `gc.collect()` in a long-lived process - Python's GC cannot defragment
a process's native heap, only the OS reclaiming the whole process can. This script did not
extend that isolation to the JOB level, so job 1's pressure bled straight into job 2.

Each queued job now runs as its own completely fresh subprocess (`scripts/train_pooled.py
--json`), the same way each variable and each year already does inside it. stdout/stderr
go to files, never `capture_output=True` - the exact pattern that caused a 35 GB parent
memory balloon on 2026-09-15 (see `_run_worker_subprocess` in app/ml/pooled_training.py).

    python -m scripts.run_pooled_overnight --queue queue.json --out overnight_report.json

`queue.json` is a list of {"train_years": [2016,2017,2018], "test_year": 2019} objects.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
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


def _parse_trailing_json(stdout_text: str) -> dict | None:
    """The last top-level (column-0) `{...}` block in `stdout_text`, or `None` if there
    isn't one. See `_run_job_subprocess`'s docstring for why "last line starting with {"
    (matches a nested object too) is wrong and this is column-0-exact instead."""
    lines = stdout_text.splitlines()
    start = next((i for i in range(len(lines) - 1, -1, -1) if lines[i] == "{"), None)
    if start is None:
        return None
    return json.loads("\n".join(lines[start:]))


def _run_job_subprocess(job: dict, cache_dir: Path, log_dir: Path) -> dict:
    """One job, in a brand-new process. Mirrors `_run_worker_subprocess` in
    app/ml/pooled_training.py: stdout/stderr to files, never captured in memory.

    `train_pooled.py --json` prints `json.dumps({...}, indent=2)` - a PRETTY-PRINTED,
    multi-line object, not a single line. Real bug 2026-09-17: the first version looked
    for "the last line starting with '{'", which matches a NESTED object's opening brace
    (e.g. inside "split_cycles": {) just as readily as the real top-level one, and grabbed
    the wrong one - a training run that genuinely finished (returncode 0, a full result in
    the log file) was reported as a parse failure and its real result discarded. `indent=2`
    means the outer brace is always alone on its own line at column 0; a nested brace is
    always indented. Take the LAST line that is exactly "{" with no leading whitespace."""
    years_arg = ",".join(str(y) for y in sorted(job["train_years"]))
    test_year = job["test_year"]
    stdout_path = log_dir / f"job_test{test_year}_stdout.log"
    stderr_path = log_dir / f"job_test{test_year}_stderr.log"
    env = {**os.environ, "ALLOW_LOCAL_RETRAIN": "true"}
    with open(stdout_path, "wb") as out_f, open(stderr_path, "wb") as err_f:
        proc = subprocess.run(
            [sys.executable, "-m", "scripts.train_pooled",
             "--train-years", years_arg, "--test-year", str(test_year),
             "--cache-dir", str(cache_dir), "--json"],
            stdout=out_f, stderr=err_f, cwd=BACKEND_DIR, env=env,
        )
    stdout_text = stdout_path.read_text(errors="replace")
    try:
        result = _parse_trailing_json(stdout_text)
    except json.JSONDecodeError as exc:
        return {"status": "crashed", "returncode": proc.returncode,
               "error": f"could not parse JSON output: {exc}\nstdout tail: {stdout_text[-2000:]}"}
    if result is None:
        tail = stderr_path.read_text(errors="replace")[-4000:]
        return {"status": "crashed", "returncode": proc.returncode,
               "error": f"no JSON output from subprocess (rc={proc.returncode}): {tail}"}
    return result


def run_queue(queue: list[dict], cache_dir: Path, out_path: Path, log_path: Path) -> list[dict]:
    from app.config import settings
    if not settings.allow_local_retrain:
        raise RuntimeError("ALLOW_LOCAL_RETRAIN is not set to true - refusing, same guard "
                           "full_retrain_pooled's own caller uses.")

    log_dir = out_path.parent
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
        result = _run_job_subprocess(job, cache_dir, log_dir)
        result.setdefault("seconds", time.time() - t0)
        record.update(result)
        record["classifier_test_roc_auc"] = (
            (result.get("classifier_metrics") or {}).get("test", {}).get("roc_auc"))
        if record.get("status") == "success":
            _log(log_path, f"[{i+1}/{len(queue)}] done: status=success "
                            f"run_id={record.get('run_id')} "
                            f"roc_auc={record['classifier_test_roc_auc']} "
                            f"skipped={list(record.get('skipped_variables') or {})} "
                            f"({record['seconds']:.0f}s)")
        else:
            _log(log_path, f"[{i+1}/{len(queue)}] CRASHED: {record.get('error')!r} "
                            f"({record['seconds']:.0f}s) - see {out_path} and "
                            f"data/job_test{job['test_year']}_std{{out,err}}.log, "
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
