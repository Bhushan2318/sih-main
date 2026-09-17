"""One job crashing must not lose the rest of the queue, and a re-dispatch after a crash
must not re-run jobs that already finished. Both are plumbing claims about the
orchestrator itself, checked against a fake `_run_job_subprocess` - no real subprocess,
no real data, no GPU, no multi-hour training.

Real crash 2026-09-17: the first version called `full_retrain_pooled` directly, in one
long-lived process, for every queued job - job 1 lost 5 of 8 variables to malloc()
failures and job 2 crashed 123s in from the pressure job 1 left behind. Each job now runs
as its own subprocess (`_run_job_subprocess`), so these tests patch at that boundary -
the real subprocess dispatch itself is exercised only by actually running the queue, not
by this file.
"""
from __future__ import annotations

import json

import pytest

from scripts import run_pooled_overnight as orch


def _queue(n=3):
    return [{"train_years": [2016 + i, 2017 + i], "test_year": 2019 + i} for i in range(n)]


def _fake_subprocess(results_by_year: dict):
    def fake(job, cache_dir, log_dir):
        return results_by_year[job["test_year"]]
    return fake


def test_one_job_crashing_does_not_stop_the_queue(tmp_path, monkeypatch):
    calls = []

    def fake(job, cache_dir, log_dir):
        calls.append(job["test_year"])
        if job["test_year"] == 2020:
            return {"status": "crashed", "error": "simulated OOM"}
        return {"status": "success", "run_id": f"run_{job['test_year']}",
               "classifier_metrics": {"test": {"roc_auc": 0.8}}}

    monkeypatch.setattr(orch, "_run_job_subprocess", fake)
    monkeypatch.setattr("app.config.settings.allow_local_retrain", True)

    out = tmp_path / "report.json"
    log = tmp_path / "log.txt"
    results = orch.run_queue(_queue(3), tmp_path, out, log)

    assert calls == [2019, 2020, 2021], "the crash on 2020 must not have stopped 2021"
    statuses = {r["test_year"]: r["status"] for r in results}
    assert statuses == {2019: "success", 2020: "crashed", 2021: "success"}
    assert "simulated OOM" in [r for r in results if r["test_year"] == 2020][0]["error"]


def test_progress_is_written_after_every_job_not_just_at_the_end(tmp_path, monkeypatch):
    written_after = []

    def fake(job, cache_dir, log_dir):
        out = tmp_path / "report.json"
        written_after.append(len(json.loads(out.read_text())) if out.exists() else 0)
        return {"status": "success", "run_id": f"run_{job['test_year']}"}

    monkeypatch.setattr(orch, "_run_job_subprocess", fake)
    monkeypatch.setattr("app.config.settings.allow_local_retrain", True)

    orch.run_queue(_queue(3), tmp_path, tmp_path / "report.json", tmp_path / "log.txt")
    assert written_after == [0, 1, 2], "each job should see one more completed job than the last"


def test_a_rerun_after_success_skips_jobs_already_recorded(tmp_path, monkeypatch):
    calls = []

    def fake(job, cache_dir, log_dir):
        calls.append(job["test_year"])
        return {"status": "success", "run_id": f"run_{job['test_year']}"}

    monkeypatch.setattr(orch, "_run_job_subprocess", fake)
    monkeypatch.setattr("app.config.settings.allow_local_retrain", True)

    out, log = tmp_path / "report.json", tmp_path / "log.txt"
    queue = _queue(3)
    orch.run_queue(queue, tmp_path, out, log)
    assert calls == [2019, 2020, 2021]

    calls.clear()
    orch.run_queue(queue, tmp_path, out, log)
    assert calls == [], "every job already succeeded - none should re-run"


def test_a_rerun_after_a_crash_retries_the_crashed_job_not_skips_it(tmp_path, monkeypatch):
    """Real bug 2026-09-17: the first version of resume treated ANY recorded status,
    including a crash, as done - so a fold that crashed would be skipped forever on
    every future resume, silently never producing a model."""
    calls = []
    should_crash = {2020: True}

    def fake(job, cache_dir, log_dir):
        calls.append(job["test_year"])
        if should_crash.get(job["test_year"]):
            return {"status": "crashed", "error": "simulated OOM"}
        return {"status": "success", "run_id": f"run_{job['test_year']}"}

    monkeypatch.setattr(orch, "_run_job_subprocess", fake)
    monkeypatch.setattr("app.config.settings.allow_local_retrain", True)

    out, log = tmp_path / "report.json", tmp_path / "log.txt"
    queue = _queue(3)
    orch.run_queue(queue, tmp_path, out, log)
    assert calls == [2019, 2020, 2021]

    # The underlying bug is now fixed (as if a code fix landed between runs).
    should_crash[2020] = False
    calls.clear()
    results = orch.run_queue(queue, tmp_path, out, log)

    assert calls == [2020], "only the previously-crashed job should retry"
    statuses = {r["test_year"]: r["status"] for r in results}
    assert statuses == {2019: "success", 2020: "success", 2021: "success"}
    assert len(results) == 3, "the stale crash record for 2020 must be replaced, not duplicated"


def test_refuses_without_allow_local_retrain(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.allow_local_retrain", False)
    with pytest.raises(RuntimeError, match="ALLOW_LOCAL_RETRAIN"):
        orch.run_queue(_queue(1), tmp_path, tmp_path / "report.json", tmp_path / "log.txt")


# --- _run_job_subprocess itself: the actual subprocess boundary ------------------------
# One real subprocess dispatch (a deliberately trivial, fast job) to pin the JSON-parsing
# contract against train_pooled.py's real --json output shape, not a hand-typed fixture
# that might drift from what the CLI actually prints.

def test_run_job_subprocess_parses_a_no_grids_style_refusal(tmp_path, monkeypatch):
    """Doesn't need real data: an empty cache_dir makes cache_year fail fast (no store to
    read), which is enough to prove the subprocess boundary itself - launch, capture
    stdout/stderr to files, parse the trailing JSON line - works end to end."""
    monkeypatch.setenv("ALLOW_LOCAL_RETRAIN", "true")
    job = {"train_years": [2099], "test_year": 2100}  # years that cannot exist
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    result = orch._run_job_subprocess(job, tmp_path / "empty_cache", log_dir)
    # Whatever the exact failure mode (ValueError from cache_year, a refusal, a crash),
    # this must come back as a parsed dict with a status - never raise, never hang.
    assert isinstance(result, dict)
    assert "status" in result
    assert (log_dir / "job_test2100_stdout.log").exists()
    assert (log_dir / "job_test2100_stderr.log").exists()
