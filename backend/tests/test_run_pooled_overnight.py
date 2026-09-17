"""One job crashing must not lose the rest of the queue, and a re-dispatch after a crash
must not re-run jobs that already finished. Both are plumbing claims about the
orchestrator itself, checked against a fake full_retrain_pooled - no real data, no GPU,
no multi-hour training."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from scripts import run_pooled_overnight as orch


@dataclass
class _FakeReport:
    run_id: str
    status: str
    error: str | None = None
    modelled_variables: list = field(default_factory=list)
    skipped_variables: dict = field(default_factory=dict)
    classifier_metrics: dict = field(default_factory=dict)
    seconds: float = 1.0


def _queue(n=3):
    return [{"train_years": [2016 + i, 2017 + i], "test_year": 2019 + i} for i in range(n)]


def test_one_job_crashing_does_not_stop_the_queue(tmp_path, monkeypatch):
    calls = []

    def fake_full_retrain_pooled(train_years, test_year, cache_dir):
        calls.append(test_year)
        if test_year == 2020:
            raise MemoryError("simulated OOM")
        return _FakeReport(run_id=f"run_{test_year}", status="success",
                           classifier_metrics={"test": {"roc_auc": 0.8}})

    monkeypatch.setattr(
        "app.ml.pooled_training.full_retrain_pooled", fake_full_retrain_pooled)
    monkeypatch.setattr("app.config.settings.allow_local_retrain", True)

    out = tmp_path / "report.json"
    log = tmp_path / "log.txt"
    results = orch.run_queue(_queue(3), tmp_path, out, log)

    assert calls == [2019, 2020, 2021], "the crash on 2020 must not have stopped 2021"
    statuses = {r["test_year"]: r["status"] for r in results}
    assert statuses == {2019: "success", 2020: "exception", 2021: "success"}
    assert "simulated OOM" in [r for r in results if r["test_year"] == 2020][0]["error"]


def test_progress_is_written_after_every_job_not_just_at_the_end(tmp_path, monkeypatch):
    written_after = []

    def fake_full_retrain_pooled(train_years, test_year, cache_dir):
        # The out file must already reflect every PRIOR job by the time this one runs.
        out = tmp_path / "report.json"
        written_after.append(len(json.loads(out.read_text())) if out.exists() else 0)
        return _FakeReport(run_id=f"run_{test_year}", status="success")

    monkeypatch.setattr(
        "app.ml.pooled_training.full_retrain_pooled", fake_full_retrain_pooled)
    monkeypatch.setattr("app.config.settings.allow_local_retrain", True)

    orch.run_queue(_queue(3), tmp_path, tmp_path / "report.json", tmp_path / "log.txt")
    assert written_after == [0, 1, 2], "each job should see one more completed job than the last"


def test_a_rerun_after_success_skips_jobs_already_recorded(tmp_path, monkeypatch):
    calls = []

    def fake_full_retrain_pooled(train_years, test_year, cache_dir):
        calls.append(test_year)
        return _FakeReport(run_id=f"run_{test_year}", status="success")

    monkeypatch.setattr(
        "app.ml.pooled_training.full_retrain_pooled", fake_full_retrain_pooled)
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
    including a crash, as done - so a fold that crashed 32 minutes in would be skipped
    forever on every future resume, silently never producing a model."""
    calls = []
    should_crash = {2020: True}

    def fake_full_retrain_pooled(train_years, test_year, cache_dir):
        calls.append(test_year)
        if should_crash.get(test_year):
            raise MemoryError("simulated OOM")
        return _FakeReport(run_id=f"run_{test_year}", status="success")

    monkeypatch.setattr(
        "app.ml.pooled_training.full_retrain_pooled", fake_full_retrain_pooled)
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
