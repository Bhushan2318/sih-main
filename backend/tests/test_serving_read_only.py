"""The serving box must refuse work it cannot survive, not attempt it.

Measured on a real 666-district cycle (app/ml/precomputed.py): scoring on the box peaks at
1,406 MB; the box is killed at 512 MB and `/api/health` read 509 MB on 2026-09-24. Replay
and the ensemble endpoint accept any `init_date`, and any cycle CI did not precompute fell
through to that scoring path, so one curious request could take the site down. Uploads
wrote into the serving store with no guard at all.

`SERVING_READ_ONLY` is set only in the Docker image Render runs. These tests pin both
sides: on the box a missing precomputed answer is a clear 409, never a live score; off the
box (CI precompute, local dev, tests) scoring still falls through as before.

The model state below is a plumbing stand-in (it only carries a run_id); nothing here
produces a metric.
"""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.ml import inference


class _State:
    run_id = "run_PLUMBING"


def _no_artifact(monkeypatch):
    monkeypatch.setattr(inference.precomputed, "read_scored_cycle", lambda *a, **k: None)
    monkeypatch.setattr(inference.parquet_store, "store_fingerprint", lambda: "fp")
    inference._score_cache.clear()


def test_box_refuses_to_score_a_cycle_ci_did_not_precompute(monkeypatch):
    monkeypatch.setattr(settings, "serving_read_only", True)
    _no_artifact(monkeypatch)

    def _must_not_read(*a, **k):
        raise AssertionError("the serving box read forecast rows to score on request")

    monkeypatch.setattr(inference.parquet_store, "read_dataset", _must_not_read)
    with pytest.raises(inference.CycleNotPrecomputed) as exc:
        inference.score_cycle(_State(), init_date="2017-12-01")
    assert "2017-12-01" in str(exc.value)


def test_off_the_box_scoring_still_falls_through(monkeypatch):
    """CI precompute depends on this path; the guard must not remove it."""
    monkeypatch.setattr(settings, "serving_read_only", False)
    _no_artifact(monkeypatch)
    calls = []

    def _empty(*a, **k):
        calls.append(k.get("value_types"))
        return pd.DataFrame()

    monkeypatch.setattr(inference.parquet_store, "read_dataset", _empty)
    assert inference.score_cycle(_State(), init_date="2017-12-01") is None
    assert calls, "score_cycle no longer reads the store when allowed to score"


def test_api_maps_a_missing_precompute_to_409(monkeypatch):
    from app.main import app
    from app.services import ensemble_service, replay_service

    def _raise(*a, **k):
        raise inference.CycleNotPrecomputed(pd.Timestamp("2017-12-01"))

    monkeypatch.setattr(replay_service, "get_replay", _raise)
    monkeypatch.setattr(ensemble_service, "get_divergence", _raise)
    client = TestClient(app)
    for url in ("/api/replay?init_date=2017-12-01", "/api/ensemble?init_date=2017-12-01"):
        r = client.get(url)
        assert r.status_code == 409, (url, r.status_code, r.text)
        assert "2017-12-01" in r.json()["detail"]


def test_prior_cycle_comparison_degrades_instead_of_failing(monkeypatch):
    """The ensemble hero compares with the previous cycle; for the oldest replay cycle that
    previous one is not precomputed. The comparison is optional - it must say so and let
    the main answer through."""
    from app.services import ensemble_service

    monkeypatch.setattr(inference, "available_cycles",
                        lambda: [pd.Timestamp("2017-11-30"), pd.Timestamp("2017-12-01")])

    def _raise(state, init_date=None):
        raise inference.CycleNotPrecomputed(pd.Timestamp(init_date))

    monkeypatch.setattr(inference, "score_cycle", _raise)
    mean, when, note = ensemble_service._prior_cycle_mean(_State(), pd.Timestamp("2017-12-01"))
    assert mean is None and when is None
    assert "not precomputed" in note


def test_replay_cycle_list_skips_a_cycle_without_a_precompute(monkeypatch):
    from app.services import replay_service

    def _raise(state, init):
        raise inference.CycleNotPrecomputed(pd.Timestamp(init))

    monkeypatch.setattr(inference, "score_cycle", _raise)
    assert replay_service._cycle_summary(_State(), pd.Timestamp("2017-12-01")) is None


def test_box_refuses_uploads(monkeypatch):
    from app.main import app

    monkeypatch.setattr(settings, "serving_read_only", True)
    client = TestClient(app)
    r = client.post("/api/upload", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 409, r.text
    r = client.post("/api/upload/some-batch/confirm-mapping", json={"mappings": []})
    assert r.status_code == 409, r.text
