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


# ------------------------- on the box, only what the dashboard asks for, and nothing that builds
#
# Each request below loaded the model and built live on a box-mode mirror of the live bundle
# (2026-09-28, fresh server each): +95 to +175 MB apiece, on a box killed at 512 that idles
# near 240. The dashboard sends none of them. Builders that are reached fail the test.

def _on_the_box(monkeypatch, tmp_path):
    from app.ml import registry
    from app.services import response_cache

    monkeypatch.setattr(settings, "serving_read_only", True)
    monkeypatch.setattr(registry, "current_run_id", lambda: "run_PLUMBING")
    monkeypatch.setattr(response_cache, "default_dir", lambda: tmp_path)
    response_cache.invalidate()


def _must_not_build(*a, **k):
    raise AssertionError("built on request although the dashboard never asks for it")


def test_box_refuses_parameters_the_dashboard_never_sends(monkeypatch, tmp_path):
    from app.main import app
    from app.services import ensemble_service, replay_service

    _on_the_box(monkeypatch, tmp_path)
    monkeypatch.setattr(ensemble_service, "get_divergence", _must_not_build)
    monkeypatch.setattr(replay_service, "get_replay", _must_not_build)
    client = TestClient(app)
    for url in ("/api/ensemble?region_id=zzz", "/api/ensemble?init_date=2017-12-01",
                "/api/replay?focus_region=zzz"):
        r = client.get(url)
        assert r.status_code == 409, (url, r.status_code, r.text)


def test_box_refuses_a_state_panel(monkeypatch, tmp_path):
    """State ids pass the unknown-region check, but CI builds district panels only and the
    dashboard never opens a state's (a click on a state drills the map into it)."""
    from app.main import app
    from app.services import region_service

    _on_the_box(monkeypatch, tmp_path)
    monkeypatch.setattr(region_service, "get_region_detail", _must_not_build)
    r = TestClient(app).get("/api/regions/IN-MH")
    assert r.status_code == 404, r.text


def test_box_serves_one_lead_day_as_a_slice_of_the_all_days_map(monkeypatch, tmp_path):
    """The payload is a plumbing stand-in shaped like AllRegionsResponse."""
    from app.main import app
    from app.services import region_service

    _on_the_box(monkeypatch, tmp_path)
    monkeypatch.setattr(region_service, "get_regions", _must_not_build)
    monkeypatch.setattr(region_service, "get_all_regions", lambda: {
        "model_trained": True,
        "days": [{"lead_time_days": d, "regions": [{"region_id": f"R{d}"}]} for d in range(1, 11)],
    })
    r = TestClient(app).get("/api/regions?lead_time_days=3")
    assert r.status_code == 200, r.text
    assert r.json() == {"lead_time_days": 3, "regions": [{"region_id": "R3"}]}


@pytest.mark.parametrize("box", [True, False])
def test_a_malformed_replay_date_is_a_422_not_a_500(monkeypatch, tmp_path, box):
    from app.main import app
    from app.services import replay_service

    _on_the_box(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "serving_read_only", box)
    monkeypatch.setattr(replay_service, "get_replay", _must_not_build)
    r = TestClient(app).get("/api/replay?init_date=garbage")
    assert r.status_code == 422, r.text


def test_a_path_with_a_null_byte_is_a_404_not_a_500(tmp_path):
    """`GET /%00` raised ValueError (embedded null byte) out of the SPA fallback."""
    from app import main

    (tmp_path / "index.html").write_text("<!doctype html>")
    assert main._static_file(tmp_path, "\x00") is None
    assert main._static_file(tmp_path, "a\x00b.js") is None
    assert main._static_file(tmp_path, "index.html") == tmp_path / "index.html"
    assert main._static_file(tmp_path, "../etc/passwd") is None
