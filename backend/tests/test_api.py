"""Phase 4a: the FastAPI layer, exercised against real data and against a genuinely
empty environment.

The empty-environment tests are the important ones: they pin the plan's core rule that a
fresh install serves `model_trained: false` and EMPTY collections rather than any
placeholder number.
"""

from __future__ import annotations

import json
import shutil

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tests.conftest import ERA5_CSV, GEFS_CSV


def _wipe_everything():
    """Blank DB, empty canonical store, and no trained model."""
    from app.db.base import engine, init_db
    from app.db.models import Base
    from app.ml import inference, registry
    from app.storage import parquet_store

    Base.metadata.drop_all(engine)
    shutil.rmtree(parquet_store.CANONICAL_DIR, ignore_errors=True)
    shutil.rmtree(registry.MODEL_DIR, ignore_errors=True)
    inference.invalidate_caches()
    init_db()


@pytest.fixture
def blank_client():
    _wipe_everything()
    from app.main import app

    with TestClient(app) as c:
        yield c
    _wipe_everything()


# ------------------------------------------------------------------ empty environment

def test_health_on_blank_install(blank_client):
    body = blank_client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_trained"] is False
    assert body["current_run_id"] is None


def test_health_reports_its_own_memory(blank_client):
    """The deployed box dies at 512 MB rather than degrading, and Render's free tier puts
    the memory dashboard behind a paywall - so the process reports its own RSS and the
    number can be read from outside at any time.

    The key is always present. The value is null where /proc is unavailable (a macOS dev
    machine), because a guessed number would be worse than none; on Linux, which is what
    the container runs, it must be a real reading.
    """
    import sys

    body = blank_client.get("/api/health").json()
    assert "memory_mb" in body
    mem = body["memory_mb"]
    assert mem is None or (isinstance(mem, (int, float)) and mem > 0)
    if sys.platform.startswith("linux"):
        assert mem is not None and mem > 0, "/proc/self/status should be readable on Linux"


def test_health_reports_how_long_the_process_has_been_up(blank_client):
    """Render's free tier stops the box after 15 idle minutes, and an out-of-memory restart
    looks the same from outside: one slow page. Uptime tells a fresh process from a slow
    endpoint with a single request."""
    first = blank_client.get("/api/health").json()["uptime_s"]
    assert isinstance(first, (int, float)) and first >= 0
    assert blank_client.get("/api/health").json()["uptime_s"] >= first


def test_health_says_whether_prebuilt_responses_are_being_served(blank_client):
    """warm-on-push reads this to decide whether warming is safe: a box without a matching
    bundle builds every screen live, and warming one is what OOM-killed it on 2026-09-26
    20:03Z and 2026-09-27 13:03Z. The key is always present; a blank install serves none."""
    p = blank_client.get("/api/health").json()["prebuilt"]
    assert p["active"] is False
    assert p["reason"]


def test_health_always_reports_a_build_commit_key(blank_client, monkeypatch):
    # CI polls this to tell whether the process answering is the one it just deployed -
    # Render auto-deploys on push, which never touches the refresh workflow, so there is
    # no other way to know before warming the caches. The key must always be present:
    # a caller can fall back on an empty string, but a missing key is an exception.
    from app import main

    monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
    monkeypatch.delenv("BUILD_COMMIT", raising=False)
    body = blank_client.get("/api/health").json()
    assert body["commit"] == ""

    monkeypatch.setenv("RENDER_GIT_COMMIT", "d2d0486cafe")
    assert blank_client.get("/api/health").json()["commit"] == "d2d0486cafe"

    # Render's variable wins; BUILD_COMMIT is only the escape hatch off Render.
    monkeypatch.setenv("BUILD_COMMIT", "ignored")
    assert blank_client.get("/api/health").json()["commit"] == "d2d0486cafe"
    monkeypatch.delenv("RENDER_GIT_COMMIT")
    assert blank_client.get("/api/health").json()["commit"] == "ignored"
    assert main._build_commit() == "ignored"


def test_regions_empty_state_has_no_placeholder_numbers(blank_client):
    body = blank_client.get("/api/regions?lead_time_days=1").json()
    assert body["model_trained"] is False
    assert body["regions"] == []
    assert body["current_run_id"] is None
    assert body["message"]                      # explains itself to the user
    assert body["risk_band_definitions"] == {}


def test_region_detail_empty_state(blank_client):
    body = blank_client.get("/api/regions/IN-MH").json()
    assert body["model_trained"] is False
    assert body["variables"] == []
    assert body["bust_probability_curve"] == []
    assert body["top_factors"] == []
    assert body["message"]


def test_alerts_empty_state(blank_client):
    body = blank_client.get("/api/alerts").json()
    assert body["model_trained"] is False
    assert body["alerts"] == []
    assert body["message"]


def test_model_status_empty_state(blank_client):
    body = blank_client.get("/api/model/status").json()
    assert body["model_trained"] is False
    assert body["current_run_id"] is None
    assert body["validation_metrics"] == {}
    assert body["data_volume"]["total_rows"] == 0
    assert body["message"]


def test_unknown_region_404(blank_client):
    assert blank_client.get("/api/regions/IN-ZZ").status_code == 404


def test_lead_time_out_of_range_422(blank_client):
    assert blank_client.get("/api/regions?lead_time_days=0").status_code == 422
    assert blank_client.get("/api/regions?lead_time_days=11").status_code == 422


# --------------------------------------------------------------------- upload flow

def test_upload_rejects_empty_file(blank_client):
    r = blank_client.post("/api/upload", files={"file": ("empty.csv", b"", "text/csv")})
    assert r.status_code == 400


def test_upload_rejects_unparseable_json(blank_client):
    r = blank_client.post(
        "/api/upload",
        files={"file": ("shape.json", b'{"type":"FeatureCollection","features":[]}', "application/json")},
    )
    assert r.status_code == 422
    assert "parse" in r.json()["detail"].lower()


def test_upload_returns_mapping_proposals(blank_client, tmp_path):
    """A real slice of the ERA5 sample: ambiguous columns must come back for confirmation
    rather than being silently guessed."""
    df = pd.read_csv(ERA5_CSV, nrows=400)
    p = tmp_path / "era5_slice.csv"
    df.to_csv(p, index=False)

    r = blank_client.post("/api/upload", files={"file": (p.name, p.read_bytes(), "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending_confirmation"
    assert body["row_count_raw"] == 400
    assert body["detected_format"] == "csv"
    proposals = {p_["source_column"]: p_ for p_ in body["mapping_proposals"]}
    assert proposals["t2m_c"]["suggested_variable"] == "temperature_c"
    # mslp/psfc both claim pressure -> collision -> confirmation, never a silent pick
    assert proposals["mslp_hpa"]["decision"] == "needs_confirmation"


def test_confirm_mapping_ingests_and_triggers_training(blank_client, tmp_path):
    df = pd.read_csv(ERA5_CSV, nrows=400)
    p = tmp_path / "era5_slice.csv"
    df.to_csv(p, index=False)
    up = blank_client.post("/api/upload", files={"file": (p.name, p.read_bytes(), "text/csv")}).json()

    seen, mappings = set(), []
    for prop in up["mapping_proposals"]:
        if prop["role"] != "measurement" or prop["decision"] != "needs_confirmation":
            continue
        if not prop["suggested_variable"]:
            continue
        key = (prop["suggested_variable"], prop["suggested_value_type"])
        if key in seen:
            continue
        seen.add(key)
        mappings.append({
            "source_column": prop["source_column"],
            "variable": prop["suggested_variable"],
            "value_type": prop["suggested_value_type"],
            "unit_conversion": prop["unit_conversion"],
        })

    r = blank_client.post(f"/api/upload/{up['batch_id']}/confirm-mapping", json={"mappings": mappings})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "training_started"
    assert body["row_count_ingested"] > 0
    assert body["canonical_variables_found"]

    # observations alone cannot train anything - status must stay honest about that
    status = blank_client.get("/api/model/status").json()
    assert status["data_volume"]["total_rows"] > 0


def test_confirm_mapping_unknown_batch_404(blank_client):
    r = blank_client.post("/api/upload/does-not-exist/confirm-mapping", json={"mappings": []})
    assert r.status_code == 404


# ------------------------------------------------------------------------- websocket

def test_websocket_sends_connected_frame(blank_client):
    with blank_client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert "event" in msg and "timestamp" in msg and "payload" in msg
        # the opening frame must NOT masquerade as a state change (e.g. training_complete)
        assert msg["event"] == "connected"
        assert msg["payload"]["model_trained"] is False
        assert msg["payload"]["current_run_id"] is None


# ------------------------------------------------------- trained model (real data)

@pytest.fixture(scope="module")
def trained_client():
    """Ingest a real multi-cycle slice, train for real, then serve it."""
    from app.db.base import SessionLocal
    from app.ingestion.pipeline import confirm_mapping, ingest_upload
    from app.main import app
    from app.ml import inference
    from app.ml.train_pipeline import full_retrain

    _wipe_everything()
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp(prefix="fg-api-"))
    g = pd.read_csv(GEFS_CSV)
    g = g[g["init_date"].isin(sorted(g["init_date"].unique())[:8])]
    gp = tmp / "gefs.csv"; g.to_csv(gp, index=False)
    ep = tmp / "era5.csv"; pd.read_csv(ERA5_CSV).to_csv(ep, index=False)

    def _confirm(res, s):
        if res.status != "pending_confirmation":
            return res
        seen, conf = set(), []
        for p in sorted(res.mapping_proposals, key=lambda x: x["source_column"]):
            if p["role"] != "measurement" or p["decision"] != "needs_confirmation":
                continue
            if not p["suggested_variable"]:
                continue
            key = (p["suggested_variable"], p["suggested_value_type"])
            if key in seen:
                continue
            seen.add(key)
            conf.append({"source_column": p["source_column"], "variable": p["suggested_variable"],
                         "value_type": p["suggested_value_type"],
                         "unit_conversion": p["unit_conversion"]})
        return confirm_mapping(s, res.batch_id, conf)

    s = SessionLocal()
    try:
        _confirm(ingest_upload(s, gp, gp.name), s)
        _confirm(ingest_upload(s, ep, ep.name), s)
        s.commit()
    finally:
        s.close()

    report = full_retrain(make_current=True)
    assert report.status == "success", report.error
    inference.invalidate_caches()

    with TestClient(app) as c:
        yield c
    shutil.rmtree(tmp, ignore_errors=True)
    _wipe_everything()


def test_regions_served_from_real_model(trained_client):
    body = trained_client.get("/api/regions?lead_time_days=3").json()
    assert body["model_trained"] is True
    assert body["current_run_id"]
    assert body["init_date"]
    assert len(body["regions"]) > 5
    for r in body["regions"]:
        assert 0.0 <= r["bust_probability"] <= 1.0
        assert r["risk_band"] in {"low", "medium", "high"}
        assert r["region_id"].startswith("IN-")
    # sorted riskiest-first
    probs = [r["bust_probability"] for r in body["regions"]]
    assert probs == sorted(probs, reverse=True)
    assert body["risk_band_definitions"]["basis"]


def test_regions_all_matches_per_day_and_covers_1_to_10(trained_client):
    """The all-days endpoint must return exactly what the per-day endpoint does for each
    lead day - it is the dashboard's only regions call, so any drift is a visible bug."""
    all_body = trained_client.get("/api/regions/all").json()
    assert all_body["model_trained"] is True
    days = all_body["days"]
    assert [d["lead_time_days"] for d in days] == list(range(1, 11))

    for d in days:
        single = trained_client.get(f"/api/regions?lead_time_days={d['lead_time_days']}").json()
        assert [r["region_id"] for r in d["regions"]] == [r["region_id"] for r in single["regions"]]
        assert [r["bust_probability"] for r in d["regions"]] == [
            r["bust_probability"] for r in single["regions"]
        ]
        assert d["valid_date"] == single["valid_date"]

    # 'all' must not be captured by the /{region_id} route
    assert trained_client.get("/api/regions/all").status_code == 200


def test_region_detail_served_from_real_model(trained_client):
    regions = trained_client.get("/api/regions?lead_time_days=1").json()["regions"]
    rid = regions[0]["region_id"]
    body = trained_client.get(f"/api/regions/{rid}").json()

    assert body["model_trained"] is True
    assert body["variables"]
    curve = body["bust_probability_curve"]
    assert curve
    assert [p["lead_time_days"] for p in curve] == sorted(p["lead_time_days"] for p in curve)

    available = [v for v in body["variables"] if v["available"]]
    assert available
    for v in available:
        assert v["unit"]
        assert v["bust_threshold"] is not None
        for pt in v["points"]:
            assert pt["predicted_value"] is not None
            assert 0.0 <= (pt["confidence"] or 0) <= 1.0

    # an unavailable variable is flagged, not fabricated
    for v in body["variables"]:
        if not v["available"]:
            assert v["points"] == []

    if body["top_factors"]:
        assert body["top_factors_method"] in {"shap", "feature_importance_fallback"}
    # analog cases are not implemented -> empty, never invented
    assert body["analog_cases"] == []


def test_alerts_served_from_real_model(trained_client):
    body = trained_client.get("/api/alerts?limit=10").json()
    assert body["model_trained"] is True
    assert len(body["alerts"]) <= 10
    for a in body["alerts"]:
        assert a["risk_band"] in {"medium", "high"}
        assert a["training_run_id"]
    probs = [a["bust_probability"] for a in body["alerts"]]
    assert probs == sorted(probs, reverse=True)

    high = trained_client.get("/api/alerts?risk_band=high").json()["alerts"]
    assert all(a["risk_band"] == "high" for a in high)


def test_model_status_reports_real_metrics(trained_client):
    body = trained_client.get("/api/model/status").json()
    assert body["model_trained"] is True
    assert body["modelled_variables"]
    assert body["explanation_method"] in {"shap", "feature_importance_fallback", "none"}

    clf = body["validation_metrics"]["classifier"]
    assert clf["split"] in {"test", "val", "train"}
    assert 0.0 <= clf["roc_auc"] <= 1.0

    for var, thr in body["thresholds"]["bust_threshold"].items():
        assert thr > 0
    cuts = body["thresholds"]["risk_band_cuts"]
    assert cuts["medium"] <= cuts["high"]


def test_map_and_alerts_agree(trained_client):
    """Alerts are derived from the same scored cycle, so they cannot contradict the map."""
    alerts = trained_client.get("/api/alerts?limit=200").json()["alerts"]
    if not alerts:
        pytest.skip("no medium/high alerts in this cycle")
    a = alerts[0]
    regions = trained_client.get(f"/api/regions?lead_time_days={a['lead_time_days']}").json()["regions"]
    match = next(r for r in regions if r["region_id"] == a["region_id"])
    assert match["bust_probability"] == pytest.approx(a["bust_probability"], abs=1e-9)
    assert match["risk_band"] == a["risk_band"]


def test_ensemble_divergence_draws_real_members(trained_client):
    """The hero view must come from the five real GEFS members in the store, not from a
    band inferred off a standard deviation - and it must refuse to invent a prior-cycle
    comparison when the only earlier cycle is years away in the reforecast archive."""
    body = trained_client.get("/api/ensemble").json()
    assert body["model_trained"] is True
    if body.get("message"):
        pytest.skip(f"no scoreable cycle in this slice: {body['message']}")

    # 0 and 360 degrees are the same bearing, so member traces around north look like a
    # collapse that never happened - the hero must never headline with it.
    assert body["variable"] and body["variable"] != "wind_direction_deg"

    assert body["members"], "no ensemble members returned"
    assert sum(1 for m in body["members"] if m["is_control"]) == 1
    for m in body["members"]:
        assert m["points"], f"member {m['member_id']} has no points"
        assert all(p["value"] is not None for p in m["points"])

    # the mean is the model's own aggregate, so it must span the members' lead days
    mean_leads = {p["lead_time_days"] for p in body["ensemble_mean"]}
    assert mean_leads
    for m in body["members"]:
        assert {p["lead_time_days"] for p in m["points"]} <= mean_leads

    assert body["n_scored_regions"] > 0
    assert 0.0 <= body["mean_bust_probability"] <= 1.0
    # no fabricated comparison: either a real prior cycle, or a stated reason for none
    assert (body["prior_mean_bust_probability"] is None) != (body["prior_note"] is None)
    assert body["headline_note"]


def test_ensemble_pins_to_the_requested_region(trained_client):
    """Clicking a region must chart that region. If the pin silently fell back to the
    auto-picked subject the hero would show one place while the UI claimed another."""
    auto = trained_client.get("/api/ensemble").json()
    if auto.get("message"):
        pytest.skip(f"no scoreable cycle in this slice: {auto['message']}")

    regions = trained_client.get("/api/regions?lead_time_days=2").json()["regions"]
    other = next(
        (r["region_id"] for r in regions
         if r["region_id"] != auto["region_id"] and r["bust_probability"] is not None),
        None,
    )
    if other is None:
        pytest.skip("only one scored region in this slice")

    pinned = trained_client.get(f"/api/ensemble?region_id={other}").json()
    assert pinned["region_id"] == other
    assert pinned["members"], "a pinned region must still carry real member traces"
    # the scope claim has to follow the pin rather than overstating it
    assert "for this region" in (pinned["subject_reason"] or "")
    assert "in this cycle" in (auto["subject_reason"] or "")


# ------------------------------------------- precomputed responses: built in CI, read on the box

# Stamped with the time they were built, so they legitimately differ between two builds.
_BUILD_TIME_FIELDS = {"generated_at", "created_at"}


def _without_build_times(o):
    if isinstance(o, dict):
        return {k: _without_build_times(v) for k, v in o.items() if k not in _BUILD_TIME_FIELDS}
    if isinstance(o, list):
        return [_without_build_times(v) for v in o]
    return o


def test_precomputed_responses_match_what_the_box_would_build(trained_client, tmp_path,
                                                              monkeypatch):
    """Every screen the dashboard opens, served from what CI built, must be exactly what
    the serving box would have built itself - and with the files present the box must
    build nothing at all.

    "What the box would build" is taken in box mode (SERVING_READ_ONLY, scored cycles
    precomputed), because that is the comparison that matters: the box already serves
    precomputed *scores*, and this only moves the step after them.

    The no-build half is asserted by making every builder raise. A test that only compared
    bodies would pass whether or not the files were used.
    """
    from app.api.routers import model_status as ms
    from app.config import settings
    from app.ml import inference, precomputed, registry
    from app.services import (alert_service, ensemble_service, region_service,
                              replay_service, response_cache)
    from scripts import package_for_deploy

    monkeypatch.setattr(precomputed, "default_dir", lambda: tmp_path / "scored")
    monkeypatch.setattr(response_cache, "default_dir", lambda: tmp_path / "responses")
    inference.invalidate_caches()
    state = inference.load_model_state()
    written, _ = package_for_deploy.precompute_cycles(state)
    assert written > 0

    monkeypatch.setattr(settings, "serving_read_only", True)
    inference.invalidate_caches()
    cycles = trained_client.get("/api/replay/cycles").json()
    assert cycles, "the trained slice should offer at least one Replay cycle"
    rid = trained_client.get("/api/regions?lead_time_days=1").json()["regions"][0]["region_id"]
    paths = [
        "/api/ensemble", "/api/regions/all", "/api/model/status", "/api/replay/cycles",
        "/api/replay", f"/api/regions/{rid}",
        "/api/alerts?limit=200", "/api/alerts?limit=200&risk_band=high",
        "/api/alerts?limit=3&risk_band=medium", "/api/alerts",
    ] + [f"/api/replay?init_date={c['init_date']}" for c in cycles]
    built_on_the_box = {p: trained_client.get(p).json() for p in paths}

    report = response_cache.precompute(registry.current_run_id())
    assert not report["skipped"], report["skipped"]
    assert settings.serving_read_only is True
    inference.invalidate_caches()  # drops the box's in-memory copies too

    def _must_not_build(*_a, **_k):
        raise AssertionError("built on request although CI had precomputed it")

    for mod, fn in [(ensemble_service, "get_divergence"), (region_service, "get_all_regions"),
                    (region_service, "get_region_detail"), (replay_service, "get_replay"),
                    (replay_service, "list_cycles"), (alert_service, "get_alerts"),
                    (ms, "build_status")]:
        monkeypatch.setattr(mod, fn, _must_not_build)

    for p in paths:
        r = trained_client.get(p)
        assert r.status_code == 200, p
        assert _without_build_times(r.json()) == _without_build_times(built_on_the_box[p]), p


def test_the_serving_box_loads_the_models_names_not_the_models(trained_client, monkeypatch):
    """The box never scores, so it never needs the XGBoost models: measured +73 MB for the
    eight regressors with xgboost, on a box killed at 512. It needs their names (the panel
    and model status list the modelled variables) and the classifier's SHAP rows."""
    from app.config import settings
    from app.ml import inference

    inference.invalidate_caches()
    full = inference.load_model_state()
    monkeypatch.setattr(settings, "serving_read_only", True)
    inference.invalidate_caches()
    box = inference.load_model_state()

    assert box.regressors == {} and box.classifier is None
    assert box.variables == full.variables and box.variables
    assert box.thresholds.risk_band_cuts == full.thresholds.risk_band_cuts
    assert box.metrics == full.metrics and box.manifest == full.manifest
    want = full.shap_summary[full.shap_summary["model"] == "classifier"]
    assert len(box.shap_summary) == len(want) > 0
    assert set(box.shap_summary["model"].astype(str)) == {"classifier"}
    inference.invalidate_caches()


def test_a_state_loaded_without_its_models_refuses_to_score(trained_client, monkeypatch):
    """Scoring with no models would return no rows - an empty map that reads as "nothing to
    score". It must fail loudly instead (refuse rather than patch)."""
    from app.config import settings
    from app.ml import inference

    monkeypatch.setattr(settings, "serving_read_only", True)
    inference.invalidate_caches()
    box = inference.load_model_state()
    monkeypatch.setattr(settings, "serving_read_only", False)
    monkeypatch.setattr(inference.precomputed, "read_scored_cycle", lambda *a, **k: None)
    inference._score_cache.clear()
    with pytest.raises(RuntimeError, match="without its models"):
        inference.score_cycle(box)
    inference.invalidate_caches()


def test_the_box_builds_every_screen_exactly_as_the_full_model_would(trained_client, tmp_path,
                                                                     monkeypatch):
    """The lean load changes what is held, never what is served: every screen the dashboard
    asks for, built live on the box, is identical to the same build with the models loaded."""
    from app.config import settings
    from app.ml import inference, precomputed
    from app.services import response_cache
    from scripts import package_for_deploy

    monkeypatch.setattr(precomputed, "default_dir", lambda: tmp_path / "scored")
    monkeypatch.setattr(response_cache, "default_dir", lambda: tmp_path / "no-prebuilt")
    inference.invalidate_caches()
    assert package_for_deploy.precompute_cycles(inference.load_model_state())[0] > 0

    def build_all():
        inference.invalidate_caches()
        return {name: json.loads(response_cache._to_json(build))
                for name, build in ((n, b()) for n, b in response_cache.catalogue())}

    monkeypatch.setattr(settings, "serving_read_only", True)
    monkeypatch.setattr(inference, "_models_stay_off_this_box", lambda: False)
    with_models = build_all()
    monkeypatch.setattr(inference, "_models_stay_off_this_box", lambda: True)
    lean = build_all()
    assert inference.load_model_state().classifier is None, "the lean path did not run"

    assert set(lean) == set(with_models) and any(n.startswith("region__") for n in lean)
    for name in with_models:
        assert _without_build_times(lean[name]) == _without_build_times(with_models[name]), name
    inference.invalidate_caches()


def test_loading_the_model_on_the_box_imports_neither_xgboost_nor_shap(trained_client):
    """Checked in a fresh interpreter: an import made anywhere earlier in this test process
    would hide one made by the loader."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    code = ("import sys; from app.ml import inference; s = inference.load_model_state(); "
            "print(s is not None, 'xgboost' in sys.modules, 'shap' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[1], timeout=300,
                         env={**os.environ, "SERVING_READ_ONLY": "true"})
    assert out.stdout.strip().splitlines()[-1] == "True False False", out.stderr[-2000:]


def test_precomputed_responses_are_sent_compressed_only_when_asked(trained_client):
    """Sent gzipped as stored when the client accepts it (every browser does), and inflated
    for one that does not, so no client ever receives bytes it cannot read."""
    gz = trained_client.get("/api/regions/all", headers={"Accept-Encoding": "gzip"})
    plain = trained_client.get("/api/regions/all", headers={"Accept-Encoding": "identity"})
    assert gz.status_code == plain.status_code == 200
    assert gz.headers.get("content-encoding") == "gzip"
    assert "content-encoding" not in plain.headers
    assert gz.json() == plain.json()


def test_pipeline_log_shows_refused_runs_not_only_successes(blank_client):
    """The activity log lists every pipeline attempt, including the ones that ingested
    nothing.

    A cycle NOAA serves too incompletely to trust is refused rather than published - a
    short rainfall *sum* is roughly half the real accumulation, and rainfall drives most
    busts - so a refusal is the guard working, and hiding it would be the same convenient
    fiction this project exists to avoid. Asserted explicitly because "show only what
    succeeded" is the easy default and nothing else would catch it.
    """
    from datetime import datetime, timedelta

    from app.db.base import get_session
    from app.db.models import IngestRun

    base = datetime(2026, 9, 1, 0, 0, 0)
    made = [
        ("forecast", "2026-09-01 00", "complete", 12780, None),
        ("forecast", "2026-09-01 06", "failed", 0, "steps 340/400: refused as incomplete"),
        ("forecast", "2026-09-01 12", "skipped", 0, None),
    ]
    with get_session() as session:
        for i, (kind, target, status, rows, err) in enumerate(made):
            session.add(IngestRun(
                kind=kind, target=target, status=status, trigger="schedule",
                rows_ingested=rows, error=err,
                started_at=base + timedelta(hours=i),
                finished_at=base + timedelta(hours=i, minutes=2),
            ))
        session.commit()

    body = blank_client.get("/api/ingest/runs?limit=50").json()
    got = {r["target"]: r for r in body["runs"]}

    for _, target, status, _, _ in made:
        assert target in got, f"{status} run is missing from the log"
        assert got[target]["status"] == status

    # The refusal keeps its reason - a red row with no explanation is not honest, it is
    # just alarming.
    assert "refused as incomplete" in got["2026-09-01 06"]["error"]
    # Newest first, and durations are derived rather than stored.
    assert [r["target"] for r in body["runs"]][:3] == [
        "2026-09-01 12", "2026-09-01 06", "2026-09-01 00"]
    assert got["2026-09-01 00"]["seconds"] == 120.0
