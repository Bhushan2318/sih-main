"""Phase 4a: the FastAPI layer, exercised against real data and against a genuinely
empty environment.

The empty-environment tests are the important ones: they pin the plan's core rule that a
fresh install serves `model_trained: false` and EMPTY collections rather than any
placeholder number.
"""

from __future__ import annotations

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
