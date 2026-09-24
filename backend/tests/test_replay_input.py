from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_replay_rejects_malformed_init_date_before_scoring():
    """A malformed query must be a 422, not an uncaught pandas parsing error after a
    potentially expensive cold replay ranking pass."""
    with TestClient(app) as client:
        response = client.get("/api/replay?init_date=not-a-date")
    assert response.status_code == 422
