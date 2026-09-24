from __future__ import annotations

from datetime import date

import pandas as pd

from app.live import orchestrator
from app.live.observations import ObsReport
from app.storage import parquet_store


def test_final_observation_refresh_rejects_partial_city_coverage(fresh_store, monkeypatch):
    frame = pd.DataFrame([{
        "region_id": "IN-MH",
        "date": date(2026, 1, 1),
        "t2m_c": 20.0,
    }])
    report = ObsReport(
        tier="final",
        start=date(2026, 1, 1),
        end=date(2026, 1, 2),
        cities=2,
        rows=1,
        failures=["IN-KL"],
    )

    monkeypatch.setattr(
        orchestrator.observations,
        "fetch_observations",
        lambda start, end, tier: (frame, report),
    )

    result = orchestrator.run_observation_refresh("final", days_back=1, retrain=True)

    assert result["status"] == "failed"
    assert "IN-KL" in result["error"]
    assert parquet_store.read_dataset().empty
