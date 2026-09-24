from __future__ import annotations

from datetime import date

import pandas as pd

from app.features import engineering as fe
from app.features import history as history_features
from app.storage import parquet_store


def _row(batch: str, hour: int, value: float) -> dict:
    return {
        "record_id": f"{batch}-{hour}",
        "upload_batch_id": batch,
        "source_file": f"gefs_operational_forecast_20260829_{hour:02d}z.csv",
        "source_column": "t2m_c",
        "variable": "temperature_c",
        "value_type": "forecast",
        "value": value,
        "region_id": "IN-MH-NAGPUR",
        "region_name": "Nagpur",
        "lat": 21.15,
        "lon": 79.09,
        "init_date": date(2026, 8, 29),
        "valid_date": date(2026, 8, 29),
        "lead_time_days": 1,
        "ensemble_member_id": "c00",
        "mapping_confidence": 1.0,
        "ingested_at": pd.Timestamp("2026-08-29T06:00:00"),
        "grain": "native",
        "region_resolution_method": "name",
        "verification_status": None,
        "init_cycle": pd.Timestamp(f"2026-08-29 {hour:02d}:00:00").to_pydatetime(),
        "cycle_hour": hour,
    }


def test_same_date_different_cycles_are_not_deduplicated(fresh_store):
    parquet_store.append_batch("cycle-00", pd.DataFrame([_row("cycle-00", 0, 20.0)]))
    parquet_store.append_batch("cycle-06", pd.DataFrame([_row("cycle-06", 6, 21.0)]))

    got = parquet_store.read_dataset(
        value_types=["forecast"], init_dates=[date(2026, 8, 29)], dedupe=True
    )
    assert len(got) == 2
    assert set(got["cycle_hour"].astype(int)) == {0, 6}
    assert set(got["init_cycle"].dt.hour) == {0, 6}


def test_same_date_cycles_remain_separate_trajectories_and_jump_history():
    rows = []
    for hour, value in ((0, 10.0), (6, 16.0), (12, 13.0)):
        rows.append({
            "region_id": "IN-MH-NAGPUR",
            "cycle_hour": hour,
            "variable": "temperature_c",
            "value_type": "forecast",
            "valid_date": pd.Timestamp("2026-08-30"),
            "init_date": pd.Timestamp("2026-08-29"),
            "lead_time_days": 2,
            "ensemble_member_id": "c00",
            "value": value,
        })

    trajectories = fe.forecast_trajectories(pd.DataFrame(rows))
    assert len(trajectories) == 3
    assert set(trajectories["cycle_hour"]) == {0, 6, 12}

    jumps = fe.compute_jumpiness(trajectories).set_index("cycle_hour")
    assert pd.isna(jumps.loc[0, "jump_abs_change"])
    assert jumps.loc[6, "jump_abs_change"] == 6
    assert jumps.loc[12, "jump_abs_change"] == 3


def test_history_uses_an_exact_cycle_cutoff(monkeypatch):
    calls = []

    def fake_read(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(history_features.parquet_store, "read_dataset", fake_read)
    history_features.forecast_history(
        pd.Timestamp("2026-08-29"),
        [(pd.Timestamp("2026-08-28").date(), 12),
         (pd.Timestamp("2026-08-29").date(), 0),
         (pd.Timestamp("2026-08-29").date(), 6)],
        first_cycle_hour=6,
    )

    assert [(c["init_dates"][0], c.get("cycle_hours")) for c in calls] == [
        (pd.Timestamp("2026-08-28").date(), [12]),
        (pd.Timestamp("2026-08-29").date(), [0]),
    ]
