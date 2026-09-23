"""Score the cycle where memory is free, serve the answer where it is not.

CLAUDE.md's first diagram splits the system because "the serving box is killed, not
throttled" at 512 MB: CI trains, Render serves. Scoring every district of a cycle is
training-shaped work that had crept onto the serving side, and it only became load-bearing
when the live feed went from 36 city points to all 666 districts.

Measured on a real cycle, 666 districts x 10 lead days, 6,660 events:

  scoring it on the box          1,406 MB peak RSS
  reading the precomputed answer   105 MB total process RSS (38 MB growth)
  the artifact itself             3.96 MB zstd (2.56 events + 1.40 per-variable)

against a model tarball that is already 11.9 MB. So this is the same move as reading
forecast cycles from Parquet footers instead of scanning a column - +253 MB became +2 MB -
applied one level up.

Note `get_region_detail` also calls `score_latest_cycle`: precomputing only
/api/regions/all would have saved nothing, because one district click would still pay the
full cost. The artifact is the whole scored cycle for that reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml import precomputed
from app.ml.inference import ScoredCycle


def _scored(run_id="run_TEST", init="2018-12-31", n=120) -> ScoredCycle:
    """A ScoredCycle shaped like the real one, including the categorical columns.

    Synthetic and labelled as such - it exercises the round trip and the wiring. Nothing
    here produces a metric.
    """
    rng = np.random.default_rng(7)
    events = pd.DataFrame({
        "region_id": [f"IN-XX-D{i % 12:02d}" for i in range(n)],
        "lead_time_days": [(i % 10) + 1 for i in range(n)],
        "bust_probability": rng.uniform(0.02, 0.98, n),
        "risk_band": pd.Categorical(
            ["low" if i % 3 == 0 else "medium" if i % 3 == 1 else "high" for i in range(n)]),
        "dominant_variable": pd.Categorical(["rainfall_mm"] * n),
        "pred_err_temperature_c": rng.uniform(0.2, 4.0, n),
        "valid_date": pd.to_datetime("2018-12-31"),
    })
    per_variable = pd.DataFrame({
        "region_id": [f"IN-XX-D{i % 12:02d}" for i in range(n * 2)],
        "variable": pd.Categorical(["temperature_c", "rainfall_mm"] * n),
        "lead_time_days": [(i % 10) + 1 for i in range(n * 2)],
        "predicted_error": rng.uniform(0.1, 9.0, n * 2),
    })
    return ScoredCycle(run_id=run_id, init_date=pd.Timestamp(init),
                       events=events, per_variable=per_variable, n_rows_scored=n * 7)


def test_round_trips_every_column_and_dtype(tmp_path):
    """A band that comes back as a string instead of a category still renders, and a
    predicted error that comes back as a string does not. So compare dtypes, not just
    values."""
    original = _scored()
    precomputed.write_scored_cycle(original, tmp_path)
    got = precomputed.read_scored_cycle("run_TEST", pd.Timestamp("2018-12-31"), tmp_path)

    assert got is not None
    assert got.run_id == original.run_id
    assert got.init_date == original.init_date
    assert got.n_rows_scored == original.n_rows_scored
    pd.testing.assert_frame_equal(got.events, original.events, check_categorical=False)
    pd.testing.assert_frame_equal(got.per_variable, original.per_variable,
                                  check_categorical=False)
    assert got.events["bust_probability"].dtype.kind == "f"


def test_absent_returns_none_rather_than_raising(tmp_path):
    assert precomputed.read_scored_cycle("run_TEST", pd.Timestamp("2018-12-31"), tmp_path) is None


def test_a_different_run_does_not_match(tmp_path):
    """A new model must not serve the old model's answers.

    The artifact is keyed on (run_id, init_date) rather than on the store fingerprint,
    because `store_fingerprint` is built from file mtimes and cannot agree between the CI
    runner that writes the artifact and the box that reads it. The model half of that key
    is what stops a promotion from silently serving stale scores.
    """
    precomputed.write_scored_cycle(_scored(run_id="run_OLD"), tmp_path)
    assert precomputed.read_scored_cycle("run_NEW", pd.Timestamp("2018-12-31"), tmp_path) is None


def test_a_different_cycle_does_not_match(tmp_path):
    precomputed.write_scored_cycle(_scored(init="2018-12-31"), tmp_path)
    assert precomputed.read_scored_cycle("run_TEST", pd.Timestamp("2019-01-01"), tmp_path) is None


def test_a_half_written_artifact_is_ignored(tmp_path):
    """A run killed mid-write must not leave something that reads as a scored cycle.

    Refuse rather than patch: a partial answer served as a whole one is worse than no
    answer, because the map would simply be missing districts with nothing to say so.
    """
    precomputed.write_scored_cycle(_scored(), tmp_path)
    target = precomputed.cycle_dir(tmp_path, "run_TEST", pd.Timestamp("2018-12-31"))
    (target / "per_variable.parquet").unlink()
    assert precomputed.read_scored_cycle("run_TEST", pd.Timestamp("2018-12-31"), tmp_path) is None


def test_scoring_reads_the_artifact_instead_of_the_model(tmp_path, monkeypatch):
    """The whole point: with an artifact present, the expensive path must not run.

    Asserted by making the scoring path raise. A test that only checked the answer would
    pass whether or not the artifact was used, which is the false green this exists to
    avoid.
    """
    from app.ml import inference

    precomputed.write_scored_cycle(_scored(), tmp_path)
    monkeypatch.setattr(precomputed, "default_dir", lambda: tmp_path)
    monkeypatch.setattr(inference.fe, "build_training_frame",
                        lambda *a, **k: pytest.fail("scored on the box despite an artifact"))
    inference.invalidate_caches()

    state = type("S", (), {"run_id": "run_TEST"})()
    got = inference.score_cycle(state, init_date="2018-12-31")
    assert got is not None
    assert len(got.events) == 120
    assert got.run_id == "run_TEST"


def test_a_cycle_with_no_artifact_still_scores_normally(tmp_path, monkeypatch):
    """Replay asks for arbitrary historical cycles, which are not precomputed. Those must
    keep working rather than return nothing."""
    from app.ml import inference

    monkeypatch.setattr(precomputed, "default_dir", lambda: tmp_path)
    inference.invalidate_caches()
    called = {"n": 0}

    def fake_frame(*a, **k):
        called["n"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(inference.fe, "build_training_frame", fake_frame)
    monkeypatch.setattr(inference.parquet_store, "read_dataset",
                        lambda **k: pd.DataFrame({"region_id": ["x"], "variable": ["t"],
                                                  "value": [1.0], "value_type": ["forecast"],
                                                  "init_date": [pd.Timestamp("2018-12-31")],
                                                  "valid_date": [pd.Timestamp("2018-12-31")],
                                                  "lead_time_days": [1],
                                                  "ensemble_member_id": ["c00"],
                                                  "verification_status": [None]}))
    monkeypatch.setattr(inference, "forecast_history", lambda *a, **k: None)

    state = type("S", (), {"run_id": "run_TEST", "historical_bust_freq": {},
                           "jump_climatology": {}})()
    assert inference.score_cycle(state, init_date="2017-06-01") is None
    assert called["n"] == 1, "fell through to the scoring path exactly once"
