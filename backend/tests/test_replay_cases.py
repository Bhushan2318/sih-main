"""Past events in Replay: the catalogue, its sample labels, and as-of-init scoring.

The end-to-end cases - scoring a real cycle as of its init, refusing an incomplete one,
writing a case and serving it back - run on the real GEFS/ERA5 slice in test_ml.py, whose
module-scoped retrain they need. What is here needs no model: the rules a case must pass
before it can be built at all.

The model states below are plumbing stand-ins (they carry only a run_id); nothing here
produces a metric.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from app.config import settings
from app.ml import inference
from app.services import replay_cases

# The split recorded in run_20260922T043925Z's manifest.json, copied as it is written
# there: pooled_train_years 2000-2016, test_year 2017.
_REAL_MANIFEST = {"pooled_train_years": list(range(2000, 2017)), "test_year": 2017}


def test_a_training_year_case_is_refused():
    """A replay of a cycle the model trained on is a recital, not evidence."""
    with pytest.raises(replay_cases.CaseRefused, match="2015"):
        replay_cases.sample_note(2015, _REAL_MANIFEST)


def test_the_test_year_is_labelled_held_out():
    note = replay_cases.sample_note(2017, _REAL_MANIFEST)
    assert "held-out test year" in note.lower()
    assert "2017" in note


def test_a_year_after_the_split_is_labelled_never_used():
    note = replay_cases.sample_note(2018, _REAL_MANIFEST)
    assert "never used" in note.lower()
    assert "2018" in note


def test_a_manifest_without_its_split_cannot_label_a_case():
    """Refuse rather than guess: an unlabelled case cannot say whether it is evidence."""
    with pytest.raises(replay_cases.CaseRefused):
        replay_cases.sample_note(2018, {})


def test_the_split_is_read_from_split_cycles_when_the_top_level_keys_are_absent():
    manifest = {"split_cycles": {"train_years": [2000, 2001], "test_year": 2002}}
    assert "held-out" in replay_cases.sample_note(2002, manifest).lower()
    with pytest.raises(replay_cases.CaseRefused):
        replay_cases.sample_note(2001, manifest)


def test_every_catalogued_case_is_out_of_sample_for_the_served_run():
    """The four shipped events, against the served run's own split. A catalogue edit that
    slipped a training-year event in would fail here before any scoring."""
    for case in replay_cases.CASES:
        replay_cases.sample_note(case.init_date.year, _REAL_MANIFEST)


def test_the_catalogue_is_well_formed():
    from app.utils import india_districts

    ids = {r.region_id for r in india_districts.load_registry()}
    dates = [c.init_date for c in replay_cases.CASES]
    assert len(set(dates)) == len(dates), "two cases share an init date"
    assert len({c.id for c in replay_cases.CASES}) == len(replay_cases.CASES)
    for c in replay_cases.CASES:
        assert c.focus_region_id in ids, c
        # valid_date = init + (lead - 1): the peak must fall inside Days 1-10
        lead = (c.peak_valid_date - c.init_date).days + 1
        assert 1 <= lead <= 10, (c.id, lead)


def test_case_for_matches_a_date_in_any_form():
    first = replay_cases.CASES[0]
    assert replay_cases.case_for(first.init_date) is first
    assert replay_cases.case_for(str(first.init_date)) is first
    assert replay_cases.case_for(pd.Timestamp(first.init_date)) is first
    assert replay_cases.case_for(dt.date(1999, 1, 1)) is None
    assert replay_cases.case_for(None) is None


class _State:
    run_id = "run_PLUMBING"


def test_as_of_init_scoring_never_reads_a_precomputed_answer(monkeypatch):
    """Precomputed cycles were scored with forecast_error_lag filled in. Serving one of
    them as an as-of-init score would put the leak straight back."""
    monkeypatch.setattr(settings, "serving_read_only", False)
    monkeypatch.setattr(inference.parquet_store, "store_fingerprint", lambda: "fp")
    inference._score_cache.clear()

    def _must_not_read(*a, **k):
        raise AssertionError("as-of-init scoring read a precomputed artifact")

    monkeypatch.setattr(inference.precomputed, "read_scored_cycle", _must_not_read)
    monkeypatch.setattr(inference.parquet_store, "read_dataset", lambda **k: pd.DataFrame())
    assert inference.score_cycle(_State(), "2018-08-13", as_of_init=True) is None


def test_as_of_init_and_default_scores_are_cached_apart(monkeypatch):
    """One cycle, two different questions: the cache must not answer one with the other."""
    monkeypatch.setattr(inference.parquet_store, "store_fingerprint", lambda: "fp")
    inference._score_cache.clear()
    sentinel = object()
    key_default = ("run_PLUMBING", str(pd.Timestamp("2018-08-13")), "fp", False)
    inference._score_cache[key_default] = sentinel
    monkeypatch.setattr(settings, "serving_read_only", False)
    monkeypatch.setattr(inference.precomputed, "read_scored_cycle", lambda *a, **k: None)
    monkeypatch.setattr(inference.parquet_store, "read_dataset", lambda **k: pd.DataFrame())
    try:
        assert inference.score_cycle(_State(), "2018-08-13") is sentinel
        assert inference.score_cycle(_State(), "2018-08-13", as_of_init=True) is None
    finally:
        inference._score_cache.clear()
