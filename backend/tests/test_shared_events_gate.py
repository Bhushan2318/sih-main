"""The gate compares the new run and the served run on the same events, with the same labels.

Each run used to be scored on its own events against its own thresholds, so the two
ROC-AUCs the gate compared measured different things, and nothing checked that the events
or labels matched. With bias-corrected busts (label version 2) the labels differ by design.
So both runs are scored as of issue on the same year, joined on the event keys, and judged
against the new run's label - "can each model see this bust coming?" - before the unchanged
gate is asked.

PLUMBING FIXTURES: hand-built event frames whose AUCs are worked out by hand.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts import publish_serving_model as pub
from scripts import score_shared_events as sse

KEYS = ["region_id", "init_date", "valid_date", "lead_time_days"]


def _events(n=4, proba=(0.1, 0.4, 0.35, 0.8), y=(0, 0, 1, 1)):
    d = pd.Timestamp("2017-07-01")
    return pd.DataFrame({"region_id": [f"D{i}" for i in range(n)], "init_date": d,
                         "valid_date": d, "lead_time_days": 1,
                         "y_bust": list(y)[:n], "model_proba": list(proba)[:n]})


# --------------------------------------------------------------- joining

def test_both_probabilities_land_on_the_new_runs_events_and_label():
    new = _events(y=(0, 0, 1, 1))
    inc = _events(proba=(0.9, 0.2, 0.1, 0.3), y=(1, 1, 1, 1))   # its own label is ignored
    out = sse.shared_events(new, inc)
    assert list(out["y_bust"]) == [0, 0, 1, 1]
    assert list(out["proba_new"]) == [0.1, 0.4, 0.35, 0.8]
    assert list(out["proba_incumbent"]) == [0.9, 0.2, 0.1, 0.3]


def test_incumbent_events_the_new_label_dropped_are_left_out():
    """Label version 2 drops events with no corrected error; the incumbent scored them."""
    new = _events(n=3, y=(0, 1, 1))
    inc = _events(n=4)
    assert len(sse.shared_events(new, inc)) == 3


def test_a_new_event_without_an_incumbent_probability_is_refused():
    with pytest.raises(ValueError, match="incumbent"):
        sse.shared_events(_events(n=4), _events(n=3))


def test_duplicate_event_keys_are_refused():
    dup = pd.concat([_events(n=2), _events(n=1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        sse.shared_events(dup, _events(n=4))


def test_the_incumbent_matches_its_reference_scoring_or_is_refused():
    inc = _events()
    ref = _events()
    sse.check_incumbent_reference(inc, ref)                       # identical: fine
    ref.loc[2, "model_proba"] += 1e-3
    with pytest.raises(ValueError, match="reference"):
        sse.check_incumbent_reference(inc, ref)


# --------------------------------------------------------------- the publish gate

def _shared_file(tmp_path, *, new_run="run_new", incumbent="run_live", year=2017,
                 proba_new=(0.1, 0.4, 0.35, 0.8), proba_inc=(0.1, 0.4, 0.35, 0.8)):
    out = sse.shared_events(_events(proba=proba_new), _events(proba=proba_inc))
    out = out.assign(new_run_id=new_run, incumbent_run_id=incumbent, year=year, label_version=2)
    p = tmp_path / "shared.parquet"
    out.to_parquet(p)
    return p


def test_both_metrics_come_from_the_one_file(tmp_path):
    # Busts are rows 3-4. Incumbent: of the four bust/no-bust pairs only 0.35 > 0.2 ranks
    # right, so 0.25; new run: three of four, 0.75.
    new, inc = pub.shared_metrics(_shared_file(tmp_path, proba_inc=(0.2, 0.4, 0.35, 0.1)),
                                  "run_new", "run_live", 2017)
    assert new["roc_auc"] == pytest.approx(0.75)
    assert inc["roc_auc"] == pytest.approx(0.25)


@pytest.mark.parametrize("kw, says", [({"new_run": "run_other"}, "run_other"),
                                      ({"incumbent": "run_old"}, "run_old"),
                                      ({"year": 2018}, "2018")])
def test_a_file_for_another_run_incumbent_or_year_is_refused(tmp_path, kw, says):
    with pytest.raises(ValueError, match=says):
        pub.shared_metrics(_shared_file(tmp_path, **kw), "run_new", "run_live", 2017)


def test_the_new_runs_shared_score_must_match_its_recorded_test_score():
    pub.check_recorded_score({"roc_auc": 0.7500}, {"classifier": {"test": {"roc_auc": 0.7505}}})
    with pytest.raises(ValueError, match="recorded"):
        pub.check_recorded_score({"roc_auc": 0.75},
                                 {"classifier": {"test": {"roc_auc": 0.78}}})


def test_a_label_version_two_run_cannot_use_the_old_incumbent_score(tmp_path):
    with pytest.raises(ValueError, match="shared-events"):
        pub.refuse_incumbent_score_for_new_labels({"label_version": 2})
    pub.refuse_incumbent_score_for_new_labels({})                  # label version 1: allowed


def test_the_unchanged_gate_decides_on_the_shared_metrics(tmp_path):
    root = tmp_path / "live"
    d = root / "data" / "models" / "run_live"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"classifier": {"test": {"roc_auc": 0.84}}}))
    (root / "data" / "models" / "current.json").write_text(json.dumps({"run_id": "run_live"}))
    new, inc = pub.shared_metrics(_shared_file(tmp_path, proba_inc=(0.8, 0.4, 0.35, 0.1)),
                                  "run_new", "run_live", 2017)
    promote, why = pub.gate_decision({"classifier": {"test": new}}, root, "run_live",
                                     incumbent=inc)
    assert promote is True, why
    assert (d / "metrics.json").read_text() == json.dumps({"classifier": {"test": {"roc_auc": 0.84}}})
