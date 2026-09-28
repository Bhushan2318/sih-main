"""Pinning a trained model as the one the site serves.

CI cannot train the model the site now serves: seventeen pooled years need a GPU and far
more memory than a runner has, and more wall clock than a job may take. So a finished run
is published as the `serving-model` release, and the refresh workflow installs it instead
of training - while still refreshing forecast data every six hours.

These tests cover the parts that decide whether a model may be served at all: the
completeness check, the bundle's integrity on the way in, and the gate. They use
hand-built run directories (clearly synthetic, no model is trained here) because what is
under test is the refusal logic, not the meteorology.
"""
from __future__ import annotations

import json
import subprocess
import tarfile

import pytest

from scripts import install_serving_model as inst
from scripts import publish_serving_model as pub


def _run_dir(root, run_id="run_x", *, complete=True, roc_auc=0.84):
    d = root / "data" / "models" / run_id
    d.mkdir(parents=True)
    for name in inst.REQUIRED_FILES:
        if not complete and name == "shap_summary.parquet":
            continue
        (d / name).write_text("{}")
    (d / "temperature_c_regressor.json").write_text("{}")
    (d / "metrics.json").write_text(json.dumps(
        {"classifier": {"test": {"roc_auc": roc_auc}}}))
    (d / "current.json").unlink(missing_ok=True)
    return d


def test_a_run_without_its_shap_summary_is_not_publishable(tmp_path):
    d = _run_dir(tmp_path, complete=False)
    assert inst.missing_files(d) == ["shap_summary.parquet"]
    assert inst.missing_files(_run_dir(tmp_path, "run_ok")) == []


def test_a_bundle_whose_bytes_changed_is_refused(tmp_path):
    src = _run_dir(tmp_path / "src")
    bundle = pub.pack(src, tmp_path / "b.tar.gz")
    manifest = tmp_path / "model.json"
    manifest.write_text(json.dumps({"run_id": "run_x", "sha256": "0" * 64}))
    with pytest.raises(ValueError, match="sha256"):
        inst.install(bundle, manifest, tmp_path / "dest")


def test_a_bundle_that_writes_outside_the_run_directory_is_refused(tmp_path):
    bundle = tmp_path / "evil.tar.gz"
    stray = tmp_path / "passwd"
    stray.write_text("x")
    with tarfile.open(bundle, "w:gz") as tar:
        tar.add(stray, arcname="etc/passwd")
    manifest = tmp_path / "model.json"
    manifest.write_text(json.dumps({"run_id": "run_x", "sha256": inst.sha256_of(bundle)}))
    with pytest.raises(ValueError, match="outside"):
        inst.install(bundle, manifest, tmp_path / "dest")


def test_install_round_trips_a_packed_run_and_makes_it_current(tmp_path, monkeypatch):
    from app.ml import registry

    src = _run_dir(tmp_path / "src")
    bundle = pub.pack(src, tmp_path / "b.tar.gz")
    manifest = tmp_path / "model.json"
    manifest.write_text(json.dumps({"run_id": "run_x", "sha256": inst.sha256_of(bundle)}))

    dest = tmp_path / "dest"
    monkeypatch.setattr(registry, "MODEL_DIR", dest / "data" / "models")
    monkeypatch.setattr(registry, "CURRENT_JSON", dest / "data" / "models" / "current.json")
    assert inst.install(bundle, manifest, dest) == "run_x"
    assert sorted(p.name for p in (dest / "data" / "models" / "run_x").iterdir()) == \
        sorted(list(inst.REQUIRED_FILES) + ["temperature_c_regressor.json"])
    assert registry.current_run_id() == "run_x"


def test_an_incomplete_bundle_is_refused_even_if_its_hash_matches(tmp_path, monkeypatch):
    from app.ml import registry

    src = _run_dir(tmp_path / "src", complete=False)
    bundle = pub.pack(src, tmp_path / "b.tar.gz")
    manifest = tmp_path / "model.json"
    manifest.write_text(json.dumps({"run_id": "run_x", "sha256": inst.sha256_of(bundle)}))
    dest = tmp_path / "dest"
    monkeypatch.setattr(registry, "MODEL_DIR", dest / "data" / "models")
    monkeypatch.setattr(registry, "CURRENT_JSON", dest / "data" / "models" / "current.json")
    with pytest.raises(ValueError, match="shap_summary"):
        inst.install(bundle, manifest, dest)


# --- the gate, unchanged, asked about a pinned model ---------------------------------

def _live_root(tmp_path, roc_auc):
    root = tmp_path / "live"
    d = _run_dir(root, "run_live", roc_auc=roc_auc)
    (root / "data" / "models" / "current.json").write_text(json.dumps({"run_id": "run_live"}))
    return root, d


def test_the_gate_refuses_a_model_well_below_the_one_being_served(tmp_path):
    root, _ = _live_root(tmp_path, 0.84)
    promote, why = pub.gate_decision({"classifier": {"test": {"roc_auc": 0.70}}},
                                     root, "run_live")
    assert promote is False
    assert "NOT promoted" in why


def test_the_gate_allows_a_model_above_the_one_being_served(tmp_path):
    root, _ = _live_root(tmp_path, 0.80)
    promote, why = pub.gate_decision({"classifier": {"test": {"roc_auc": 0.8435}}},
                                     root, "run_live")
    assert promote is True
    assert "promoted" in why


def test_the_gate_question_leaves_this_checkouts_registry_pointing_where_it_was(tmp_path):
    from app.ml import registry

    before, before_json = registry.MODEL_DIR, registry.CURRENT_JSON
    root, _ = _live_root(tmp_path, 0.80)
    pub.gate_decision({"classifier": {"test": {"roc_auc": 0.9}}}, root, "run_live")
    assert (registry.MODEL_DIR, registry.CURRENT_JSON) == (before, before_json)


def _incumbent_scores(tmp_path, *, run_id="run_live", year=2017, as_of_issue=True):
    """A scripts.score_run_on_year output with a ROC-AUC of exactly 0.75: three of its four
    bust/no-bust pairs rank the right way round."""
    import pandas as pd

    path = tmp_path / f"{run_id}_on_{year}_as_of_issue.parquet"
    pd.DataFrame({"y_bust": [0, 0, 1, 1], "model_proba": [0.10, 0.40, 0.35, 0.80],
                  "scored_run_id": run_id, "scored_year": year,
                  "as_of_issue": as_of_issue}).to_parquet(path)
    return path


def test_like_for_like_the_gate_compares_against_the_served_runs_as_of_issue_score(tmp_path):
    """A served run trained with forecast_error_lag recorded its test score with the lag filled
    in (docs/known-issues.md). A run trained without it is refused against that figure for
    being honest, and promoted against the served run's own skill as of issue time."""
    root, _ = _live_root(tmp_path, 0.8435)
    new = {"classifier": {"test": {"roc_auc": 0.74}}}
    assert pub.gate_decision(new, root, "run_live")[0] is False

    honest = pub.incumbent_as_of_issue(_incumbent_scores(tmp_path), "run_live", 2017)
    assert honest["roc_auc"] == pytest.approx(0.75)
    promote, why = pub.gate_decision(new, root, "run_live", incumbent=honest)
    assert promote is True, why
    assert "0.7500" in why


def test_the_like_for_like_question_leaves_the_downloaded_metrics_as_they_were(tmp_path):
    root, d = _live_root(tmp_path, 0.8435)
    before = (d / "metrics.json").read_bytes()
    honest = pub.incumbent_as_of_issue(_incumbent_scores(tmp_path), "run_live", 2017)
    pub.gate_decision({"classifier": {"test": {"roc_auc": 0.74}}}, root, "run_live", incumbent=honest)
    assert (d / "metrics.json").read_bytes() == before


@pytest.mark.parametrize("kw, says", [({"run_id": "run_other"}, "run_other"),
                                      ({"year": 2018}, "2018"),
                                      ({"as_of_issue": False}, "as of issue time")])
def test_a_score_for_another_run_year_or_mode_is_refused(tmp_path, kw, says):
    with pytest.raises(ValueError, match=says):
        pub.incumbent_as_of_issue(_incumbent_scores(tmp_path, **kw), "run_live", 2017)


def test_the_serving_check_runs_without_a_gpu(tmp_path, monkeypatch):
    """The box it stands in for has none, and a training run may be using the one here.

    A pooled model is trained with device="cuda", which XGBoost records in the saved
    model, so a check that ran on a GPU would exercise a path Render never takes.
    """
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw["env"])
        return subprocess.CompletedProcess(cmd, 0, stdout='{"ok": true}\n', stderr="")

    monkeypatch.setattr(pub.subprocess, "run", fake_run)
    src = _run_dir(tmp_path / "src")
    assert pub.serving_check(tmp_path / "live", src) == {"ok": True}
    assert seen["CUDA_VISIBLE_DEVICES"] == ""


def test_the_check_reads_regions_from_the_shape_that_endpoint_actually_returns():
    """/api/regions/all answers for every lead day at once.

    It returns `days[]`, each with its own `regions`, and no top-level `regions` key -
    see AllRegionsResponse in the frontend's types. Reading a top-level `regions` gave an
    empty list, and the check refused a model that was scoring 36 districts across all ten
    lead days. Caught 2026-09-22 on the first real dry run of the seventeen-year model.
    """
    from scripts import _serving_check_worker as w

    all_regions = {"days": [{"lead_time_days": 1, "regions": [{"region_id": "IN-TN-CHENNAI"}]},
                            {"lead_time_days": 2, "regions": [{"region_id": "IN-TN-CHENNAI"},
                                                              {"region_id": "IN-HP-SHIMLA"}]}]}
    rows = w.scored_regions(all_regions)
    assert [r["region_id"] for r in rows] == ["IN-TN-CHENNAI", "IN-TN-CHENNAI", "IN-HP-SHIMLA"]
    # The single-lead endpoint's shape still works, and an empty body is empty.
    assert w.scored_regions({"regions": [{"region_id": "IN-HP-SHIMLA"}]})[0]["region_id"] \
        == "IN-HP-SHIMLA"
    assert w.scored_regions({"days": []}) == []
    assert w.scored_regions({}) == []


def test_the_check_refuses_a_run_whose_past_events_do_not_serve():
    """A run that carries replay_cases/ must list them and open Replay on the first one.
    Otherwise the site's Replay would open on an unverified live cycle again and nothing
    would say why. Shape-only payloads; nothing here is a metric."""
    from scripts import _serving_check_worker as w

    event = {"init_date": "2018-08-13", "kind": "event", "focus_region_id": "IN-KL-IDUKKI"}
    live = {"init_date": "2026-09-25", "kind": "forecast"}
    opened = {"init_date": "2018-08-13", "steps": [{}] * 10,
              "focus": {"region_id": "IN-KL-IDUKKI"}}

    assert w.replay_event_problem([event, live], opened, expect_events=True) is None
    assert "no past event" in w.replay_event_problem([live], opened, expect_events=True)
    assert "opened on" in w.replay_event_problem(
        [event, live], {**opened, "init_date": "2026-09-25"}, expect_events=True)
    assert "lead days" in w.replay_event_problem(
        [event], {**opened, "steps": [{}] * 3}, expect_events=True)
    assert "focus" in w.replay_event_problem(
        [event], {**opened, "focus": {"region_id": "IN-TN-CHENNAI"}}, expect_events=True)
    # A run without cases is checked for nothing new.
    assert w.replay_event_problem([live], {}, expect_events=False) is None
