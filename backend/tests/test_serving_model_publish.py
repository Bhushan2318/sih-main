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
