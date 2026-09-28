"""The refresh workflow's contract with the pinned serving model.

A workflow cannot be unit tested by running it, but the wiring that decides whether CI
trains its own model can be: if the `if:` guards ever drift from the step that installs
the pinned model, the pipeline would train a model and quietly publish it over the one
that was pinned - which is exactly the failure the pinning exists to prevent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
TRAINING_STEPS = (
    "Train on the full archive, in a throwaway copy of the store",
    "Prove the archive training did not touch the serving store",
    "Score the baseline ladder for this run",
)


def _steps(path: Path) -> list:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    job = next(iter(doc["jobs"].values()))
    return doc, job["steps"]


def test_every_training_step_is_skipped_when_a_model_is_pinned():
    doc, steps = _steps(WORKFLOWS / "refresh-data.yml")
    pin = [s for s in steps if s.get("id") == "pin"]
    assert len(pin) == 1, "the step that installs the pinned model must exist, with id 'pin'"
    assert "install_serving_model" in pin[0]["run"]

    by_name = {s.get("name"): s for s in steps}
    for name in TRAINING_STEPS:
        assert name in by_name, f"{name} is gone - update this test with the pipeline"
        assert by_name[name].get("if") == "steps.pin.outputs.pinned != 'true'", \
            f"{name} would run even with a model pinned"


def test_the_pinned_model_is_installed_before_anything_packages_it():
    _, steps = _steps(WORKFLOWS / "refresh-data.yml")
    order = [s.get("id") or s.get("name") for s in steps]
    assert order.index("pin") < order.index("pack"), \
        "the model must be installed before the artifact is packed"
    # And after the restore, or the restored artifact's own current.json would win.
    assert order.index("Restore the previous store") < order.index("pin")


def test_one_switch_turns_pinning_off_for_a_run():
    doc, _ = _steps(WORKFLOWS / "refresh-data.yml")
    inputs = doc[True]["workflow_dispatch"]["inputs"]  # `on:` parses as the bool True
    assert "ignore_pinned_model" in inputs


def test_the_memory_measurement_can_measure_the_pinned_model():
    doc, steps = _steps(WORKFLOWS / "measure-serving-memory.yml")
    assert "use_serving_model" in doc[True]["workflow_dispatch"]["inputs"]
    install = [s for s in steps if "install_serving_model" in str(s.get("run", ""))]
    assert len(install) == 1 and install[0].get("if") == "inputs.use_serving_model"


def test_warm_on_push_never_warms_a_box_without_its_prebuilt_bundle():
    """A box running new code on an old bundle builds every screen live, and a warm request
    on top of a visitor is what OOM-killed it on 2026-09-26 20:03Z and 2026-09-27 13:03Z.
    The workflow must read /api/health's `prebuilt` before sending anything heavy, and say
    what to do instead of warming."""
    _, steps = _steps(WORKFLOWS / "warm-on-push.yml")
    script = "\n".join(s.get("run", "") for s in steps)
    assert "prebuilt" in script
    assert "refresh-data.yml" in script
    first_check = script.index("prebuilt")
    for ep in ("/api/regions/all", "/api/ensemble", "/api/replay"):
        if ep in script:
            assert script.index(ep) > first_check, f"{ep} is requested before the check"
