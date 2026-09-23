"""The deck generator has to describe a run you name, and survive a run without baselines.

Two defects found 2026-09-23 while generating the deck for the live model:

`collect()` read `registry.current_run_id()` and nothing else, so the only way to report on
a run was to point current.json at it. The 4060 session did exactly that - set it, generated,
restored it - and it worked. It is still a footgun: anything failing in between leaves the
machine serving a model nobody chose, and on that machine current.json is what the site
serves.

`to_markdown` then crashed on a run with no baseline ladder: `m.get('test_events'):,`
against None raises TypeError. Pooled runs have no ladder by design (run_baselines needs a
train split the pooled path does not carry), so the documented limitation presented itself
as a crash in the tool meant to work around it. A section that explains its own absence is
the correct behaviour.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts import ppt_figures as pf


def _events(run_id: str, n: int = 240) -> pd.DataFrame:
    """A minimal eval-events frame carrying every column the deck reads.

    Synthetic and labelled as such - it exercises the reporting path's plumbing, not any
    measurement. Values are shaped so both classes and all ten lead days are present.
    """
    rows = []
    for i in range(n):
        lead = (i % 10) + 1
        y = i % 2
        row = {
            "region_id": f"IN-XX-D{i % 12:02d}",
            "init_date": pd.Timestamp("2017-07-01"),
            "valid_date": pd.Timestamp("2017-07-01") + pd.Timedelta(days=lead - 1),
            "lead_time_days": lead,
            "month": 7,
            "season": "JJAS",
            "spread_mean": 1.0 + i % 3,
            "spread_max": 4.0,
            "historical_bust_frequency_region_season": 0.3,
            "bust_ratio": 0.5,
            "y_bust": y,
            "split": "test" if i % 4 else "train",
            "model_proba": 0.8 if y else 0.2,
        }
        for var in ("temperature_c", "rainfall_mm"):
            row[f"actual_err_{var}"] = 12.0 if y else 0.4
            row[f"pred_err_{var}"] = 3.0 if y else 0.3
            row[f"conf_{var}"] = 0.2 if y else 0.9
            row[f"spread_{var}"] = 0.7
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def stub(monkeypatch):
    """A run that exists, with metrics and thresholds but no baseline ladder."""
    def _apply(run_id="run_STUB", baselines=None):
        monkeypatch.setattr(pf.registry, "current_run_id", lambda: "run_CURRENT")
        monkeypatch.setattr(pf.registry, "load_metrics",
                            lambda r: {"classifier": {"test": {"roc_auc": 0.84, "pr_auc": 0.83,
                                                              "brier": 0.16, "f1": 0.75,
                                                              "precision": 0.77, "recall": 0.72,
                                                              "n": 180}}})
        monkeypatch.setattr(pf.registry, "load_baselines", lambda r: baselines)
        monkeypatch.setattr(pf.registry, "load_thresholds", lambda r: None)
        monkeypatch.setattr(pf, "_events", lambda r: _events(r))
        monkeypatch.setattr(pf, "_shap_top", lambda r, n=5: [])
        return run_id
    return _apply


def test_run_id_argument_overrides_current_json(stub):
    """Naming a run must not require pointing the served model at it."""
    stub()
    d = pf.collect(run_id="run_NAMED")
    assert d["run_id"] == "run_NAMED", (
        "collect() ignored the run it was given and described current.json instead")


def test_current_json_is_still_the_default(stub):
    stub()
    assert pf.collect()["run_id"] == "run_CURRENT"


def test_collect_never_touches_current_json(stub, monkeypatch):
    """Generating a deck must not be able to change which model is served.

    The previous way to report on a run was to set current.json and put it back. This
    asserts the tool has no reason to write it at all.
    """
    def explode(*a, **k):
        raise AssertionError("ppt_figures must never call registry.set_current")

    monkeypatch.setattr(pf.registry, "set_current", explode)
    stub()
    pf.collect(run_id="run_NAMED")


def test_markdown_survives_a_run_with_no_baseline_ladder(stub):
    """Pooled runs have no ladder by design; that is a missing section, not a crash."""
    stub(baselines=None)
    d = pf.collect(run_id="run_NAMED")
    assert d["baselines"] == []
    text = pf.to_markdown(d)          # used to raise TypeError on None:,
    assert "Baseline ladder" in text
    # The absence has to explain itself, or a reader assumes the ladder said nothing.
    low = text.lower()
    assert "not available" in low or "no baseline" in low


def test_the_ladder_still_renders_when_it_exists(stub):
    stub(baselines={"test_events": 1000, "test_cycles": 55, "bust_rate": 0.49,
                    "models": [{"name": "climatology", "brier": 0.25, "bss": 0.0,
                                "roc_auc": 0.5}]})
    text = pf.to_markdown(pf.collect(run_id="run_NAMED"))
    assert "climatology" in text
    assert "1,000 events" in text


def test_json_output_is_still_machine_readable(stub):
    stub()
    d = pf.collect(run_id="run_NAMED")
    json.loads(json.dumps(d, default=str))
