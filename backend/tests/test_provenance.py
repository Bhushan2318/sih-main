"""What a finished variable's checkpoint is keyed on, and what a run records about itself.

A seventeen-year retrain checkpoints each finished variable so a crash can resume
(`pooled_training._variable_checkpoint_path`). The key covered the years, the cycles and
the model parameters - not the features, the code or the cached data. So a retrain after
a change to the feature list, the label or the cache would quietly reuse a regressor
trained the old way, or crash hours in when its saved feature list no longer matched.

The run manifest recorded none of what produced the run either: no commit, no library
versions, no fit settings. These tests pin both.

Inputs are hand-built (cycle sets, tiny parquet files) for keying logic only.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.ml import pooled_training as pt
from app.ml import provenance


CYCLES = {pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-02")}
VAL = {pd.Timestamp("2000-02-01")}


def _key(tmp_path, context):
    return pt._variable_checkpoint_path(tmp_path, "temperature_c", [2000], CYCLES, VAL,
                                        context=context)


def _ctx(**over):
    base = {"features": ["lead_time_days", "spread"], "code": "abc", "cache": {"2000": "f1"},
            "libraries": {"xgboost": "2.0.3"}}
    base.update(over)
    return base


def test_identical_context_gives_the_same_key(tmp_path):
    assert _key(tmp_path, _ctx()) == _key(tmp_path, _ctx())


@pytest.mark.parametrize("change", [
    {"features": ["lead_time_days"]},                 # an input removed
    {"code": "abd"},                                   # training code edited
    {"cache": {"2000": "f2"}},                         # the cached year rebuilt
    {"libraries": {"xgboost": "2.1.0"}},               # a different booster
])
def test_any_change_that_alters_the_model_changes_the_key(tmp_path, change):
    assert _key(tmp_path, _ctx(**change)) != _key(tmp_path, _ctx())


def test_context_order_does_not_matter(tmp_path):
    a = _ctx(cache={"2000": "f1", "2001": "g1"})
    b = _ctx(cache={"2001": "g1", "2000": "f1"})
    assert _key(tmp_path, a) == _key(tmp_path, b)


# --------------------------------------------------------------- the fingerprints

def test_source_fingerprint_changes_when_training_code_changes(tmp_path, monkeypatch):
    pkg = tmp_path / "app"
    (pkg / "ml").mkdir(parents=True)
    f = pkg / "ml" / "x.py"
    f.write_text("A = 1\n")
    monkeypatch.setattr(provenance, "TRAINING_SOURCES", [pkg / "ml"])
    one = provenance.source_fingerprint()
    f.write_text("A = 2\n")
    assert provenance.source_fingerprint() != one


def test_cache_fingerprint_follows_the_file_not_its_name(tmp_path):
    p = tmp_path / "paired_2000.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(p)
    one = provenance.cache_fingerprint({2000: p})
    pd.DataFrame({"a": [1, 2, 3, 4]}).to_parquet(p)
    two = provenance.cache_fingerprint({2000: p})
    assert one["2000"] != two["2000"]


def test_the_feature_list_reaches_the_key_from_the_cache_schema(tmp_path, monkeypatch):
    """The context is built from what the regressors would actually read: the feature
    columns `regressors.feature_columns` picks from the cached schema."""
    p = tmp_path / "paired_2000.parquet"
    pd.DataFrame({"lead_time_days": [1], "ensemble_spread": [0.1],
                  "variable": ["t"]}).to_parquet(p)
    ctx = pt._checkpoint_context({2000: p}, [2000])
    assert "lead_time_days" in ctx["features"]
    assert set(ctx) >= {"features", "code", "cache", "libraries"}


# --------------------------------------------------------------- the manifest

def test_run_provenance_names_everything_that_produced_the_run():
    rec = provenance.run_provenance(fit_mode="sample", max_fit_cycles=2000)
    for key in ("git_sha", "git_dirty", "source_sha256", "libraries", "python",
                "fit_mode", "max_fit_cycles"):
        assert key in rec, key
    for lib in ("xgboost", "numpy", "pandas", "pyarrow", "sklearn"):
        assert lib in rec["libraries"], lib
    json.dumps(rec)   # the manifest is JSON
