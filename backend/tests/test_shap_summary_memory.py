"""The served SHAP summary must not cost hundreds of megabytes to hold.

The box is killed at 512 MB, not throttled. `inference.load_model_state` reads
`shap_summary.parquet` with a bare `pd.read_parquet`, which was fine while a run
explained 36 districts and became a serving hazard the moment one explained 666:
run_20260922T043925Z's summary is 2,101,395 rows and measured **+424 MB of RSS** to
load on 2026-09-23, against ~388 MB already in use.

Nothing about the file needs those bytes. Four of its six columns are low-cardinality
labels - 9 models, 667 region groups, ~95 features, 2 methods - repeated across two
million rows. As categories the same frame is 22 MB. This is a representation change
only: no row is dropped and no value changes, because `top_factors_for` takes a `model`
argument and a caller may legitimately ask for a regressor's rows, not just the
classifier's.

These are plumbing tests. The frames are built here and no number in them is a metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml import inference as inf


def _summary(n_regions=200, n_features=40):
    """The shape a pooled run produces: few distinct labels, many rows."""
    rows = n_regions * n_features
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "model": np.where(rng.random(rows) < 0.5, "classifier", "regressor::rainfall_mm"),
        "group_region_id": np.repeat([f"IN-XX-{i:04d}" for i in range(n_regions)], n_features),
        "group_lead_time_days": rng.integers(1, 11, rows),
        "feature": np.tile([f"pred_err_{i}" for i in range(n_features)], n_regions),
        "mean_abs_shap": rng.random(rows),
        "method": "shap",
    })


def test_the_label_columns_are_held_as_categories(tmp_path):
    """Object dtype over two million rows is where the 424 MB went."""
    path = tmp_path / "shap_summary.parquet"
    _summary().to_parquet(path, index=False)

    got = inf._read_shap_summary(path)
    for col in ("model", "group_region_id", "feature", "method"):
        assert str(got[col].dtype) == "category", f"{col} is {got[col].dtype}"
    assert got.memory_usage(deep=True).sum() < \
        pd.read_parquet(path).memory_usage(deep=True).sum()


def test_no_row_and_no_value_is_lost(tmp_path):
    """A representation change, not a filter - the regressor rows must survive.

    top_factors_for takes a `model` argument, so dropping everything but the
    classifier's rows would quietly break any caller that asks for a regressor's.
    """
    path = tmp_path / "shap_summary.parquet"
    original = _summary()
    original.to_parquet(path, index=False)

    got = inf._read_shap_summary(path)
    assert len(got) == len(original)
    assert set(got["model"].astype(str)) == set(original["model"])
    pd.testing.assert_series_equal(
        got["mean_abs_shap"].astype(float).reset_index(drop=True),
        original["mean_abs_shap"].reset_index(drop=True),
        check_names=False, rtol=1e-6)


def test_a_missing_summary_is_an_empty_frame_not_an_error(tmp_path):
    """A run without one serves no explanation; it must not fail to boot."""
    assert inf._read_shap_summary(tmp_path / "absent.parquet").empty


def test_the_panel_still_finds_its_factors_through_the_categories(tmp_path):
    """Categories must not change what top_factors_for returns."""
    from app.ml.explain import top_factors_for

    path = tmp_path / "shap_summary.parquet"
    original = _summary()
    original.to_parquet(path, index=False)
    region = original["group_region_id"].iloc[0]
    lead = int(original["group_lead_time_days"].iloc[0])

    a = top_factors_for(pd.read_parquet(path), region, lead, model="classifier", k=5)
    b = top_factors_for(inf._read_shap_summary(path), region, lead, model="classifier", k=5)
    assert a == b


# --------------------------------------------- the serving box reads the classifier's rows only
#
# On the box the SHAP summary is only ever asked for the classifier's factors (the district
# panel), so it reads just those rows, in batches: measured 2026-09-27 on
# run_20260922T043925Z (macOS), +247 MB resident for the whole summary against +73 MB for the
# classifier's 632,795 rows read 64k at a time. Filtering after a whole-file read saves almost
# nothing (+195 MB) - Arrow keeps the decode buffers of each million-row group.

def test_one_models_rows_read_in_batches_equal_the_full_read_filtered(tmp_path, monkeypatch):
    path = tmp_path / "shap_summary.parquet"
    _summary().to_parquet(path, index=False)
    monkeypatch.setattr(inf, "_SHAP_BATCH_ROWS", 997)  # many batches, none aligned to anything

    full = inf._read_shap_summary(path)
    want = full[full["model"] == "classifier"].reset_index(drop=True)
    got = inf._read_shap_summary(path, only_model="classifier")

    assert len(got) == len(want) > 0
    for col in want.columns:
        assert list(got[col].astype(str)) == list(want[col].astype(str)), col
        assert got[col].dtype.name == want[col].dtype.name, col


def test_the_panel_finds_the_same_factors_in_the_classifier_rows(tmp_path):
    from app.ml.explain import top_factors_for

    path = tmp_path / "shap_summary.parquet"
    _summary().to_parquet(path, index=False)
    full = inf._read_shap_summary(path)
    only = inf._read_shap_summary(path, only_model="classifier")
    for rid in ("IN-XX-0000", "IN-XX-0137", "IN-XX-9999"):  # the last falls back to __all__
        for lead in (1, 5, 10):
            assert top_factors_for(only, rid, lead, model="classifier", k=6) == \
                top_factors_for(full, rid, lead, model="classifier", k=6)


def test_the_site_only_ever_asks_for_the_classifiers_factors():
    """What the box's classifier-only read relies on. A new caller that wants a regressor's
    explanation must load the full summary, not find an empty list here."""
    import re
    from pathlib import Path

    root = Path(inf.__file__).resolve().parents[1]
    calls = []
    for p in list((root / "services").rglob("*.py")) + list((root / "api").rglob("*.py")):
        calls += re.findall(r"top_factors_for\(([^)]*)\)", p.read_text())
    assert calls, "the district panel's call to top_factors_for has moved - update this test"
    for args in calls:
        assert 'model="classifier"' in args, args
