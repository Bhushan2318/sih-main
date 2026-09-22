"""SHAP must stay SHAP at district scale.

explain_model ran TreeExplainer over the whole validation frame - 1.8 M rows per variable
for the 2017 district retrain - and _shap_values swallows any exception, so a run that
ran out of time or memory degraded silently to feature importance, and the region panel's
"what drove this prediction" (rule 9) stopped being SHAP without anything failing.

Sampling is stratified by the groups the summary is published at - (region_id,
lead_time_days) - so every group the panel can ask for is still explained, by at most
SHAP_ROWS_PER_GROUP rows. These are plumbing tests: the frames are built here, SHAP itself
is replaced by a recorder, and no number from them is a metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import pytest

from app.ml import explain


def _frame(n_regions=3, leads=(1, 2), rows_each=50):
    rows = []
    for r in range(n_regions):
        for lead in leads:
            for i in range(rows_each):
                rows.append({"region_id": f"IN-XX-R{r}", "lead_time_days": lead,
                             "f1": float(i), "f2": float(r * 10 + i)})
    df = pd.DataFrame(rows)
    df["region_id"] = df["region_id"].astype("category")
    return df


class _Recorder:
    def __init__(self):
        self.seen = []

    def __call__(self, model, X):
        self.seen.append(X.index.to_numpy().copy())
        return np.ones(X.shape)


def test_explains_at_most_the_cap_per_group(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(explain, "_shap_values", rec)
    out = explain.explain_model(object(), _frame(), ["f1", "f2"], [], "m", max_rows_per_group=10)
    assert len(rec.seen[0]) == 3 * 2 * 10
    assert set(out["method"]) == {"shap"}
    groups = out[out.group_region_id != "__all__"][["group_region_id", "group_lead_time_days"]]
    assert len(groups.drop_duplicates()) == 6, "every (region, lead) group is still explained"


def test_small_groups_keep_every_row(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(explain, "_shap_values", rec)
    explain.explain_model(object(), _frame(rows_each=7), ["f1", "f2"], [], "m", max_rows_per_group=100)
    assert len(rec.seen[0]) == 3 * 2 * 7


def test_the_sample_is_reproducible(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(explain, "_shap_values", rec)
    df = _frame()
    explain.explain_model(object(), df, ["f1", "f2"], [], "m", max_rows_per_group=10)
    explain.explain_model(object(), df, ["f1", "f2"], [], "m", max_rows_per_group=10)
    assert np.array_equal(np.sort(rec.seen[0]), np.sort(rec.seen[1]))


def test_no_cap_explains_every_row(monkeypatch):
    """The classifier is what the region panel serves, and at district scale it costs about
    half a minute in full - so it is explained without a cap."""
    rec = _Recorder()
    monkeypatch.setattr(explain, "_shap_values", rec)
    explain.explain_model(object(), _frame(), ["f1", "f2"], [], "m")
    assert len(rec.seen[0]) == 3 * 2 * 50


def test_the_regressor_cap_is_a_real_bound():
    assert explain.SHAP_ROWS_PER_GROUP is not None and 10 <= explain.SHAP_ROWS_PER_GROUP <= 100


# --- a model reloaded from the registry can still be explained -------------------------
# Real failure 2026-09-22: XGBoost's JSON round-trip does not preserve the sklearn
# wrapper's enable_categorical flag, so TreeExplainer refused a reloaded model whose
# frame has categorical columns ("Invalid columns: season: cat") and explain_model
# silently fell back to feature importance. Training never saw it - it explains the
# in-memory model - but anything explaining a SAVED run does. The values with the flag
# restored are identical to the in-memory model's.

def test_a_registry_loaded_model_keeps_its_categorical_flag(tmp_path, monkeypatch):
    """Shapes-only synthetic frame."""
    import numpy as np
    import pandas as pd
    import xgboost as xgb

    from app.ml import registry

    monkeypatch.setattr(registry, "MODEL_DIR", tmp_path)
    X = pd.DataFrame({"a": np.random.default_rng(0).random(60),
                      "season": pd.Categorical(["DJF", "JJAS"] * 30)})
    y = np.random.default_rng(1).integers(0, 2, 60)
    clf = xgb.XGBClassifier(n_estimators=5, max_depth=2, enable_categorical=True)
    clf.fit(X, y)
    reg = xgb.XGBRegressor(n_estimators=5, max_depth=2, enable_categorical=True)
    reg.fit(X, y.astype(float))
    registry.save_classifier("run_x", clf, list(X.columns))
    registry.save_regressor("run_x", "temperature_c", reg, list(X.columns))

    loaded_clf, _ = registry.load_classifier("run_x")
    loaded_reg, _ = registry.load_regressors("run_x")["temperature_c"]
    assert loaded_clf.get_params()["enable_categorical"] is True
    assert loaded_reg.get_params()["enable_categorical"] is True
    # The point of the flag: predicting and explaining a categorical frame both work.
    assert len(loaded_clf.predict_proba(X)) == len(X)
    shap = pytest.importorskip("shap")
    vals = shap.TreeExplainer(loaded_clf).shap_values(X, check_additivity=False)
    assert np.asarray(vals).shape == X.shape


def test_per_group_means_describe_their_own_group_when_the_index_repeats(monkeypatch):
    """A frame concatenated from batches has repeated labels. The summary must not care.

    finalize_for_serving builds each regressor's explanation sample one batch of forecast
    dates at a time, and every batch is reset to its own 0..n-1 index. Grouping by label
    rather than by position then averaged rows from other groups into each group - which
    does not fail, it just makes every district's "what drove this prediction" identical
    to the national mean. Caught 2026-09-22 while finalising the seventeen-year run.
    """
    # Each half is one region, all of whose contributions are the same number, so the
    # right per-region answer is visible without computing anything.
    def half(region, value):
        return pd.DataFrame({"f1": np.full(50, value), "f2": np.full(50, value),
                             "region_id": region, "lead_time_days": 1}, index=range(50))

    repeated = pd.concat([half("R_A", 0.0), half("R_B", 10.0)])
    assert repeated.index.has_duplicates, "this test is pointless without repeated labels"

    # The recorded "SHAP values" are the feature values themselves, so a group's mean
    # absolute contribution is just its own value.
    monkeypatch.setattr(explain, "_shap_values", lambda model, X: X.to_numpy(float))

    out = explain.explain_model(object(), repeated, ["f1", "f2"], [], model_name="m",
                                max_rows_per_group=None)
    per_region = out[(out["group_region_id"] != "__all__") & (out["feature"] == "f1")]
    assert dict(zip(per_region["group_region_id"], per_region["mean_abs_shap"])) == \
        {"R_A": 0.0, "R_B": 10.0}
    # And the same rows with unique labels must give the same answer.
    unique = repeated.reset_index(drop=True)
    same = explain.explain_model(object(), unique, ["f1", "f2"], [], model_name="m",
                                 max_rows_per_group=None)
    pd.testing.assert_frame_equal(out.sort_values(list(out.columns)).reset_index(drop=True),
                                  same.sort_values(list(same.columns)).reset_index(drop=True))
