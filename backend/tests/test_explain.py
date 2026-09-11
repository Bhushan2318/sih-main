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
