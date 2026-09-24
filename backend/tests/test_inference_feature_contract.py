"""The served model's feature contract must come from the model, not from a literal.

Regression test for a silent failure found on 2026-09-19. `inference._prep` hardcoded
`categorical = {"state_id", "season"}`, describing whatever the *current* code trains
with. Workstream C4 swapped the district-identity feature from `region_id` to
`state_id`, so every model trained before C4 still asks for `region_id` - and `_prep`
then sent that column, fully populated and correctly typed as `category`, down the
`pd.to_numeric(errors="coerce")` branch, turning 100% of rows into NaN.

Nothing raised. XGBoost routes NaN down a default branch learned at training, and since
`region_id` was never NaN in training that direction is arbitrary and identical for
every row, so every district got the same constant push. Measured on the local instance:
median bust probability 0.9927 on a cycle the model had *trained on*, against a held-out
ROC-AUC of 0.8411 - the two numbers disagreed because they came from different code
paths.

CLAUDE.md rule 3 is "refuse rather than patch; missing never becomes zero". Silently
coercing a real, present column into all-NaN is that rule's exact failure mode, so these
tests pin both halves: read the contract from the model, and refuse when a column that
had data is destroyed on the way in.
"""
import numpy as np
import pandas as pd
import pytest

from app.ml import inference


class _FakeBooster:
    def __init__(self, names, types):
        self.feature_names = names
        self.feature_types = types


class _FakeModel:
    """Stands in for XGBClassifier: only `get_booster()` matters to the code under test."""

    def __init__(self, names, types):
        self._b = _FakeBooster(names, types)

    def get_booster(self):
        return self._b


def _frame():
    return pd.DataFrame({
        "region_id": pd.Series(["IN-MH-NAGPUR", "IN-TN-CHENNAI"], dtype="category"),
        "season": pd.Series(["winter", "winter"], dtype="category"),
        "lead_time_days": [1, 2],
    })


def test_categorical_columns_come_from_the_model_not_a_literal():
    # A pre-C4 model: its own booster records region_id as categorical ('c').
    model = _FakeModel(["region_id", "season", "lead_time_days"], ["c", "c", "int"])
    assert inference.categorical_features(model) == {"region_id", "season"}


def test_a_post_c4_model_reports_state_id_instead():
    model = _FakeModel(["state_id", "season", "lead_time_days"], ["c", "c", "int"])
    assert inference.categorical_features(model) == {"state_id", "season"}


def test_prep_keeps_a_categorical_column_the_model_asked_for():
    cols = ["region_id", "season", "lead_time_days"]
    X = inference._prep(_frame(), cols, categorical={"region_id", "season"})
    # The bug: this column used to come back 100% NaN.
    assert X["region_id"].notna().all()
    assert str(X["region_id"].dtype) == "category"


def test_prep_refuses_when_a_populated_column_is_destroyed_by_coercion():
    """The exact 2026-09-19 failure: region_id present and populated, but not named as
    categorical, so numeric coercion wipes it. That must raise, not sail through."""
    cols = ["region_id", "season", "lead_time_days"]
    with pytest.raises(ValueError, match="region_id"):
        inference._prep(_frame(), cols, categorical={"state_id", "season"})


def test_a_retired_noncausal_feature_is_refused_even_as_missing_data():
    """Old artifacts can ask for forecast_error_lag, but current feature engineering must
    never fabricate it. Serving such an artifact would be a train/serve contract mismatch,
    so the model is refused rather than filled with NaN."""
    with pytest.raises(ValueError, match="forecast_error_lag"):
        inference._prep(_frame(), ["forecast_error_lag"], categorical=set())


def test_a_retired_label_derived_feature_is_refused():
    with pytest.raises(ValueError, match="historical_bust_frequency_region_season"):
        inference._prep(_frame(), ["historical_bust_frequency_region_season"], categorical=set())


def test_a_genuinely_absent_column_is_still_allowed_to_be_nan():
    """Not every NaN is a bug. Soil moisture stops at day 3 and wind at day 5 in the
    reforecast archive, so a column the frame simply does not carry stays NaN by design -
    only destroying data that *was* there is refused."""
    cols = ["region_id", "season", "lead_time_days", "pred_err_soil_moisture_pct"]
    X = inference._prep(_frame(), cols, categorical={"region_id", "season"})
    assert X["pred_err_soil_moisture_pct"].isna().all()
    assert X["region_id"].notna().all()


def test_an_all_null_input_column_is_not_mistaken_for_destruction():
    """A column present but entirely null going in is missing data, not coercion damage,
    so it must not trip the refusal."""
    df = _frame()
    df["pred_err_rainfall_mm"] = np.nan
    cols = ["region_id", "season", "pred_err_rainfall_mm"]
    X = inference._prep(df, cols, categorical={"region_id", "season"})
    assert X["pred_err_rainfall_mm"].isna().all()
