"""No model input may be built from an observation made after the forecast was issued.

`forecast_error_lag` was exactly that: the realised error of the same forecast one lead
earlier. Every training and held-out row had it, and no live forecast can, because the
observation arrives days after issue. In training the lag was missing only on Day 1, so the
model learned that a missing lag means a Day-1-sized error. Live it is missing at every
lead, so every lead was scored like Day 1. Measured 2026-09-28 on September 2017 (666
districts, 10 cycles, a year the served model never trained on):

- ROC-AUC was 0.797 with the lag filled in and 0.699 scored as live.
- Day 1 was 0.840 either way, because the lag never exists there.
- The live bust probability fell from 0.399 on Day 1 to 0.313 on Day 10, while the real
  bust rate rose from 0.371 to 0.495.

See docs/known-issues.md. These tests pin the rule where the regressors choose their inputs,
so a later feature cannot bring the same leak back without failing here first.
"""
import pandas as pd
import pytest

from app import contracts
from app.ml import regressors as reg_mod


def _every_paired_column() -> pd.DataFrame:
    return pd.DataFrame(columns=list(contracts.PAIRED_ROW_COLUMNS))


def test_regressors_take_no_input_built_from_observations():
    cols = reg_mod.feature_columns(_every_paired_column())
    leaked = sorted(set(cols) & contracts.OBSERVATION_DERIVED)
    assert not leaked, f"model inputs built from observations after issue time: {leaked}"


def test_an_input_built_from_observations_is_refused(monkeypatch):
    monkeypatch.setattr(reg_mod, "NUMERIC_FEATURES", [*reg_mod.NUMERIC_FEATURES, "forecast_error_lag"])
    with pytest.raises(ValueError, match="forecast_error_lag"):
        reg_mod.feature_columns(_every_paired_column())


def test_as_of_issue_blanks_the_lag_and_keeps_what_the_label_needs():
    from scripts.score_run_on_year import as_of_issue

    paired = pd.DataFrame({
        "forecast_error_lag": [1.5, 2.0], "abs_error": [0.4, 3.1],
        "observed_value": [20.0, 21.0], "forecast_value": [19.6, 24.1],
    })
    out = as_of_issue(paired.copy())
    assert out["forecast_error_lag"].isna().all()
    # the label is built from abs_error/observed_value: those must come through untouched
    pd.testing.assert_frame_equal(out.drop(columns="forecast_error_lag"),
                                  paired.drop(columns="forecast_error_lag"))
