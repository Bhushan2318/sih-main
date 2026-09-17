"""The three frozen contracts.

Five people building in parallel against an unwritten schema invent five incompatible
ones. These tests exist so that a change to any of the three shared shapes fails loudly
here, in one place, rather than quietly downstream in someone else's workstream.

Frozen here, and deliberately nothing else:
  1. the paired-row frame  - features (C) joined to labels (B), consumed by the trainer
  2. /api/model/status     - D writes the metrics, F reads them
  3. the region panel      - C writes the SHAP factors, F renders them

What these tests do NOT do is retype the live response models. FastAPI filters a response
to its model, so tightening `list` to `list[TopFactor]` on a served endpoint can silently
drop fields the frontend already reads. Freezing means failing on drift, not rewriting a
working payload.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app import contracts
from app.api import schemas


# --------------------------------------------------------------- 1. paired rows

def test_paired_row_columns_are_frozen():
    """The exact column set, in order, as produced by build_training_frame and measured
    against the real 2017 store on 2026-09-10 (one cycle, 10,695 rows).

    C1 (2026-09-16) added the four jump_* columns; order re-printed from a real frame
    built over the serving store's 2026-09-10 cycle.

    If this fails you either added a feature - update the contract in the same commit and
    say so in the message - or something upstream changed shape without meaning to.
    """
    assert contracts.PAIRED_ROW_COLUMNS == (
        "region_id", "variable", "valid_date", "forecast_value", "value_type",
        "init_date", "lead_time_days", "ensemble_member_id", "observed_value",
        "verification_status", "abs_error", "jump_abs_change", "jump_std",
        "jump_sign_flips", "jump_rel_climatology", "month", "season", "ensemble_spread",
        "ensemble_member_count", "pressure_rate_of_change", "moisture_rate_of_change",
        "forecast_error_lag", "fc_atmospheric_moisture_kgm2", "fc_humidity_pct",
        "fc_pressure_hpa", "fc_rainfall_mm", "fc_soil_moisture_pct", "fc_temperature_c",
        "fc_wind_direction_deg", "fc_wind_speed_ms",
        "historical_bust_frequency_region_season",
    )


def test_the_label_is_not_a_paired_row_column():
    """y_bust is applied after the split, because the bust threshold is computed on
    training data only (CLAUDE.md). A label inside the paired frame would mean the
    threshold had already seen the held-out rows."""
    assert "y_bust" not in contracts.PAIRED_ROW_COLUMNS


def test_event_keys_identify_a_row_and_are_all_present():
    for key in ("region_id", "init_date", "valid_date", "lead_time_days",
                "ensemble_member_id"):
        assert key in contracts.PAIRED_ROW_COLUMNS


def _conforming_frame(n: int = 3) -> pd.DataFrame:
    """A frame built to the contract. Values are arbitrary - the shape is the point."""
    return pd.DataFrame({c: contracts.example_column(c, n)
                         for c in contracts.PAIRED_ROW_COLUMNS})


def test_a_conforming_frame_validates():
    contracts.validate_paired_frame(_conforming_frame())


def test_a_missing_column_is_rejected():
    df = _conforming_frame().drop(columns=["ensemble_spread"])
    with pytest.raises(ValueError, match="ensemble_spread"):
        contracts.validate_paired_frame(df)


def test_an_unexpected_column_is_rejected():
    df = _conforming_frame()
    df["some_new_idea"] = 1.0
    with pytest.raises(ValueError, match="some_new_idea"):
        contracts.validate_paired_frame(df)


def test_a_retyped_column_is_rejected():
    """forecast_value arriving as text is the classic silent break: it survives a merge
    and poisons every metric downstream."""
    df = _conforming_frame()
    df["forecast_value"] = df["forecast_value"].astype(str)
    with pytest.raises(ValueError, match="forecast_value"):
        contracts.validate_paired_frame(df)


def test_column_order_is_not_enforced():
    """Order is documented, not required - a reordered frame is still usable, and
    failing on it would be brittle for no gain."""
    df = _conforming_frame()
    contracts.validate_paired_frame(df[list(reversed(contracts.PAIRED_ROW_COLUMNS))])


# --------------------------------------------------------------- 2. model status

def test_model_status_required_fields_are_frozen():
    """D writes these, F reads them. Removing or renaming one breaks the dashboard
    silently - it renders an empty panel rather than an error."""
    fields = set(schemas.ModelStatusResponse.model_fields)
    for required in ("model_trained", "current_run_id", "training_data",
                     "validation_metrics", "thresholds", "baselines",
                     "modelled_variables", "explanation_method"):
        assert required in fields, f"/api/model/status lost {required}"


def test_model_status_tolerates_an_untrained_box():
    """Every consumer must handle the no-model case; it is the state a fresh deploy is
    in before the first artifact lands."""
    m = schemas.ModelStatusResponse(model_trained=False)
    assert m.current_run_id is None
    assert m.training_data == {} and m.validation_metrics == {}


# --------------------------------------------------------------- 3. region panel

def test_region_panel_required_fields_are_frozen():
    fields = set(schemas.RegionDetailResponse.model_fields)
    for required in ("region_id", "model_trained", "variables",
                     "bust_probability_curve", "top_factors", "top_factors_method"):
        assert required in fields, f"the region panel lost {required}"


def test_shap_factor_shape_is_frozen():
    """The 'what drove this prediction' panel is a shipped feature (CLAUDE.md rule 9).
    Anything that changes this shape breaks it."""
    f = schemas.TopFactor(feature="ensemble_spread", importance=0.31, method="shap")
    assert set(schemas.TopFactor.model_fields) == {"feature", "importance", "method"}
    assert f.method == "shap"


def test_region_panel_survives_a_region_with_no_data():
    r = schemas.RegionDetailResponse(region_id="IN-KL-IDUKKI", model_trained=True)
    assert r.variables == [] and r.top_factors == []
    assert r.message is None


# --------------------------------------------------------------- against real data

def _one_real_cycle(pick: str):
    """A paired frame built from the real store, or None when there is no store.

    conftest.py redirects the data directories to a temp path on import, so this is
    normally None under pytest. It is the check that actually matters, though: the
    synthetic tests above prove the validator works, this proves the contract matches
    what the trainer is really handed.
    """
    import pandas as pd
    from app.storage import parquet_store
    from app.features import engineering as fe
    from app.ml.train_pipeline import _TRAINING_COLUMNS

    inits = parquet_store.read_dataset(
        value_types=["forecast"], columns=["init_date"], dedupe=False,
    )["init_date"].dropna().unique()
    if len(inits) == 0:
        return None
    inits = sorted(pd.to_datetime(pd.Series(inits)).dt.normalize().unique())
    c = pd.Timestamp(inits[0] if pick == "first" else inits[-1]).date()
    fc = parquet_store.read_dataset(value_types=["forecast"], columns=_TRAINING_COLUMNS,
                                    init_dates=[c], exclude_provisional=True)
    ob = parquet_store.read_dataset(
        value_types=["observed"], columns=_TRAINING_COLUMNS,
        valid_date_min=(pd.Timestamp(c) - pd.Timedelta(days=3)).date(),
        valid_date_max=(pd.Timestamp(c) + pd.Timedelta(days=13)).date(),
        exclude_provisional=True)
    if fc.empty:
        return None
    return fe.build_training_frame(pd.concat([fc, ob], ignore_index=True),
                                   historical_bust_freq=None)


@pytest.mark.parametrize("pick", ["first", "last"])
def test_a_real_paired_frame_conforms(pick):
    """Verified by hand against the 2017 store on 2026-09-10: 10,695 rows for both
    2017-01-01 and 2017-12-31, each conforming. Two cycles from opposite ends of the
    year, so a seasonal column cannot pass by luck."""
    frame = _one_real_cycle(pick)
    if frame is None or frame.empty:
        pytest.skip("no canonical store here - synthetic tests above still apply")
    contracts.validate_paired_frame(frame)
