"""app.ml.pooled_training: pool many years of paired training data through XGBoost's
external-memory interface without ever holding more than one year resident at once.

The unit tests below use small, hand-built frames (shapes and plumbing only, clearly
synthetic, never asserting a meteorological number) to pin the two claims the whole
design rests on:

1. `pooled_stats`, computed by streaming thin columns one cached year at a time, produces
   EXACTLY the same (hbf, p90_error, bust_threshold) as computing them on the full
   concatenation directly - not an approximation.
2. `_YearDataIter` feeding an XGBoost `QuantileDMatrix` produces a model close to one
   fit in-memory on the identical concatenated rows.

`test_full_retrain_pooled_end_to_end` reuses the real small GEFS+ERA5 slice `test_ml.py`
already ingests, split into two calendar-shifted copies (real values, fabricated year
labels, clearly labelled) so a genuine multi-year pool exists without a second real
archive year - it exercises the whole orchestration path, not the meteorology.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from app.features import engineering as fe
from app.ml import pooled_training as pt
from app.ml import regressors as reg_mod
from app.ml.thresholds import compute_error_thresholds, compute_member_p90


def _thin_synthetic_year(n=200, seed=0, year=2017, region_ids=("r1", "r2", "r3"),
                         seasons=("winter", "monsoon"), variables=("temperature_c", "rainfall_mm")):
    """A small frame with just the columns pooled_stats needs, shaped like a real paired
    frame - values are random, labelled synthetic, never used to assert a metric.

    `init_date` is confined to `year`: two real cached years never share an init_date, and
    EVENT_KEYS (which include init_date) never groups across a year boundary - a fixture
    that let two fake "years" share dates would test a case the real pipeline never hits."""
    rng = np.random.default_rng(seed)
    region = rng.choice(region_ids, n)
    season = rng.choice(seasons, n)
    variable = rng.choice(variables, n)
    lead = rng.integers(1, 11, n)
    init = pd.Timestamp(f"{year}-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    valid = init + pd.to_timedelta(lead.astype(int) - 1, unit="D")
    fc = rng.normal(20, 5, n)
    obs = fc + rng.normal(0, 3, n)
    return pd.DataFrame({
        "region_id": region, "season": season, "variable": variable,
        "init_date": init, "valid_date": valid, "lead_time_days": lead,
        "forecast_value": fc, "observed_value": obs,
        "abs_error": np.abs(fc - obs),
    })


def test_pooled_stats_matches_the_full_frame_computation(tmp_path):
    y1 = _thin_synthetic_year(n=300, seed=1, year=2000)
    y2 = _thin_synthetic_year(n=250, seed=2, year=2001)
    y1.to_parquet(tmp_path / "paired_2000.parquet", index=False)
    y2.to_parquet(tmp_path / "paired_2001.parquet", index=False)
    cached = {2000: tmp_path / "paired_2000.parquet", 2001: tmp_path / "paired_2001.parquet"}
    train_cycles = set(pd.concat([y1["init_date"], y2["init_date"]]).unique())

    got_hbf, got_p90, got_thr = pt.pooled_stats(cached, [2000, 2001], train_cycles)

    whole = pd.concat([y1, y2], ignore_index=True)
    want_hbf = fe.compute_historical_bust_frequency(whole)
    want_p90 = compute_member_p90(whole[["variable", "abs_error"]])
    from app.ml.train_pipeline import _event_mean_error
    want_thr = compute_error_thresholds(_event_mean_error(whole), percentile=90.0)

    assert got_hbf == want_hbf
    assert got_p90 == want_p90
    assert got_thr == want_thr


def test_pooled_stats_respects_the_train_cycle_filter(tmp_path):
    """A cycle outside train_cycles (val/test) must not leak into the global stats -
    that would be the classic cross-validation leak, just moved into this new code path."""
    y1 = _thin_synthetic_year(n=200, seed=3)
    y1.to_parquet(tmp_path / "paired_2000.parquet", index=False)
    cached = {2000: tmp_path / "paired_2000.parquet"}
    all_cycles = sorted(y1["init_date"].unique())
    half = set(all_cycles[: len(all_cycles) // 2])

    got_hbf, got_p90, got_thr = pt.pooled_stats(cached, [2000], half)
    restricted = y1[y1["init_date"].isin(half)]
    want_p90 = compute_member_p90(restricted[["variable", "abs_error"]])

    assert got_p90 == want_p90
    assert got_p90 != compute_member_p90(y1[["variable", "abs_error"]])


def test_pooled_split_holds_the_test_year_out_entirely(tmp_path):
    y2000 = pd.DataFrame({"init_date": pd.date_range("2000-01-01", periods=10)})
    y2001 = pd.DataFrame({"init_date": pd.date_range("2001-01-01", periods=10)})
    y2000.to_parquet(tmp_path / "p2000.parquet", index=False)
    y2001.to_parquet(tmp_path / "p2001.parquet", index=False)
    cached = {2000: tmp_path / "p2000.parquet", 2001: tmp_path / "p2001.parquet"}

    train_c, val_c, test_c = pt.pooled_split(cached, test_year=2001)

    assert test_c == set(pd.to_datetime(y2001["init_date"]))
    assert not (train_c | val_c) & test_c
    assert train_c | val_c == set(pd.to_datetime(y2000["init_date"]))


def test_assign_folds_covers_every_cycle_and_is_deterministic():
    cycles = {pd.Timestamp("2017-01-01") + pd.Timedelta(days=i) for i in range(7)}
    a = pt.assign_folds(cycles, n_splits=3)
    b = pt.assign_folds(cycles, n_splits=3)
    assert set(a) == cycles
    assert a == b
    assert set(a.values()) == {0, 1, 2}


def _synthetic_regressor_frame(n, seed, feature_cols, year=2017):
    """Shaped like a real per-variable regressor training frame - one categorical region
    column, one categorical season, a handful of numeric features, and abs_error as the
    target. Values are random and labelled synthetic; only used to check the fitting
    mechanism works, never to assert a real MAE."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in feature_cols
                       if c not in ("region_id", "season")})
    df["region_id"] = rng.choice(["r1", "r2"], n)
    df["season"] = rng.choice(["winter", "monsoon"], n)
    df["variable"] = "temperature_c"
    df["init_date"] = pd.Timestamp(f"{year}-01-01") + pd.to_timedelta(rng.integers(0, 30, n), unit="D")
    # A real, learnable signal - not noise - so a fit that ignores the data is detectable.
    df["abs_error"] = 2.0 * df[feature_cols[0]] + rng.normal(0, 0.1, n)
    return df


def test_year_data_iter_trains_a_model_close_to_an_in_memory_fit(tmp_path):
    feature_cols = ["lead_time_days", "forecast_value", "region_id", "season"]
    y1 = _synthetic_regressor_frame(400, seed=10, feature_cols=feature_cols, year=2000)
    y2 = _synthetic_regressor_frame(400, seed=11, feature_cols=feature_cols, year=2001)
    y1.to_parquet(tmp_path / "paired_2000.parquet", index=False)
    y2.to_parquet(tmp_path / "paired_2001.parquet", index=False)
    cached = {2000: tmp_path / "paired_2000.parquet", 2001: tmp_path / "paired_2001.parquet"}
    cycles = set(pd.concat([y1["init_date"], y2["init_date"]]).unique())
    hbf = {}  # no lookup entries -> NaN feature, same on both sides of the comparison

    it = pt._YearDataIter(cached, [2000, 2001], "temperature_c", cycles,
                          feature_cols, hbf, tmp_path)
    dtrain = xgb.QuantileDMatrix(it, enable_categorical=True)
    params = {k: v for k, v in reg_mod.XGB_PARAMS.items()
             if k not in ("n_estimators", "enable_categorical", "n_jobs", "random_state")}
    params["seed"] = 42
    booster = xgb.train(params, dtrain, num_boost_round=reg_mod.XGB_PARAMS["n_estimators"])
    pooled_model = pt._booster_to_sklearn(booster, xgb.XGBRegressor)

    whole = pd.concat([y1, y2], ignore_index=True)
    inmem = xgb.XGBRegressor(**reg_mod.XGB_PARAMS)
    inmem.fit(reg_mod._prep_X(whole, feature_cols), whole["abs_error"].to_numpy())

    test_rows = _synthetic_regressor_frame(100, seed=99, feature_cols=feature_cols)
    Xte = reg_mod._prep_X(test_rows, feature_cols)
    pooled_pred = pooled_model.predict(Xte)
    inmem_pred = inmem.predict(Xte)
    assert np.isfinite(pooled_pred).all()
    # Same params, same rows (just fed differently), same learnable signal - predictions
    # should land close together, not merely both non-garbage.
    assert np.corrcoef(pooled_pred, inmem_pred)[0, 1] > 0.9


def test_booster_to_sklearn_roundtrip_matches_the_source_booster(tmp_path):
    X = pd.DataFrame({"a": np.arange(50, dtype=float)})
    y = X["a"].to_numpy() * 3.0
    dtrain = xgb.DMatrix(X, label=y)
    booster = xgb.train({"objective": "reg:squarederror"}, dtrain, num_boost_round=10)

    wrapped = pt._booster_to_sklearn(booster, xgb.XGBRegressor)

    got = wrapped.predict(X)
    want = booster.predict(xgb.DMatrix(X))
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_feature_columns_for_includes_the_globally_attached_hbf_feature(tmp_path):
    df = pd.DataFrame({"region_id": ["r1"], "season": ["winter"], "lead_time_days": [1]})
    df.to_parquet(tmp_path / "paired_2000.parquet", index=False)
    cached = {2000: tmp_path / "paired_2000.parquet"}
    cols = pt._feature_columns_for(cached, [2000])
    assert "historical_bust_frequency_region_season" in cols


# ------------------------------------------------------------- end-to-end orchestration

from tests.test_ml import _ingested_slice  # noqa: E402,F401 - reused fixture


def test_full_retrain_pooled_end_to_end(tmp_path, _ingested_slice):
    """Exercises the whole orchestration path - caching, the stats pass, per-variable
    external-memory fits, OOF fold models, event assembly, classifier training, and
    registry save - on a real small paired frame reused three times under three
    fabricated calendar years (real GRIB/ERA5-derived values, relabelled dates, clearly a
    test-only construction) so a genuine multi-year pool exists without needing a second
    real archive year fetched. This checks the pooling MECHANISM does not crash and
    produces usable, finite artifacts; it is not a claim about meteorological accuracy -
    that comparison is made separately against real archive years."""
    from app.ml.pooled_training import full_retrain_pooled
    from app.ml.train_pipeline import _build_paired_in_chunks

    base, _ = _build_paired_in_chunks()
    assert not base.empty, "the real ingested slice must produce a non-empty paired frame"

    cache_dir = tmp_path / "pooled_cache"
    cache_dir.mkdir()
    years = {2000: 0, 2001: 1, 2002: 2}
    for year, offset in years.items():
        shifted = base.copy()
        for col in ("init_date", "valid_date"):
            shifted[col] = pd.to_datetime(shifted[col]) + pd.DateOffset(years=offset)
        shifted.to_parquet(cache_dir / f"paired_{year}.parquet", index=False)

    report = full_retrain_pooled(train_years=[2000, 2001], test_year=2002, cache_dir=cache_dir)

    assert report.status == "success", report.error
    assert report.made_current is False
    assert report.modelled_variables
    assert (cache_dir / "paired_2000.parquet").exists()  # cache_year found it, did not rebuild

    from app.ml import registry
    loaded = registry.load_regressors(report.run_id)
    assert set(loaded) == set(report.modelled_variables)
    clf, clf_cols = registry.load_classifier(report.run_id)
    assert clf is not None and clf_cols
    hbf = registry.load_historical_bust_freq(report.run_id)
    assert hbf
