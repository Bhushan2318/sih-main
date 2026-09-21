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
    pytest.importorskip("torch")  # full_retrain_pooled's GPU/CPU split unconditionally
    # imports torch; requirements-train.txt only, not installed by setup.yml's test step.
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


# --------------------------------------------------------------- attach_hbf_column
# Real crash 2026-09-17: the previous implementation cast region_id/season to a dense
# per-row string array via .astype(str) - fine on pooled_stats's thin frame, but on the
# FULL wide per-year training frame (already resident, 76.5M rows for a dense year) that
# needed an 8.27 GiB contiguous allocation and hit ArrayMemoryError. The fix works in
# category-code space instead - these tests pin correctness against a naive per-row dict
# lookup (the actual spec, independent of either implementation), not against the old
# merge's own output.

def _naive_hbf_lookup(df, hbf, out_col):
    """The obviously-correct, obviously-slow reference: one dict lookup per row."""
    return [hbf.get((r, s), np.nan)
           for r, s in zip(df["region_id"].astype(str), df["season"].astype(str))]


def test_attach_hbf_column_matches_a_naive_dict_lookup():
    df = pd.DataFrame({
        "region_id": pd.Categorical(["r1", "r2", "r1", "r3", "r2"]),
        "season": pd.Categorical(["winter", "winter", "monsoon", "monsoon", "monsoon"]),
    })
    hbf = {("r1", "winter"): 0.1, ("r2", "winter"): 0.2, ("r1", "monsoon"): 0.3,
          ("r2", "monsoon"): 0.4}
    want = _naive_hbf_lookup(df, hbf, "hbf")
    got = pt.attach_hbf_column(df.copy(), hbf, out_col="hbf")["hbf"].tolist()
    assert got == pytest.approx(want, nan_ok=True)


def test_attach_hbf_column_missing_keys_are_nan_not_zero():
    """A (region, season) combination with no historical bust frequency is missing
    signal, not zero risk - CLAUDE.md rule 3: missing never becomes zero."""
    df = pd.DataFrame({"region_id": pd.Categorical(["r1", "r2"]),
                       "season": pd.Categorical(["winter", "summer"])})
    hbf = {("r1", "winter"): 0.5}  # r2/summer is not in the table
    out = pt.attach_hbf_column(df, hbf, out_col="hbf")["hbf"]
    assert out.iloc[0] == pytest.approx(0.5)
    assert np.isnan(out.iloc[1])


def test_attach_hbf_column_empty_hbf_is_all_nan():
    df = pd.DataFrame({"region_id": pd.Categorical(["r1", "r2"]),
                       "season": pd.Categorical(["winter", "summer"])})
    out = pt.attach_hbf_column(df, {}, out_col="hbf")["hbf"]
    assert out.isna().all()


def test_attach_hbf_column_works_on_non_categorical_columns_too():
    """A hand-built test frame (or any caller) may hand this plain object-dtype
    columns, not just the Categorical the real pipeline always produces."""
    df = pd.DataFrame({"region_id": ["r1", "r2"], "season": ["winter", "monsoon"]})
    hbf = {("r1", "winter"): 0.7}
    out = pt.attach_hbf_column(df, hbf, out_col="hbf")["hbf"]
    assert out.iloc[0] == pytest.approx(0.7)
    assert np.isnan(out.iloc[1])


def test_attach_hbf_column_row_count_and_order_are_preserved():
    """The output must line up positionally with the input - the historical bug class
    this whole function exists to avoid is a lookup that silently reindexes."""
    df = pd.DataFrame({
        "region_id": pd.Categorical(["r3", "r1", "r2", "r1"]),
        "season": pd.Categorical(["monsoon", "winter", "winter", "monsoon"]),
    })
    hbf = {("r1", "winter"): 0.1, ("r2", "winter"): 0.2, ("r3", "monsoon"): 0.3,
          ("r1", "monsoon"): 0.4}
    out = pt.attach_hbf_column(df, hbf, out_col="hbf")["hbf"].tolist()
    assert out == pytest.approx([0.3, 0.1, 0.2, 0.4])


def test_attach_hbf_column_handles_a_large_row_count_without_a_dense_string_array():
    """Not a memory-ceiling test (pytest cannot assert that portably) - a scale smoke
    test: many rows, few distinct categories, the exact shape of the real crash
    (76.5M rows, 666 districts). If this silently regressed back to .astype(str) on the
    full column, it would still pass at this size, but the whole point is that the
    lookup-table approach's cost is O(distinct values), not O(rows) - this at least
    exercises that path for real rather than only ever at hand-built sizes of 2-5 rows."""
    n = 2_000_000
    rng = np.random.default_rng(0)
    region_ids = [f"r{i}" for i in range(666)]
    seasons = ["winter", "summer", "monsoon", "post-monsoon"]
    df = pd.DataFrame({
        "region_id": pd.Categorical(rng.choice(region_ids, size=n)),
        "season": pd.Categorical(rng.choice(seasons, size=n)),
    })
    hbf = {(r, s): float(i % 100) / 100
          for i, (r, s) in enumerate((r, s) for r in region_ids for s in seasons)}
    out = pt.attach_hbf_column(df, hbf, out_col="hbf")["hbf"]
    assert len(out) == n
    assert out.notna().all()
    # Spot-check against the naive reference on a small random sample, not all 2M rows.
    sample = df.sample(500, random_state=0)
    want = _naive_hbf_lookup(sample, hbf, "hbf")
    assert out.loc[sample.index].tolist() == pytest.approx(want)


# --- GPU detection must not require PyTorch ------------------------------------------
# full_retrain_pooled imported torch unconditionally just to ask whether a GPU exists.
# XGBoost needs no torch, and the function already had a "no GPU -> everything on CPU"
# branch - but a machine without torch (CI's core install, any XGBoost-only environment)
# crashed with ModuleNotFoundError before reaching it.

def test_cuda_probe_reports_no_gpu_when_torch_is_not_installed(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    assert pt._cuda_available() is False


def test_cuda_probe_is_a_plain_bool_when_torch_is_present():
    torch = pytest.importorskip("torch")
    got = pt._cuda_available()
    assert isinstance(got, bool)
    assert got == bool(torch.cuda.is_available())


def test_cuda_probe_honours_pooled_force_cpu(monkeypatch):
    """POOLED_FORCE_CPU=1 overrides the real answer to False, even with a real GPU
    present - added 2026-09-18 to test whether GPU/CUDA paths contribute to a real
    memory-pressure signature observed mid-run, without touching the CUDA probe itself."""
    monkeypatch.setenv("POOLED_FORCE_CPU", "1")
    assert pt._cuda_available() is False


def test_cuda_probe_ignores_pooled_force_cpu_when_unset(monkeypatch):
    monkeypatch.delenv("POOLED_FORCE_CPU", raising=False)
    torch = pytest.importorskip("torch")
    assert pt._cuda_available() == bool(torch.cuda.is_available())


def test_cache_year_rebuilds_a_truncated_cache_file(tmp_path, monkeypatch):
    """A cache file is only valid if its footer reads. Real crash 2026-09-21: killing the
    trainer mid-write left paired_2000.parquet at 1.59 GB of an expected 3.65 GB, and
    every later run trusted it because it existed - the whole pooled run died nine hours
    later in pooled_split, after re-caching seventeen other years."""
    from app.ml import pooled_training as pt

    truncated = tmp_path / "paired_2000.parquet"
    truncated.write_bytes(b"PAR1" + b"\x00" * 64)  # header, no footer

    built = {}

    def fake_build(init_date_min=None, init_date_max=None):
        built["called"] = True
        return pd.DataFrame({"init_date": [pd.Timestamp("2000-01-01")], "x": [1.0]}), 1

    monkeypatch.setattr(pt, "_build_paired_in_chunks", fake_build)
    out = pt.cache_year(2000, tmp_path)

    assert built.get("called"), "a truncated cache file was trusted instead of rebuilt"
    assert pd.read_parquet(out)["x"].tolist() == [1.0]


def test_cache_year_writes_atomically(tmp_path, monkeypatch):
    """A killed write must leave no file at all, never a partial one at the real path."""
    from app.ml import pooled_training as pt

    def exploding_build(init_date_min=None, init_date_max=None):
        raise KeyboardInterrupt("killed mid-build")

    monkeypatch.setattr(pt, "_build_paired_in_chunks", exploding_build)
    with pytest.raises(KeyboardInterrupt):
        pt.cache_year(2001, tmp_path)
    assert not (tmp_path / "paired_2001.parquet").exists()


def _reference_pooled_stats(frames, train_cycles):
    """The pre-2026-09-21 implementation, kept only as an oracle for the streaming one."""
    from app.features import engineering as fe
    from app.ml.thresholds import compute_member_p90
    thin_frames = []
    for df in frames:
        df = df[df["init_date"].isin(train_cycles)]
        if df.empty:
            continue
        thin = df[["region_id", "season", "variable", "abs_error"]].copy()
        thin["region_id"] = thin["region_id"].astype(str)
        thin["season"] = thin["season"].astype(str)
        thin_frames.append(thin)
    thin = pd.concat(thin_frames, ignore_index=True)
    return fe.compute_historical_bust_frequency(thin), compute_member_p90(
        thin[["variable", "abs_error"]])


def test_pooled_stats_matches_the_frame_building_implementation(tmp_path):
    """Streaming must not change a single number. At 17 years the old path concatenated
    ~0.8 billion rows, cast region_id to a <U29 array (8.07 GiB, the real crash on
    2026-09-21) and then walked it with itertuples."""
    from app.ml import pooled_training as pt

    rng = np.random.default_rng(11)
    cycles = pd.to_datetime(["2000-01-01", "2000-01-02", "2001-01-01"])
    frames, paths = [], {}
    for i, year in enumerate((2000, 2001)):
        n = 400
        df = pd.DataFrame({
            "init_date": rng.choice(cycles, n),
            "region_id": pd.Categorical(rng.choice([f"IND.{k}" for k in range(5)], n)),
            "season": pd.Categorical(rng.choice(["DJF", "JJAS"], n)),
            "variable": pd.Categorical(rng.choice(["temperature_c", "rainfall_mm"], n)),
            "abs_error": np.where(rng.random(n) < 0.1, np.nan, rng.gamma(2, 2, n)),
            "forecast_value": rng.normal(size=n),
            "observed_value": rng.normal(size=n),
            "lead_time_days": rng.integers(1, 11, n),
            "valid_date": rng.choice(cycles, n),
        })
        p = tmp_path / f"paired_{year}.parquet"
        df.to_parquet(p, index=False)
        frames.append(df)
        paths[year] = p

    train_cycles = set(cycles[:2])
    want_hbf, want_p90 = _reference_pooled_stats(frames, train_cycles)
    got_hbf, got_p90, _ = pt.pooled_stats(paths, [2000, 2001], train_cycles)

    assert set(got_hbf) == set(want_hbf)
    for k in want_hbf:
        assert got_hbf[k] == pytest.approx(want_hbf[k], rel=1e-9, abs=1e-12)
    assert set(got_p90) == set(want_p90)
    for k in want_p90:
        assert got_p90[k] == pytest.approx(want_p90[k], rel=1e-6)
