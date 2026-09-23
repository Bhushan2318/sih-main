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

import json

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


def test_val_frame_worker_spills_by_variable_and_parent_reads_one(tmp_path):
    """The parent must never allocate the validation frame: it needs to be lean when it
    spawns a training worker that wants one ~9.19 GB allocation for seventeen years. With
    the frame built in-parent it sat at 16.8 GB of 23.7 GB and all eight variables failed
    that malloc (2026-09-21). Runs the real worker, so an interface mistake is caught."""
    from app.ml import pooled_training as pt

    rng = np.random.default_rng(5)
    n = 240
    cycles = pd.to_datetime(["2016-01-01", "2016-01-02", "2016-01-03"])
    df = pd.DataFrame({
        "init_date": rng.choice(cycles, n),
        "valid_date": rng.choice(cycles, n),
        "lead_time_days": rng.integers(1, 11, n),
        "region_id": pd.Categorical(rng.choice(["IND.1", "IND.2"], n)),
        "season": pd.Categorical(rng.choice(["DJF", "JJAS"], n)),
        "variable": pd.Categorical(rng.choice(["temperature_c", "rainfall_mm"], n)),
        "abs_error": rng.gamma(2, 2, n),
        "forecast_value": rng.normal(size=n),
        "observed_value": rng.normal(size=n),
    })
    path = tmp_path / "paired_2016.parquet"
    df.to_parquet(path, index=False)

    spill = tmp_path / "_va_by_variable"
    hbf = {("IND.1", "DJF"): 0.25, ("IND.2", "JJAS"): 0.5}
    res = pt._run_val_frame_subprocess({2016: path}, [2016], set(cycles),
                                       set(df.columns), hbf, spill)

    assert res.get("error") is None, res.get("error")
    assert res["n_rows"] == n

    one = pt._read_va_var(spill, "temperature_c")
    assert not one.empty
    assert set(one["variable"].unique()) == {"temperature_c"}
    assert len(one) == int((df["variable"] == "temperature_c").sum())
    assert one.index.name == "_va_row"
    assert one.index.max() < n
    assert "historical_bust_frequency_region_season" in one.columns
    assert pt._read_va_var(spill, "not_a_variable").empty


# --- every variable on the GPU, and a caught worker failure is retried ----------------
# Real failure 2026-09-21, 17-year pool: the four variables trained on CUDA all finished,
# and every variable handed to the CPU half hit Windows' commit limit (python.exe at
# 60-62 GB of virtual memory, System event 2004 at 16:28, 17:37 and 18:43 UTC). The
# GPU/CPU split existed only so the two halves could run concurrently, which was removed
# on 2026-09-17, so half the variables were being sent to the device that cannot hold
# them for no remaining reason. A MemoryError the worker caught came back as a pickled
# result and was treated as a finished attempt - the variable was dropped without a retry,
# and nothing was printed until the whole run ended.

def _fake_worker(results, seen_devices):
    calls = iter(results)

    def fake(script, job):
        seen_devices.append(job["device"])
        return next(calls)
    return fake


def _ok_result(art="ART"):
    return {"artifact": art, "skipped": None, "val_pred": None, "fold_models": {}, "error": None}


def test_variable_worker_retries_a_caught_exception(monkeypatch, capsys):
    """Shapes-only: the worker results are hand-built dicts, no training happens."""
    seen = []
    caught = {"artifact": None, "skipped": "worker exception: MemoryError", "val_pred": None,
              "fold_models": {}, "error": "Traceback ...\nMemoryError"}
    monkeypatch.setattr(pt, "_run_worker_subprocess", _fake_worker([caught, _ok_result()], seen))
    result = pt._run_variable_subprocess({}, [2000], "wind_speed_ms", set(), pd.DataFrame(),
                                         {}, None, "cuda", {})
    assert result["artifact"] == "ART"
    assert seen == ["cuda", "cuda"]
    err = capsys.readouterr().err
    assert "wind_speed_ms attempt 1 device=cuda FAILED" in err
    assert "wind_speed_ms attempt 2 device=cuda ok" in err


def test_variable_worker_does_not_retry_a_legitimate_too_few_rows_skip(monkeypatch):
    seen = []
    too_few = {"artifact": None, "skipped": "regressor training returned None or too few rows",
               "val_pred": None, "fold_models": {}, "error": None}
    monkeypatch.setattr(pt, "_run_worker_subprocess", _fake_worker([too_few], seen))
    result = pt._run_variable_subprocess({}, [2000], "rainfall_mm", set(), pd.DataFrame(),
                                         {}, None, "cuda", {})
    assert result["artifact"] is None
    assert seen == ["cuda"]


def test_variable_worker_falls_back_to_cpu_after_two_cuda_failures(monkeypatch):
    seen = []
    crash = {"error": "worker produced no output, rc=3221225477: "}
    monkeypatch.setattr(pt, "_run_worker_subprocess",
                        _fake_worker([crash, crash, _ok_result()], seen))
    result = pt._run_variable_subprocess({}, [2000], "humidity_pct", set(), pd.DataFrame(),
                                         {}, None, "cuda", {})
    assert result["artifact"] == "ART"
    assert seen == ["cuda", "cuda", "cpu"]


def test_full_retrain_pooled_sends_every_variable_to_cuda_when_a_gpu_exists(
        tmp_path, _ingested_slice, monkeypatch):
    """Real paired slice, relabelled years (as in the end-to-end test above). The GPU probe
    is forced True; the real worker is still run, on CPU, so this passes on machines
    without a GPU - what it asserts is the device each variable was dispatched with."""
    from app.ml.train_pipeline import _build_paired_in_chunks

    base, _ = _build_paired_in_chunks()
    cache_dir = tmp_path / "pooled_cache"
    cache_dir.mkdir()
    for year, offset in {2000: 0, 2001: 1, 2002: 2}.items():
        shifted = base.copy()
        for col in ("init_date", "valid_date"):
            shifted[col] = pd.to_datetime(shifted[col]) + pd.DateOffset(years=offset)
        shifted.to_parquet(cache_dir / f"paired_{year}.parquet", index=False)

    dispatched = {}
    real = pt._run_variable_subprocess

    def spy(cached, train_years, variable, train_cycles, va_var, hbf, cdir, device, fold_of):
        dispatched[variable] = device
        return real(cached, train_years, variable, train_cycles, va_var, hbf, cdir, "cpu", fold_of)

    monkeypatch.setattr(pt, "_cuda_available", lambda: True)
    monkeypatch.setattr(pt, "_run_variable_subprocess", spy)
    report = pt.full_retrain_pooled(train_years=[2000, 2001], test_year=2002, cache_dir=cache_dir)

    assert report.status == "success", report.error
    assert len(dispatched) == base["variable"].nunique()
    assert set(dispatched.values()) == {"cuda"}


def test_variable_checkpoint_round_trips_and_is_keyed_on_the_split(tmp_path):
    """Shapes-only: the stored result is a hand-built dict."""
    cycles = {pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-02")}
    val = {pd.Timestamp("2000-02-01")}
    p = pt._variable_checkpoint_path(tmp_path, "temperature_c", [2000], cycles, val)
    assert pt._load_variable_checkpoint(p) is None
    pt._save_variable_checkpoint(p, _ok_result())
    assert pt._load_variable_checkpoint(p)["artifact"] == "ART"
    other = pt._variable_checkpoint_path(tmp_path, "temperature_c", [2000],
                                         cycles | {pd.Timestamp("2000-01-03")}, val)
    assert other != p and pt._load_variable_checkpoint(other) is None


def test_full_retrain_pooled_reuses_finished_variables_on_a_rerun(
        tmp_path, _ingested_slice, monkeypatch):
    """Real paired slice, relabelled years. The second run on the same pool must not
    train any variable again - each one comes back from its checkpoint."""
    from app.ml.train_pipeline import _build_paired_in_chunks

    base, _ = _build_paired_in_chunks()
    cache_dir = tmp_path / "pooled_cache"
    cache_dir.mkdir()
    for year, offset in {2000: 0, 2001: 1, 2002: 2}.items():
        shifted = base.copy()
        for col in ("init_date", "valid_date"):
            shifted[col] = pd.to_datetime(shifted[col]) + pd.DateOffset(years=offset)
        shifted.to_parquet(cache_dir / f"paired_{year}.parquet", index=False)

    monkeypatch.setattr(pt, "_cuda_available", lambda: False)
    first = pt.full_retrain_pooled(train_years=[2000, 2001], test_year=2002, cache_dir=cache_dir)
    assert first.status == "success", first.error

    calls = []
    real = pt._run_variable_subprocess
    monkeypatch.setattr(pt, "_run_variable_subprocess",
                        lambda *a, **k: calls.append(a[2]) or real(*a, **k))
    second = pt.full_retrain_pooled(train_years=[2000, 2001], test_year=2002, cache_dir=cache_dir)
    assert second.status == "success", second.error
    assert calls == []
    assert second.modelled_variables == first.modelled_variables
    assert second.regressor_metrics == first.regressor_metrics


def test_year_data_iter_reads_only_its_own_variable_from_disk(tmp_path, monkeypatch):
    """Shapes-only synthetic frame. Real failure 2026-09-21: the iterator read every
    variable of a 76.5M-row year and filtered in pandas, 17 years x every QuantileDMatrix
    pass; measured on real 2009+2010 caches, peak commit 29.87 GB and 16.58 GB left
    resident, against 9.38 GB / 4.82 GB with the filter pushed into the read - identical
    rows and identical abs_error sums. At 17 years the worker climbed past 43 GB of commit
    before its first boosting round."""
    df = _thin_synthetic_year(n=400, variables=("temperature_c", "rainfall_mm"))
    df["abs_error"] = np.abs(np.random.default_rng(1).normal(size=len(df)))
    df["lead_day"] = 1
    p = tmp_path / "paired_2017.parquet"
    df.to_parquet(p, index=False)

    calls = []
    real = pd.read_parquet
    monkeypatch.setattr(pt.pd, "read_parquet", lambda *a, **k: calls.append(k) or real(*a, **k))
    hbf = {(r, s): 0.1 for r in df["region_id"].unique() for s in df["season"].unique()}
    it = pt._YearDataIter({2017: p}, [2017], "rainfall_mm", set(df["init_date"]),
                          ["lead_day"], hbf, tmp_path)
    got = []
    it.next(lambda data, label: got.append(len(label)))
    assert calls and calls[0].get("filters") == [("variable", "==", "rainfall_mm")]
    assert got == [int((df["variable"] == "rainfall_mm").sum())]


# --- regressor fits on a bounded, evenly spread sample of training cycles -------------
# Measured 2026-09-21 on the real caches, one variable: XGBoost's QuantileDMatrix peaked
# at 17.32 GB of commit for one year and 26.11 GB for two (thread count made no
# difference: 25.89 GB at nthread=4), while the parquet iterator alone stayed flat at
# ~8.6 GB. Seventeen years at every-day density cannot fit this machine's 83 GB commit
# limit on any device. Every year stays in the pool; the per-variable fits see a
# bounded sample of its days, and everything else (thresholds, validation, test,
# classifier events) keeps every day.

def _daily_cycles(years):
    return {pd.Timestamp(d) for y in years for d in pd.date_range(f"{y}-01-01", f"{y}-12-31")}


def test_fit_cycles_keeps_a_small_pool_whole():
    cycles = _daily_cycles([2000])
    assert pt.fit_cycles(cycles, cap=2000) == cycles


def test_fit_cycles_caps_a_large_pool_and_spans_every_year_and_season():
    cycles = _daily_cycles(range(2000, 2017))
    got = pt.fit_cycles(cycles, cap=2000)
    assert len(got) == 2000 and got <= cycles
    assert got == pt.fit_cycles(cycles, cap=2000)  # deterministic
    assert {c.year for c in got} == set(range(2000, 2017))
    assert {c.month for c in got} == set(range(1, 13))
    per_year = pd.Series([c.year for c in got]).value_counts()
    assert per_year.min() >= 0.7 * per_year.mean()


def test_fit_cycles_leaves_every_oof_fold_with_training_cycles():
    """assign_folds numbers sorted cycles i % 3 - an every-third-day sample would put
    every kept cycle in one fold and leave that fold's model nothing to train on."""
    cycles = _daily_cycles(range(2000, 2017))
    fold_of = pt.assign_folds(cycles)
    got = pt.fit_cycles(cycles, cap=2000)
    per_fold = pd.Series([fold_of[c] for c in got]).value_counts()
    assert set(per_fold.index) == {0, 1, 2}
    assert per_fold.min() >= 0.25 * len(got)


def test_full_retrain_pooled_fits_regressors_on_the_capped_sample(
        tmp_path, _ingested_slice, monkeypatch):
    """Real paired slice, relabelled years. The cap is forced below the pool so the
    sample is exercised end to end; the worker must receive the sample, not the pool."""
    from app.ml.train_pipeline import _build_paired_in_chunks

    base, _ = _build_paired_in_chunks()
    cache_dir = tmp_path / "pooled_cache"
    cache_dir.mkdir()
    for year, offset in {2000: 0, 2001: 1, 2002: 2}.items():
        shifted = base.copy()
        for col in ("init_date", "valid_date"):
            shifted[col] = pd.to_datetime(shifted[col]) + pd.DateOffset(years=offset)
        shifted.to_parquet(cache_dir / f"paired_{year}.parquet", index=False)

    n_train = len(pt.pooled_split({y: cache_dir / f"paired_{y}.parquet" for y in (2000, 2001, 2002)},
                                  2002)[0])
    cap = max(2, n_train - 1)
    monkeypatch.setattr(pt, "MAX_FIT_CYCLES", cap)
    monkeypatch.setattr(pt, "_cuda_available", lambda: False)
    seen = []
    real = pt._run_variable_subprocess
    monkeypatch.setattr(pt, "_run_variable_subprocess",
                        lambda *a, **k: seen.append(len(a[3])) or real(*a, **k))
    report = pt.full_retrain_pooled(train_years=[2000, 2001], test_year=2002, cache_dir=cache_dir)
    assert report.status == "success", report.error
    assert report.split_cycles["fit"] == cap < report.split_cycles["train"]
    assert seen and set(seen) == {cap}


# --- cached years carry no inf ---------------------------------------------------------
# Real failure 2026-09-21: laf_spread_ratio divided by an ensemble spread of exactly zero
# (members agreeing - dry-day rainfall, saturated humidity), leaving 1.2-1.4 M inf values
# per cached year, and XGBoost refused humidity_pct outright. The feature code now yields
# NaN there; the caches built before the fix are repaired in place to exactly that output
# (x/0 with x > 0 was the only way to get inf, and it is now NaN), rather than rebuilt.

def _cache_with_inf(path, bad_col="laf_spread_ratio"):
    """Shapes-only synthetic frame, labelled as such."""
    df = _thin_synthetic_year(n=300, variables=("rainfall_mm", "humidity_pct"))
    df["laf_spread_ratio"] = np.linspace(0.5, 1.5, len(df)).astype("float32")
    df["jump_rel_climatology"] = np.float32(1.0)
    df.loc[df.index[::7], bad_col] = np.inf
    df.loc[df.index[::11], bad_col] = -np.inf
    df.to_parquet(path, index=False, row_group_size=64)
    return df


def test_ensure_finite_cache_turns_ratio_inf_into_missing_and_keeps_everything_else(tmp_path):
    p = tmp_path / "paired_2000.parquet"
    before = _cache_with_inf(p)
    n_fixed = pt._ensure_finite_cache(p)
    after = pd.read_parquet(p)
    assert n_fixed == int(np.isinf(before["laf_spread_ratio"]).sum())
    assert not np.isinf(after["laf_spread_ratio"]).any()
    was_inf = np.isinf(before["laf_spread_ratio"]).to_numpy()
    assert after["laf_spread_ratio"][was_inf].isna().all()
    np.testing.assert_array_equal(after["laf_spread_ratio"][~was_inf].to_numpy(),
                                  before["laf_spread_ratio"][~was_inf].to_numpy())
    pd.testing.assert_frame_equal(after.drop(columns="laf_spread_ratio"),
                                  before.drop(columns="laf_spread_ratio"))


def test_ensure_finite_cache_is_a_footer_read_once_a_file_is_checked(tmp_path):
    p = tmp_path / "paired_2000.parquet"
    _cache_with_inf(p)
    pt._ensure_finite_cache(p)
    mtime = p.stat().st_mtime_ns
    assert pt._ensure_finite_cache(p) == 0
    assert p.stat().st_mtime_ns == mtime


def test_ensure_finite_cache_refuses_inf_in_a_column_it_has_no_rule_for(tmp_path):
    p = tmp_path / "paired_2000.parquet"
    _cache_with_inf(p, bad_col="jump_rel_climatology")
    before = p.read_bytes()
    with pytest.raises(ValueError, match="jump_rel_climatology"):
        pt._ensure_finite_cache(p)
    assert p.read_bytes() == before


def test_cache_year_repairs_inf_in_a_cache_it_reuses(tmp_path):
    p = tmp_path / "paired_2000.parquet"
    _cache_with_inf(p)
    assert pt.cache_year(2000, tmp_path) == p
    assert not np.isinf(pd.read_parquet(p)["laf_spread_ratio"]).any()
# --- staged fit: every training day, in chunks that each fit in memory --------------
# MAX_FIT_CYCLES bounds what one QuantileDMatrix holds. The staged mode covers the whole
# pool anyway: disjoint chunks of at most the cap, boosted one after another into the
# same booster (xgb.train's xgb_model=), n_estimators split across them. One model per
# variable, so the registry, inference and SHAP need nothing new.

def test_fit_chunks_partition_the_pool_into_capped_disjoint_chunks():
    cycles = _daily_cycles(range(2000, 2017))
    chunks = pt.fit_chunks(cycles, cap=2000)
    assert len(chunks) == 4  # 6,210 days / 2,000, rounded up
    assert all(len(c) <= 2000 for c in chunks)
    assert set().union(*chunks) == cycles
    assert sum(len(c) for c in chunks) == len(cycles)
    assert chunks == pt.fit_chunks(cycles, cap=2000)
    fold_of = pt.assign_folds(cycles)
    for c in chunks:
        assert {d.year for d in c} == set(range(2000, 2017))
        assert {fold_of[d] for d in c} == {0, 1, 2}


def test_fit_chunks_keeps_a_small_pool_as_one_chunk():
    cycles = _daily_cycles([2000])
    assert pt.fit_chunks(cycles, cap=2000) == [cycles]


def test_split_rounds_sums_to_the_total():
    assert pt._split_rounds(300, 4) == [75, 75, 75, 75]
    assert pt._split_rounds(300, 7) == [43, 43, 43, 43, 43, 43, 42]
    assert sum(pt._split_rounds(300, 7)) == 300


def test_staged_fit_boosts_every_chunk_into_one_booster(tmp_path):
    """Shapes-only synthetic frames: two chunks must yield one booster carrying the full
    n_estimators, trained on every row of both chunks."""
    df = _thin_synthetic_year(n=600, variables=("rainfall_mm",))
    df["abs_error"] = np.abs(np.random.default_rng(3).normal(size=len(df)))
    df["lead_day"] = np.random.default_rng(4).integers(1, 11, size=len(df))
    p = tmp_path / "paired_2017.parquet"
    df.to_parquet(p, index=False)
    hbf = {(r, s): 0.1 for r in df["region_id"].unique() for s in df["season"].unique()}
    cycles = sorted(set(df["init_date"]))
    chunks = [set(cycles[: len(cycles) // 2]), set(cycles[len(cycles) // 2:])]
    booster, n_rows = pt._fit_booster({2017: p}, [2017], "rainfall_mm", chunks,
                                      ["lead_day"], hbf, tmp_path, "cpu", n_estimators=10)
    assert n_rows == len(df)
    assert booster.num_boosted_rounds() == 10


def test_full_retrain_pooled_staged_fits_every_training_cycle_in_chunks(
        tmp_path, _ingested_slice, monkeypatch):
    """Real paired slice, relabelled years. With the cap forced below the pool, the staged
    mode must hand the worker chunks covering every training cycle, and still succeed."""
    from app.ml.train_pipeline import _build_paired_in_chunks

    base, _ = _build_paired_in_chunks()
    cache_dir = tmp_path / "pooled_cache"
    cache_dir.mkdir()
    for year, offset in {2000: 0, 2001: 1, 2002: 2}.items():
        shifted = base.copy()
        for col in ("init_date", "valid_date"):
            shifted[col] = pd.to_datetime(shifted[col]) + pd.DateOffset(years=offset)
        shifted.to_parquet(cache_dir / f"paired_{year}.parquet", index=False)

    n_train = len(pt.pooled_split({y: cache_dir / f"paired_{y}.parquet" for y in (2000, 2001, 2002)},
                                  2002)[0])
    monkeypatch.setattr(pt, "MAX_FIT_CYCLES", max(1, n_train // 2))
    monkeypatch.setattr(pt, "_cuda_available", lambda: False)
    seen = []
    real = pt._run_variable_subprocess
    monkeypatch.setattr(pt, "_run_variable_subprocess",
                        lambda *a, **k: seen.append(a[3]) or real(*a, **k))
    report = pt.full_retrain_pooled(train_years=[2000, 2001], test_year=2002,
                                    cache_dir=cache_dir, fit_mode="staged")
    assert report.status == "success", report.error
    assert report.split_cycles["fit"] == report.split_cycles["train"] == n_train
    assert report.split_cycles["fit_chunks"] >= 2
    assert seen and all(isinstance(c, list) and len(set().union(*c)) == n_train for c in seen)


def test_staged_and_sample_checkpoints_never_collide(tmp_path):
    cycles = _daily_cycles([2000])
    val = {pd.Timestamp("2001-01-01")}
    one = pt._variable_checkpoint_path(tmp_path, "rainfall_mm", [2000], cycles, val)
    staged = pt._variable_checkpoint_path(tmp_path, "rainfall_mm", [2000], [cycles], val)
    assert one != staged


def test_full_retrain_pooled_rejects_an_unknown_fit_mode(tmp_path):
    with pytest.raises(ValueError):
        pt.full_retrain_pooled([2000], 2001, tmp_path, fit_mode="everything")


def test_staged_fit_skips_a_chunk_with_no_rows_and_keeps_every_round(tmp_path):
    """Shapes-only synthetic frame. A chunk of cycles that holds no rows for the variable
    (XGBoost raises on an iterator that never yields a batch) is skipped, and the
    booster still carries the full n_estimators."""
    df = _thin_synthetic_year(n=600, variables=("rainfall_mm",))
    df["abs_error"] = np.abs(np.random.default_rng(3).normal(size=len(df)))
    df["lead_day"] = np.random.default_rng(4).integers(1, 11, size=len(df))
    p = tmp_path / "paired_2017.parquet"
    df.to_parquet(p, index=False)
    hbf = {(r, s): 0.1 for r in df["region_id"].unique() for s in df["season"].unique()}
    empty = {pd.Timestamp("1990-01-01")}
    booster, n_rows = pt._fit_booster({2017: p}, [2017], "rainfall_mm",
                                      [empty, set(df["init_date"])], ["lead_day"], hbf,
                                      tmp_path, "cpu", n_estimators=10)
    assert n_rows == len(df)
    assert booster.num_boosted_rounds() == 10


# --- the per-year event frame is built one batch of forecast dates at a time --------
# Real crash 2026-09-22, 17-year pool: the year-events worker read a whole 76.5M-row
# year (41.6 GB peak commit measured on real 2015) and died on a 1.14 GiB allocation on
# both attempts. EVENT_KEYS include init_date and build_event_frame only groups within
# an event, so building it batch by batch and concatenating is exactly the whole-year
# frame - the same argument the per-year split already rests on.

def test_year_event_frame_is_identical_whether_built_whole_or_one_cycle_at_a_time(
        tmp_path, _ingested_slice):
    """Real paired slice (GEFS + ERA5 samples), real fold models trained on it."""
    from app.ml.train_pipeline import _build_paired_in_chunks

    base, _ = _build_paired_in_chunks()
    p = tmp_path / "paired_2019.parquet"
    base.to_parquet(p, index=False)
    cached = {2019: p}
    cycles = set(pd.to_datetime(base["init_date"]).dt.normalize().unique())
    assert len(cycles) >= 3
    hbf, p90_error, bust_threshold = pt.pooled_stats(cached, [2019], cycles)
    fold_of = pt.assign_folds(cycles)
    fold_models = {v: pt.oof_fold_models(cached, [2019], v, cycles, hbf, fold_of, tmp_path)
                   for v in sorted(base["variable"].unique())}
    columns = set(base.columns) - {"historical_bust_frequency_region_season"}

    whole = pt.year_event_frame(p, cycles, hbf, p90_error, bust_threshold, fold_models,
                                fold_of, columns, max_cycles_per_batch=10**6)
    batched = pt.year_event_frame(p, cycles, hbf, p90_error, bust_threshold, fold_models,
                                  fold_of, columns, max_cycles_per_batch=1)

    def norm(df):
        df = df.copy()
        df["region_id"] = df["region_id"].astype(str)
        df["season"] = df["season"].astype(str)
        return df.sort_values(list(fe.EVENT_KEYS)).reset_index(drop=True)[sorted(df.columns)]

    assert len(whole) > 0
    pd.testing.assert_frame_equal(norm(whole), norm(batched), check_dtype=False)


def test_cycle_batches_cover_every_cycle_once_in_order():
    cycles = [pd.Timestamp("2015-01-01") + pd.Timedelta(days=i) for i in range(365)]
    batches = pt._cycle_batches(set(cycles), 100)
    assert [len(b) for b in batches] == [100, 100, 100, 65]
    assert [c for b in batches for c in b] == cycles


def test_test_event_frame_and_metrics_are_identical_whole_or_batched(tmp_path, _ingested_slice):
    """Real paired slice, real regressors trained on it. The held-out year is scored the
    same way: batched by forecast date, with each variable's metrics computed once over
    every row, so they equal the whole-year computation exactly."""
    from app.ml.train_pipeline import _build_paired_in_chunks

    base, _ = _build_paired_in_chunks()
    p = tmp_path / "paired_2019.parquet"
    base.to_parquet(p, index=False)
    cached = {2019: p}
    cycles = set(pd.to_datetime(base["init_date"]).dt.normalize().unique())
    hbf, p90_error, bust_threshold = pt.pooled_stats(cached, [2019], cycles)
    artifacts = {}
    for v in sorted(base["variable"].unique()):
        art = pt.train_variable_regressor_pooled(cached, [2019], v, cycles, pd.DataFrame(
            columns=base.columns), hbf, tmp_path)
        if art is not None:
            artifacts[v] = art
    assert artifacts
    columns = set(base.columns) - {"historical_bust_frequency_region_season"}

    whole, m_whole = pt.test_event_frame(p, cycles, hbf, p90_error, bust_threshold,
                                         artifacts, columns, max_cycles_per_batch=10**6)
    batched, m_batched = pt.test_event_frame(p, cycles, hbf, p90_error, bust_threshold,
                                             artifacts, columns, max_cycles_per_batch=1)

    def norm(df):
        df = df.copy()
        df["region_id"] = df["region_id"].astype(str)
        df["season"] = df["season"].astype(str)
        return df.sort_values(list(fe.EVENT_KEYS)).reset_index(drop=True)[sorted(df.columns)]

    pd.testing.assert_frame_equal(norm(whole), norm(batched), check_dtype=False)
    assert m_whole.keys() == m_batched.keys() and m_whole
    for v in m_whole:
        assert m_batched[v] == pytest.approx(m_whole[v], rel=1e-9, nan_ok=True)


# --- validation events off the parent, and bounded classifier training events --------
# Real crash 2026-09-22 (04:26 UTC): with every training cycle's events resident (float64,
# ~39 M rows at seventeen years), the parent read the whole validation spill back to build
# event_va and failed a 306 MB malloc. Validation events are now built in a worker, one
# batch of forecast dates at a time; the classifier's training events come from the same
# bounded cycle sample the regressors fit on; event frames are float32 - the precision
# XGBoost trains in anyway.

def _spill_like_the_worker(base, spill_dir):
    """Real slice, laid out exactly as _build_val_frame_worker writes it."""
    df = base.reset_index(drop=True).copy()
    df["_va_row"] = range(len(df))
    df.to_parquet(spill_dir, partition_cols=["variable"], index=False)
    return len(df)


def test_val_event_frame_matches_the_old_whole_frame_path(tmp_path, _ingested_slice):
    from app.features import pivot as pv
    from app.ml.train_pipeline import _build_paired_in_chunks

    base, _ = _build_paired_in_chunks()
    p = tmp_path / "paired_2019.parquet"
    base.to_parquet(p, index=False)
    cycles = set(pd.to_datetime(base["init_date"]).dt.normalize().unique())
    hbf, p90_error, bust_threshold = pt.pooled_stats({2019: p}, [2019], cycles)
    spill = tmp_path / "_va"
    n = _spill_like_the_worker(pt.attach_hbf_column(base.copy(), hbf), spill)
    val_pred = np.random.default_rng(0).random(n)  # stand-in predictions, shapes only

    old = pv.build_event_frame(pd.read_parquet(spill).set_index("_va_row").sort_index(),
                               pd.Series(val_pred), p90_error, bust_threshold, hbf,
                               copy_input=False)
    new = pt.val_event_frame(spill, val_pred, hbf, p90_error, bust_threshold,
                             max_cycles_per_batch=1)

    def norm(df):
        df = df.copy()
        for c in ("region_id", "season"):
            df[c] = df[c].astype(str)
        return df.sort_values(list(fe.EVENT_KEYS)).reset_index(drop=True)[sorted(df.columns)]

    assert len(old) > 0
    pd.testing.assert_frame_equal(norm(old), norm(new), check_dtype=False, rtol=1e-6)


def test_event_frames_come_back_as_float32():
    df = pd.DataFrame({"a": np.array([1.5, 2.5]), "y_bust": [0, 1],
                       "region_id": pd.Categorical(["r1", "r2"])})
    out = pt._events_float32(df)
    assert out["a"].dtype == np.float32
    assert out["y_bust"].dtype == df["y_bust"].dtype
    assert isinstance(out["region_id"].dtype, pd.CategoricalDtype)


def test_classifier_training_events_come_from_the_fit_sample(tmp_path, _ingested_slice,
                                                             monkeypatch):
    """Real paired slice, relabelled years. With the cap below the pool, the events the
    classifier trains on must cover only the sampled cycles."""
    from app.ml.train_pipeline import _build_paired_in_chunks

    base, _ = _build_paired_in_chunks()
    cache_dir = tmp_path / "pooled_cache"
    cache_dir.mkdir()
    for year, offset in {2000: 0, 2001: 1, 2002: 2}.items():
        shifted = base.copy()
        for col in ("init_date", "valid_date"):
            shifted[col] = pd.to_datetime(shifted[col]) + pd.DateOffset(years=offset)
        shifted.to_parquet(cache_dir / f"paired_{year}.parquet", index=False)
    n_train = len(pt.pooled_split({y: cache_dir / f"paired_{y}.parquet" for y in (2000, 2001, 2002)},
                                  2002)[0])
    cap = max(2, n_train - 2)
    monkeypatch.setattr(pt, "MAX_FIT_CYCLES", cap)
    monkeypatch.setattr(pt, "_cuda_available", lambda: False)
    seen = []
    real = pt.build_pooled_train_events
    monkeypatch.setattr(pt, "build_pooled_train_events",
                        lambda *a, **k: seen.append(len(a[2])) or real(*a, **k))
    report = pt.full_retrain_pooled(train_years=[2000, 2001], test_year=2002, cache_dir=cache_dir)
    assert report.status == "success", report.error
    assert seen == [cap]
    assert report.split_cycles["classifier"] == cap


# --- a pooled run is made servable: SHAP summary and manifest, from the same code -----
# Real gap 2026-09-22: full_retrain_pooled saved models, thresholds and metrics but not
# shap_summary.parquet or the manifest's shap_method, both of which full_retrain writes.
# The region panel's "what drove this prediction" reads that summary, so serving a pooled
# run would have silently served no explanation. finalize_for_serving rebuilds the
# validation events from the caches in batches, refuses unless they reproduce the saved
# validation ROC-AUC exactly, and explains the models the way full_retrain does.

def _pooled_run_on_slice(tmp_path, monkeypatch):
    from app.ml.train_pipeline import _build_paired_in_chunks

    base, _ = _build_paired_in_chunks()
    cache_dir = tmp_path / "pooled_cache"
    cache_dir.mkdir()
    for year, offset in {2000: 0, 2001: 1, 2002: 2}.items():
        shifted = base.copy()
        for col in ("init_date", "valid_date"):
            shifted[col] = pd.to_datetime(shifted[col]) + pd.DateOffset(years=offset)
        shifted.to_parquet(cache_dir / f"paired_{year}.parquet", index=False)
    monkeypatch.setattr(pt, "_cuda_available", lambda: False)
    report = pt.full_retrain_pooled(train_years=[2000, 2001], test_year=2002, cache_dir=cache_dir)
    assert report.status == "success", report.error
    return report, cache_dir


def test_a_pooled_run_ships_with_a_shap_summary_and_manifest(tmp_path, _ingested_slice,
                                                             monkeypatch):
    from app.ml import registry

    report, _ = _pooled_run_on_slice(tmp_path, monkeypatch)
    rd = registry.run_dir(report.run_id)
    shap = pd.read_parquet(rd / "shap_summary.parquet")
    assert "classifier" in set(shap["model"])
    assert any(m.startswith("regressor::") for m in set(shap["model"]))
    manifest = json.loads((rd / "manifest.json").read_text())
    assert manifest["shap_method"] == "shap"
    assert manifest["paired_rows"] > 0


def test_finalize_refuses_a_run_whose_classifier_does_not_reproduce_its_metrics(
        tmp_path, _ingested_slice, monkeypatch):
    from app.ml import registry

    report, cache_dir = _pooled_run_on_slice(tmp_path, monkeypatch)
    metrics_path = registry.run_dir(report.run_id) / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["classifier"]["val"]["roc_auc"] = 0.123  # no longer what the model scores
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="validation ROC-AUC"):
        pt.finalize_for_serving(report.run_id, cache_dir)


# --- eval events for a pooled run ----------------------------------------------------

def test_a_pooled_run_emits_the_eval_events_the_deck_and_ladder_read(tmp_path, monkeypatch):
    """scripts/ppt_figures.py and scripts/run_baselines both read one file.

    They read data/analysis/eval_events/<run_id>.parquet, filter split == "test", and need
    y_bust, model_proba and lead_time_days. full_retrain_pooled scored exactly those rows
    and then threw them away, so a pooled model could not produce the per-lead-day POD/FAR
    table, the confusion counts or the baseline ladder - which meant the deck had to be
    built from a model that was not the one serving the site.
    """
    import pandas as pd

    from app.ml import pooled_training as pt

    written = {}

    def fake_emit(run_id, clf_art, splits):
        written["run_id"] = run_id
        written["splits"] = {k: len(v) for k, v in splits.items() if v is not None}
        return tmp_path / f"{run_id}.parquet"

    monkeypatch.setattr(pt, "_emit_eval_events_for_pooled", fake_emit)

    ev = pd.DataFrame({c: [0, 1] for c in pt._PPT_EVENT_COLUMNS})
    for prefix in pt._PPT_EVENT_PER_VARIABLE:
        ev[f"{prefix}_rainfall_mm"] = [0.0, 1.0]
    out = pt._publish_eval_events("run_z", object(), ev, ev.iloc[:1])
    assert written["run_id"] == "run_z"
    # Test is what the deck reads; validation is kept because the ladder compares splits.
    assert written["splits"] == {"val": 2, "test": 1}
    assert out is not None


def test_eval_events_are_never_published_from_an_empty_test_split():
    """An empty file is worse than none: ppt_figures would report zeros as measurements."""
    import pandas as pd

    from app.ml import pooled_training as pt

    assert pt._publish_eval_events("run_z", object(), pd.DataFrame(), pd.DataFrame()) is None


def test_the_case_study_columns_are_required_on_write_not_discovered_on_a_blank_slide():
    """ppt_figures reads the per-variable columns with a NaN default, so absence is silent.

    `getattr(row, f"pred_err_{var}", float("nan"))` turns a missing column into NaN, the
    case-study ranking then has nothing to rank on, and the script emits a deck with an
    empty bust case study and no error. Measured on a known-good non-pooled eval-events
    file 2026-09-23: the full frame produced a case study (Khordha, rainfall_mm, actual
    error 97.6 against a 13.58 threshold); the same rows cut to the headline columns
    produced region None, variable None and zero exceedances, silently.
    """
    import pandas as pd

    from app.ml import pooled_training as pt

    full = pd.DataFrame({c: [0] for c in pt._PPT_EVENT_COLUMNS})
    for prefix in pt._PPT_EVENT_PER_VARIABLE:
        full[f"{prefix}_rainfall_mm"] = [0]
    assert pt.eval_event_contract_gaps(full) == []

    # The headline columns alone are the seven-column file that produced a blank case
    # study, so they must come back as a gap - and as the per-variable gap specifically.
    # See test_a_frame_with_no_per_variable_columns_is_a_gap_not_a_pass for why that
    # floor has to be explicit rather than derived.
    gaps = pt.eval_event_contract_gaps(full[["y_bust", "lead_time_days"]])
    assert gaps and any("per-variable" in g for g in gaps)

    dropped = full.drop(columns=["actual_err_rainfall_mm"])
    assert pt.eval_event_contract_gaps(dropped) == ["actual_err_rainfall_mm"]
    with pytest.raises(ValueError, match="case study"):
        pt._publish_eval_events("run_z", object(), None, dropped)


def test_a_frame_with_no_per_variable_columns_is_a_gap_not_a_pass():
    """Deriving the requirement from the data means an absent requirement cannot fail.

    The per-variable requirement is read off the `pred_err_*` columns present, because
    `skipped_variables` legitimately shrinks the set. With no floor, a frame carrying none
    of them requires only the base columns and passes - the same frame shape that empties
    the bust case study. skipped_variables can shrink the set; it cannot empty it.
    """
    import pandas as pd

    from app.ml import pooled_training as pt

    base_only = pd.DataFrame({c: [0] for c in pt._PPT_EVENT_COLUMNS})
    assert pt.eval_event_contract_gaps(base_only) != []

    one_variable = base_only.copy()
    for prefix in pt._PPT_EVENT_PER_VARIABLE:
        one_variable[f"{prefix}_temperature_c"] = [0.0]
    assert pt.eval_event_contract_gaps(one_variable) == []
    # A run that skipped seven of eight variables is still publishable.
    assert pt.eval_event_contract_gaps(
        one_variable.drop(columns=["spread_temperature_c"])) == ["spread_temperature_c"]


# --- a regressor that is worse than useless must not become an artifact -------------

def test_a_regressor_worse_than_predicting_the_mean_is_refused():
    """run_20260922T100055Z saved a temperature regressor with held-out r2 -254107.

    It predicted absolute temperature errors from -80,235 to +270 degrees. The promotion
    gate did not see it, because the gate reads the classifier's ROC-AUC and the
    classifier had been TRAINED on those values, so it had learned to read them - the
    served bust distribution was 0.008 from the good model's median and a degeneracy
    check passed it. Nothing downstream can catch this; it has to be refused where it is
    made. Rule 3: a model that got worse does not ship.

    Both conditions must fail, not either: a genuinely hard variable can have a weak r2
    while still beating the trivial predictor, and wind_direction_deg legitimately sits
    at r2 0.41. Worse on squared error AND worse on absolute error is not ambiguous.
    """
    from app.ml import pooled_training as pt

    # The real numbers from that run's temperature_c, and from the run that is serving.
    assert pt.regressor_is_unusable({"r2": -254107.19, "mae": 5.746,
                                     "baseline_mae_predict_mean": 0.92})
    assert not pt.regressor_is_unusable({"r2": 0.5783, "mae": 0.6497,
                                         "baseline_mae_predict_mean": 0.92})
    # Weak but genuinely useful: beats the mean on both. wind_direction_deg's shape.
    assert not pt.regressor_is_unusable({"r2": 0.41, "mae": 38.0,
                                         "baseline_mae_predict_mean": 55.3})
    # Negative r2 but still beating the mean on absolute error - not refused, because
    # squared error alone is dominated by a handful of outliers.
    assert not pt.regressor_is_unusable({"r2": -0.2, "mae": 0.8,
                                         "baseline_mae_predict_mean": 0.92})
    # No metrics at all cannot be judged, and must not be refused on a guess.
    assert not pt.regressor_is_unusable({})
    assert not pt.regressor_is_unusable({"r2": float("nan"), "mae": 1.0,
                                         "baseline_mae_predict_mean": 0.9})


def test_the_refusal_is_wired_into_the_function_that_makes_the_artifact(tmp_path, monkeypatch):
    """The predicate being right is not the same as it being reached.

    CI installs requirements.txt and requirements-dev.txt only, so torch is absent and
    every end-to-end pooled test skips - which means a green CI run says nothing about
    whether the refusal actually fires. This test calls train_variable_regressor_pooled
    directly. It needs no torch: torch is imported only inside _cuda_available, which
    full_retrain_pooled calls and this function does not. So the wiring is covered
    wherever the suite runs, not only where a GPU stack happens to be installed.

    The fit itself is stubbed - this is a plumbing test and no number in it is a metric.
    """
    import numpy as np
    import pandas as pd

    from app.ml import pooled_training as pt

    class _Stub:
        """Returns a constant far from the target, so it loses to predicting the mean."""
        def __init__(self, value):
            self.value = value

        def predict(self, X):
            return np.full(len(X), self.value, dtype=float)

    cols = ["ensemble_spread"]
    monkeypatch.setattr(pt, "_feature_columns_for", lambda *a, **k: cols)
    monkeypatch.setattr(pt, "_fit_booster", lambda *a, **k: (object(), 1000))
    monkeypatch.setattr(pt, "attach_hbf_column", lambda df, hbf: df)

    rng = np.random.default_rng(0)
    va = pd.DataFrame({"variable": "temperature_c",
                       "ensemble_spread": rng.random(200),
                       "abs_error": rng.random(200)})

    # A model predicting 500 where the target is in [0, 1): worse than the mean on both
    # squared and absolute error, which is the shape run_20260922T100055Z shipped.
    monkeypatch.setattr(pt, "_booster_to_sklearn", lambda *a, **k: _Stub(500.0))
    assert pt.train_variable_regressor_pooled(
        {2000: tmp_path / "x.parquet"}, [2000], "temperature_c",
        {pd.Timestamp("2000-01-01")}, va, {}, tmp_path) is None

    # A model predicting near the mean of the target is weak, not unusable, and must
    # still produce an artifact - the floor exists to catch damage, not mediocrity.
    monkeypatch.setattr(pt, "_booster_to_sklearn",
                        lambda *a, **k: _Stub(float(va["abs_error"].mean())))
    art = pt.train_variable_regressor_pooled(
        {2000: tmp_path / "x.parquet"}, [2000], "temperature_c",
        {pd.Timestamp("2000-01-01")}, va, {}, tmp_path)
    assert art is not None and art.variable == "temperature_c"


# ------------------------------------------------------- the baseline ladder's train rows


def test_the_ladder_fits_from_these_columns_alone(_ingested_slice):
    """`BASELINE_FIT_*` is a contract, and this is what enforces it.

    Fits every baseline twice on identical real rows - once on the whole event frame,
    once on nothing but the declared columns - and requires the predictions to be equal.
    If a new baseline starts reading a column outside the set (a `pred_err_*`, say, which
    would quietly make it a second model rather than a baseline), this fails here instead
    of producing a ladder fitted on a frame missing what it needed.

    Same comparison as the measurement behind the constant, which ran on 1,048,576 real
    held-out rows and found every baseline identical to 0.000e+00.
    """
    import numpy as np
    from app.features import pivot as pv
    from app.ml import baselines as bl
    from app.storage.parquet_store import read_dataset

    paired = fe.build_training_frame(read_dataset())
    cycles = sorted(paired["init_date"].dropna().unique())
    pred = pd.Series(np.random.default_rng(3).random(len(paired)), index=paired.index)
    p90 = {v: 5.0 for v in paired["variable"].unique()}
    thr = {v: 3.0 for v in paired["variable"].unique()}
    ev = pv.build_event_frame(paired, pred, p90, thr)
    assert len(ev) > 4 and len(cycles) >= 2

    keep = pt.baseline_fit_columns(ev)
    assert bl.LABEL in keep and "lead_time_days" in keep
    assert not [c for c in keep if c.startswith("pred_err_") or c.startswith("conf_")]

    half = max(2, len(ev) // 2)
    tr_full, te_full = ev.iloc[:half], ev.iloc[half:]
    full = bl.fit_all(tr_full)
    thin = bl.fit_all(tr_full[keep])
    for name, model in full.items():
        np.testing.assert_array_equal(
            model.predict_proba(te_full), thin[name].predict_proba(te_full[keep]),
            err_msg=f"{name} reads an event column outside BASELINE_FIT_*")


def test_a_nan_prediction_changes_nothing_the_ladder_reads(_ingested_slice):
    """The whole saving depends on this: driving the shipped event builder with an
    all-NaN prediction vector must leave every baseline-fit column untouched.

    If it did not, `baseline_fit_events` would be fitting the ladder on rows that differ
    from the ones the model was scored against - which is the exact failure
    `run_baselines`' docstring warns about, and which nothing downstream could detect.

    Verified here on real-derived rows, and separately against the live 17-year run: all
    twenty baseline-fit columns bit-identical across 79,920 real held-out events built
    the expensive way with all eight regressors run.
    """
    import numpy as np
    from app.features import pivot as pv
    from app.storage.parquet_store import read_dataset

    paired = fe.build_training_frame(read_dataset())
    p90 = {v: 5.0 for v in paired["variable"].unique()}
    thr = {v: 3.0 for v in paired["variable"].unique()}

    real = pv.build_event_frame(paired.copy(),
                                pd.Series(np.random.default_rng(5).random(len(paired)),
                                          index=paired.index), p90, thr)
    blank = pv.build_event_frame(paired.copy(),
                                 pd.Series(np.nan, index=paired.index, dtype=float),
                                 p90, thr)

    keep = pt.baseline_fit_columns(real)
    assert len(keep) > len(fe.EVENT_KEYS), "no baseline-fit columns were built at all"
    assert set(keep) <= set(blank.columns)
    pd.testing.assert_frame_equal(real[keep].reset_index(drop=True),
                                  blank[keep].reset_index(drop=True))

    # And the prediction really was dropped, so this is not passing by accident.
    assert real["pred_err_temperature_c"].notna().any()
    assert ("pred_err_temperature_c" not in blank.columns
            or blank["pred_err_temperature_c"].isna().all())
