"""Pool many years of paired training data without ever holding more than one year's
full-width frame resident at once.

Why this exists
----------------
`_build_paired_in_chunks` builds ONE frame spanning every requested year at once, and
`full_retrain`'s `tr`/`va`/`te` are further `.copy()`'d subsets of it - so training on N
years costs memory proportional to N, and even 3 years does not fit in 23.7 GB (see
docs/known-issues.md, "does more data help" entries). XGBoost's `DataIter` /
`QuantileDMatrix` external-memory interface is built for exactly this: read one batch -
here, one year - at a time, discard it, and never materialise the pooled whole.

The three global statistics every regressor and the classifier need - the member p90, the
event-level bust threshold, and the historical bust frequency by (region, season) - each
reduce to a groupby whose keys never cross a year boundary (EVENT_KEYS includes
`init_date`; `(region_id, season)` is a per-row label). So each can be computed by
streaming every cached year once, keeping only a handful of thin columns or an
already-aggregated result, and combining the per-year partial results afterward. This is
proven equal to computing them on the full concatenation, not merely an approximation -
see test_pooled_matches_full_frame_on_stats.

What still needs one year fully in memory
-------------------------------------------
Building `event_tr` needs out-of-fold regressor predictions attached to that year's own
rows, so `full_retrain_pooled` streams the cached train years a second time, one at a
time, to do that join and immediately reduce to the (much smaller) event grain before
moving to the next year. Validation and test are each a single held-out slice (the pool's
chronological tail, and one whole calendar year respectively) and are held fully in
memory throughout, matching what `full_retrain` already does for them.

Compatibility with the rest of the system
-------------------------------------------
A regressor trained via `xgb.train()` on a `QuantileDMatrix` is a `Booster`, not the
`xgb.XGBRegressor` the registry, `app.ml.inference` and `app.ml.explain` (SHAP) expect.
`_booster_to_sklearn` round-trips it through `save_model`/`load_model` - the same format
`registry.py` already persists - so every downstream consumer sees an ordinary
`XGBRegressor`/`XGBClassifier` and nothing else has to change.
"""
from __future__ import annotations

import gc
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from app.features import engineering as fe
from app.features import pivot as pv
from app.ml import classifier as clf_mod
from app.ml import regressors as reg_mod
from app.ml.thresholds import (
    Thresholds,
    compute_error_thresholds,
    compute_member_p90,
    compute_risk_bands,
)
from app.ml.train_pipeline import (
    TRAIN_FRAC,
    VAL_FRAC,
    TrainReport,
    _build_paired_in_chunks,
    _event_mean_error,
)

_THIN_STATS_COLUMNS = ["region_id", "season", "variable", "abs_error"]


def cache_year(year: int, cache_dir: Path) -> Path:
    """One year's downcast paired frame, written to disk and never held alongside another
    year's. Idempotent: a year already cached is reused, matching the fetch scripts'
    own resumability convention."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"paired_{year}.parquet"
    if path.exists():
        return path
    lo, hi = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
    paired, _ = _build_paired_in_chunks(init_date_min=lo, init_date_max=hi)
    if paired.empty:
        raise ValueError(f"no paired forecast+observation data for {year} in the store")
    paired.to_parquet(path, index=False)
    del paired
    return path


def pooled_split(cached_paths: dict, test_year: int):
    """Train/val/test cycles across a pool of cached years, holding `test_year` out
    entirely - the multi-year generalisation of `_split_by_year`. Reads only the
    `init_date` column of each cached file, never a full year's frame."""
    pre_cycles: list = []
    test_cycles: set = set()
    for year, path in sorted(cached_paths.items()):
        inits = pd.read_parquet(path, columns=["init_date"])["init_date"].dropna().unique()
        inits = sorted(pd.to_datetime(pd.Series(inits)).dt.normalize().unique())
        if year == test_year:
            test_cycles.update(inits)
        else:
            pre_cycles.extend(inits)
    pre_cycles = sorted(set(pre_cycles))
    n = len(pre_cycles)
    if n < 2:
        return set(pre_cycles), set(), test_cycles
    a = max(1, int(round(n * TRAIN_FRAC / (TRAIN_FRAC + VAL_FRAC))))
    a = min(a, n - 1)
    return set(pre_cycles[:a]), set(pre_cycles[a:]), test_cycles


def pooled_stats(cached_paths: dict, train_years: list, train_cycles: set):
    """`(hbf, p90_error, bust_threshold)`, streamed one cached year at a time.

    Reads only the columns each statistic needs - never the full feature-engineered
    frame - and concatenates just those thin, already-reduced results across years before
    the real computation, which is otherwise identical to the single-frame functions
    (`fe.compute_historical_bust_frequency`, `compute_member_p90`,
    `compute_error_thresholds`) and produces identical output for identical input rows.
    """
    event_cols = list(dict.fromkeys(
        fe.EVENT_KEYS + ["variable", "forecast_value", "observed_value"]))
    thin_frames, event_frames = [], []
    for year in train_years:
        df = pd.read_parquet(cached_paths[year],
                             columns=list(dict.fromkeys(_THIN_STATS_COLUMNS + event_cols)))
        df = df[df["init_date"].isin(train_cycles)]
        if df.empty:
            continue
        thin = df[_THIN_STATS_COLUMNS].copy()
        # Cast per year, on the smaller frame, not after concatenation. Real crash
        # 2026-09-14: two years' region_id/season categoricals whose category sets or
        # order did not align exactly (e.g. 2019's, cached separately from 2016/2017's)
        # made pd.concat silently fall back to plain object dtype for the combined
        # column, and compute_historical_bust_frequency's own `.astype(str)` then had to
        # materialise a fresh ~190M-row string array in one shot - 20.5 GiB, over the
        # ceiling. Every year's own category set converts to str safely on its own; the
        # concat below then only ever joins already-plain string columns.
        thin["region_id"] = thin["region_id"].astype(str)
        thin["season"] = thin["season"].astype(str)
        thin_frames.append(thin)
        event_frames.append(_event_mean_error(df))
    if not thin_frames:
        return {}, {}, {}
    thin = pd.concat(thin_frames, ignore_index=True)
    hbf = fe.compute_historical_bust_frequency(thin)
    p90_error = compute_member_p90(thin[["variable", "abs_error"]])
    event_err = pd.concat(event_frames, ignore_index=True)
    bust_threshold = compute_error_thresholds(event_err, percentile=90.0)
    return hbf, p90_error, bust_threshold


def _feature_columns_for(cached_paths: dict, years: list) -> list:
    """Which columns `reg_mod.feature_columns` would pick for this variable's frame,
    without reading any row data - schema only, plus the one feature attached after
    caching (`historical_bust_frequency_region_season`, computed globally in
    `pooled_stats` rather than known at cache time)."""
    import pyarrow.parquet as pq
    names = set(pq.ParquetFile(cached_paths[years[0]]).schema_arrow.names)
    names.add("historical_bust_frequency_region_season")
    return reg_mod.feature_columns(pd.DataFrame(columns=sorted(names)))


class _YearDataIter(xgb.DataIter):
    """Feeds one variable's training rows to XGBoost one cached year at a time.

    `hbf` is looked up per row here, not baked into the cache: it is only known once
    every training year has been scanned once for `pooled_stats`, after every year is
    already cached.
    """

    def __init__(self, cached_paths: dict, years: list, variable: str, cycles: set,
                 feature_cols: list, hbf: dict, cache_dir: Path):
        self._paths = [cached_paths[y] for y in years]
        self._variable = variable
        self._cycles = cycles
        self._feature_cols = feature_cols
        self._hbf = hbf
        self._read_cols = list(dict.fromkeys(
            [c for c in feature_cols if c != "historical_bust_frequency_region_season"]
            + ["region_id", "season", "variable", "init_date", "abs_error"]))
        self._i = 0
        # No cache_prefix: QuantileDMatrix builds its quantile sketch batch by batch and
        # never writes batches to disk itself - passing one is a QuantileDMatrix-specific
        # ValueError, unlike the plain external-memory DMatrix this class is modelled on.
        super().__init__()

    def next(self, input_data) -> int:
        if self._i == len(self._paths):
            return 0
        df = pd.read_parquet(self._paths[self._i], columns=self._read_cols)
        df = df[(df["variable"] == self._variable) & (df["init_date"].isin(self._cycles))]
        self._i += 1
        if df.empty:
            return 1
        key = list(zip(df["region_id"].astype(str), df["season"].astype(str)))
        # No .copy() before this assignment: boolean-mask filtering above already
        # produced a new frame, and an extra .copy() here forces pandas to consolidate
        # its blocks into one contiguous array per dtype - a second full-sized allocation
        # on top of the filtered frame already in memory. Measured real crash 2026-09-13,
        # ArrayMemoryError on a single cached year's own frame.
        df["historical_bust_frequency_region_season"] = [self._hbf.get(k, np.nan) for k in key]
        X = reg_mod._prep_X(df, self._feature_cols)
        input_data(data=X, label=df["abs_error"].to_numpy())
        return 1

    def reset(self) -> None:
        self._i = 0


def _booster_to_sklearn(booster: "xgb.Booster", cls) -> "xgb.XGBModel":
    """Round-trip a Learning-API Booster into the Scikit-Learn wrapper `registry.py`,
    `app.ml.inference` and `app.ml.explain` all expect - same JSON format
    `registry.save_regressor`/`save_classifier` already write, so nothing downstream
    needs to know this model was trained differently."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.json"
        booster.save_model(p)
        model = cls()
        model.load_model(p)
    return model


def _xgb_train_params(device: str) -> dict:
    """`reg_mod.XGB_PARAMS`, minus the sklearn-only keys `xgb.train`'s Learning API does
    not take, plus the device to train on. `device="cuda"` needs no other change: a
    `QuantileDMatrix` built from a `DataIter` supports GPU training directly, verified
    against a real fit before this was wired in here - the batches XGBoost pulls off the
    iterator stay ordinary pandas/numpy on the host side either way."""
    params = {k: v for k, v in reg_mod.XGB_PARAMS.items()
             if k not in ("n_estimators", "enable_categorical", "n_jobs", "random_state")}
    params["nthread"] = 0
    params["seed"] = reg_mod.XGB_PARAMS["random_state"]
    params["device"] = device
    return params


def train_variable_regressor_pooled(cached_paths: dict, train_years: list, variable: str,
                                    train_cycles: set, val_df: "pd.DataFrame",
                                    hbf: dict, cache_dir: Path,
                                    device: str = "cpu") -> "reg_mod.RegressorArtifact | None":
    """The pooled-training equivalent of `regressors.train_variable_regressor`: same
    params, same features, fit via an external-memory `QuantileDMatrix` instead of a
    single in-memory `.fit()` so `train_years` is never all resident at once."""
    cols = _feature_columns_for(cached_paths, train_years)
    it = _YearDataIter(cached_paths, train_years, variable, train_cycles, cols, hbf, cache_dir)
    dtrain = xgb.QuantileDMatrix(it, enable_categorical=True)
    if dtrain.num_row() < reg_mod.MIN_ROWS:
        return None

    params = _xgb_train_params(device)
    booster = xgb.train(params, dtrain, num_boost_round=reg_mod.XGB_PARAMS["n_estimators"])
    model = _booster_to_sklearn(booster, xgb.XGBRegressor)

    va = val_df[val_df["variable"] == variable]
    n_train = int(dtrain.num_row())
    del it, dtrain, booster  # see oof_fold_models for why: real fragmentation crashes
    gc.collect()
    # No train-split metric here, unlike the single-frame path: recomputing it would mean
    # a second full external-memory pass over every pooled year just to score rows the
    # model already fit on. Held-out val/test - what the promotion gate actually reads -
    # are unaffected.
    metrics = {}
    if len(va):
        key = list(zip(va["region_id"].astype(str), va["season"].astype(str)))
        va = va.assign(historical_bust_frequency_region_season=[hbf.get(k, np.nan) for k in key])
        Xva = reg_mod._prep_X(va, cols)
        if len(va) >= 5:
            metrics["val"] = reg_mod._evaluate(va["abs_error"], model.predict(Xva))
    return reg_mod.RegressorArtifact(variable, model, cols, metrics, n_train, len(va))


def assign_folds(train_cycles: set, n_splits: int = 3) -> dict:
    """cycle -> fold id, computed once and shared across every variable - the pooled
    equivalent of `regressors.oof_predict`'s `GroupKFold` on `init_date`. Folds are the
    same regardless of which variable trains on them, so this only needs computing once,
    not once per variable."""
    cycles = sorted(train_cycles)
    n_splits = min(n_splits, max(len(cycles), 1))
    return {c: i % n_splits for i, c in enumerate(cycles)}


def oof_fold_models(cached_paths: dict, train_years: list, variable: str,
                    train_cycles: set, hbf: dict, fold_of: dict, cache_dir: Path,
                    device: str = "cpu"):
    """fold id -> (fitted, sklearn-wrapped booster, feature columns), each excluding its
    own fold's cycles. Used only to compute out-of-fold predictions for the training
    events the classifier trains on - never saved as an artifact."""
    if len(train_cycles) < 2:
        return {}
    n_splits = len(set(fold_of.values()))
    cols = _feature_columns_for(cached_paths, train_years)
    models = {}
    for fold in range(n_splits):
        fold_cycles = {c for c in train_cycles if fold_of[c] != fold}
        it = _YearDataIter(cached_paths, train_years, variable, fold_cycles, cols, hbf, cache_dir)
        dtrain = xgb.QuantileDMatrix(it, enable_categorical=True)
        if dtrain.num_row() < reg_mod.MIN_ROWS:
            del it, dtrain
            gc.collect()
            continue
        params = _xgb_train_params(device)
        booster = xgb.train(params, dtrain, num_boost_round=reg_mod.XGB_PARAMS["n_estimators"])
        models[fold] = (_booster_to_sklearn(booster, xgb.XGBRegressor), cols)
        # Explicit cleanup, not left to Python's own GC timing: real repeated crashes
        # 2026-09-14, a small (~900 MB) pyarrow malloc failing after ~1.5-2 hours of a
        # process that had been running fine - the signature of fragmentation, not a
        # leak, from many short-lived DataIter/QuantileDMatrix/Booster objects (each
        # wrapping native pyarrow/XGBoost C++ allocations Python's cyclic GC does not
        # prioritise) accumulating across dozens of fits without being freed promptly.
        del it, dtrain, booster
        gc.collect()
    return models


def build_pooled_train_events(cached_paths: dict, train_years: list, train_cycles: set,
                              hbf: dict, p90_error: dict, bust_threshold: dict,
                              fold_models: dict, fold_of: dict,
                              columns: "set | None" = None) -> "pd.DataFrame":
    """`event_tr`, assembled one cached year at a time: load that year's train rows,
    attach out-of-fold predictions via the fold each row's cycle belongs to, reduce to
    event grain, discard the year, move on. Concatenating the (small) per-year event
    frames afterward is exactly `pv.build_event_frame` on the full multi-year `tr` would
    produce, since EVENT_KEYS never crosses a year boundary.

    `columns` should be the same narrowed set `full_retrain_pooled` reads `va`/`te` with:
    `build_event_frame` (unchanged, shared with the single-frame path) does its own
    `paired.copy()` internally, and that is only affordable per year, not per pooled
    train set, if `df` going in is narrow - the full ~90-column frame made this crash for
    real on 2026-09-13, one call after the `.copy()`/`.drop()` calls in this file were
    already fixed."""
    frames = []
    gc.collect()  # reclaim whatever the training loop above left fragmented, before
                  # this loop's own first big read competes with it for free space
    for year in train_years:
        df = pd.read_parquet(cached_paths[year], columns=sorted(columns) if columns else None)
        df = df[df["init_date"].isin(train_cycles)]
        if df.empty:
            continue
        key = list(zip(df["region_id"].astype(str), df["season"].astype(str)))
        # No .copy() before these assignments - see _YearDataIter.next() for why this
        # exact line is where the real 2026-09-13 crash happened: an extra .copy() here
        # forces pandas to consolidate a freshly-filtered, still nearly-year-sized frame's
        # blocks into one contiguous array per dtype, a second full allocation on top of
        # the one already resident.
        df["historical_bust_frequency_region_season"] = [hbf.get(k, np.nan) for k in key]
        df["_fold"] = df["init_date"].map(fold_of)
        oof = pd.Series(np.nan, index=df.index, dtype=float)
        for variable in sorted(df["variable"].unique()):
            models_for_var = fold_models.get(variable, {})
            if not models_for_var:
                continue
            vmask = df["variable"] == variable
            for fold, (model, cols) in models_for_var.items():
                fmask = vmask & (df["_fold"] == fold)
                if not fmask.any():
                    continue
                oof.loc[fmask] = model.predict(reg_mod._prep_X(df.loc[fmask], cols))
        # No .drop(columns=["_fold"]) here: real crash 2026-09-13, ArrayMemoryError on a
        # full year's frame. .drop() reindexes every remaining column's block through the
        # same take_nd path .copy() does - just as expensive on a frame this size, for a
        # column build_event_frame below never looks at (it names its own group keys and
        # aggregation columns explicitly, so an extra unused column costs one int64
        # column's worth of memory, not a second full-frame allocation).
        frames.append(pv.build_event_frame(df, oof, p90_error, bust_threshold, hbf))
        del df, oof
        gc.collect()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def full_retrain_pooled(train_years: list, test_year: int, cache_dir: Path,
                        run_id: str | None = None) -> "TrainReport":
    """The pooled-training equivalent of `train_pipeline.full_retrain`: any number of
    years, never more than one held fully in memory at once. Always `make_current=False`
    - this is a comparison tool, not a promotion path, and this code has not earned the
    same trust yet.

    `train_years` should not include `test_year`; if it does, `test_year` is excluded
    from the training pool automatically (holding it out is the entire point).
    """
    from app.ml import registry

    t0 = pd.Timestamp.now()
    rid = run_id or registry.new_run_id()
    report = TrainReport(run_id=rid, status="failed")

    train_years = [y for y in train_years if y != test_year]
    all_years = sorted(set(train_years) | {test_year})
    cached = {y: cache_year(y, cache_dir) for y in all_years}

    train_c, val_c, test_c = pooled_split(
        {y: cached[y] for y in train_years + [test_year]}, test_year)
    report.split_cycles = {
        "train": len(train_c), "val": len(val_c), "test": len(test_c),
        "train_years": train_years, "test_year": test_year,
    }
    if not train_c:
        report.status = "no_data"
        return report

    hbf, p90_error, bust_threshold = pooled_stats(cached, train_years, train_c)
    if not hbf and not p90_error:
        report.status = "no_data"
        report.error = "no training rows survived the pooled stats pass"
        return report

    # Validation and test are each a single bounded slice - the pool's own chronological
    # tail, and one whole held-out year - so both are read into memory once, exactly as
    # full_retrain already does for them. `test_c` is the WHOLE test year (that is the
    # point of holding a year out entirely), so filtering rows does not shrink `te` the
    # way it shrinks `va` - reading only the columns anything downstream actually touches
    # is what keeps it from being a second full-year frame resident for the rest of this
    # function, on top of whichever training year build_pooled_train_events is streaming
    # at the same time. Real crash 2026-09-13: te (2019, full year) plus one training
    # year plus one redundant .copy() exceeded 23.7 GB; the .copy() is fixed above, this
    # narrows the other big contributor.
    needed = set(_feature_columns_for(cached, train_years)) | set(fe.EVENT_KEYS) | {
        "variable", "forecast_value", "observed_value", "ensemble_spread",
        "abs_error", "region_id", "season"}
    needed.discard("historical_bust_frequency_region_season")  # attached below, not cached

    # `te` (the whole held-out year) is deliberately NOT loaded yet. Real crash
    # 2026-09-13: with te resident for the rest of this function, build_event_frame's own
    # internal `paired.copy()` on the training year build_pooled_train_events is
    # streaming had nowhere left to allocate, even after narrowing columns and removing
    # every redundant .copy()/.drop() this file added. Loading te only after event_tr is
    # built frees exactly that headroom for the step that needs it; te's own
    # predict-and-build-event work below is no more expensive alone than it was
    # alongside everything else.
    val_years = sorted({y for y in train_years
                        for c in val_c if pd.Timestamp(c).year == y}) or train_years
    _va_parts = []
    for y in val_years:
        part = pd.read_parquet(cached[y], columns=sorted(needed))
        # Cast per year before concatenating - see pooled_stats for why: categoricals
        # from different cached years are not guaranteed to share category sets/order,
        # and pd.concat silently degrades a mismatch to plain object dtype.
        part["region_id"] = part["region_id"].astype(str)
        part["season"] = part["season"].astype(str)
        _va_parts.append(part)
    va = pd.concat(_va_parts, ignore_index=True)
    va = va[va["init_date"].isin(val_c)]
    if not va.empty:
        key = list(zip(va["region_id"].astype(str), va["season"].astype(str)))
        va["historical_bust_frequency_region_season"] = [hbf.get(k, np.nan) for k in key]

    variables = sorted(pd.read_parquet(cached[train_years[0]], columns=["variable"])
                       ["variable"].unique())
    fold_of = assign_folds(train_c)

    # Split the variables across a GPU thread and a CPU thread, both running against the
    # same `va` and `hbf`. XGBoost's training call is a C++ extension that releases the
    # GIL while it runs, so this is genuine concurrent use of both the GPU and the CPU
    # cores, not two halves of one resource taking turns - verified with a real GPU fit
    # via QuantileDMatrix+DataIter before this was wired in. Each half returns its own
    # partial results; the halves touch disjoint variables and disjoint rows of `va` (via
    # vmask), so merging afterward rather than writing into shared dicts/Series from both
    # threads avoids relying on pandas' assignment being thread-safe.
    def _train_group(var_subset: list, device: str):
        g_artifacts, g_skipped, g_fold_models = {}, {}, {}
        g_val_pred = pd.Series(np.nan, index=va.index, dtype=float)
        for var in var_subset:
            art = train_variable_regressor_pooled(
                cached, train_years, var, train_c, va, hbf, cache_dir, device=device)
            if art is None:
                g_skipped[var] = "regressor training returned None or too few rows"
                continue
            g_artifacts[var] = art
            vmask = va["variable"] == var
            if vmask.any():
                g_val_pred.loc[vmask] = reg_mod.predict_variable_error(art, va[vmask])
            g_fold_models[var] = oof_fold_models(
                cached, train_years, var, train_c, hbf, fold_of, cache_dir, device=device)
        return g_artifacts, g_skipped, g_val_pred, g_fold_models

    half = max(1, len(variables) // 2)
    gpu_vars, cpu_vars = variables[:half], variables[half:]

    import torch
    gpu_available = torch.cuda.is_available()

    artifacts: dict = {}
    val_pred = pd.Series(np.nan, index=va.index, dtype=float)
    fold_models: dict = {}
    if gpu_available and cpu_vars:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_gpu = ex.submit(_train_group, gpu_vars, "cuda")
            fut_cpu = ex.submit(_train_group, cpu_vars, "cpu")
            results = [fut_gpu.result(), fut_cpu.result()]
    else:
        # No GPU (or nothing left for a second group) - everything on CPU, one group.
        results = [_train_group(variables, "cpu")]

    for g_artifacts, g_skipped, g_val_pred, g_fold_models in results:
        artifacts.update(g_artifacts)
        report.skipped_variables.update(g_skipped)
        val_pred = val_pred.combine_first(g_val_pred)
        fold_models.update(g_fold_models)
        for var, art in g_artifacts.items():
            report.regressor_metrics[var] = art.metrics

    report.modelled_variables = sorted(artifacts)
    if not artifacts:
        report.status = "failed"
        report.error = "no variable had enough paired rows to train a regressor"
        return report

    event_tr = build_pooled_train_events(
        cached, train_years, train_c, hbf, p90_error, bust_threshold, fold_models, fold_of,
        columns=needed)
    del fold_models  # only needed for event_tr's out-of-fold predictions, above
    gc.collect()
    event_va = pv.build_event_frame(va, val_pred, p90_error, bust_threshold, hbf)

    # Now safe to load: event_tr is built, fold_models is gone, and only the small saved
    # regressor artifacts (not the pooled training data) are still needed to score te.
    gc.collect()
    te = pd.read_parquet(cached[test_year], columns=sorted(needed))
    te = te[te["init_date"].isin(test_c)]
    test_pred = pd.Series(np.nan, index=te.index, dtype=float)
    if not te.empty:
        key = list(zip(te["region_id"].astype(str), te["season"].astype(str)))
        te["historical_bust_frequency_region_season"] = [hbf.get(k, np.nan) for k in key]
        for var, art in artifacts.items():
            tmask = te["variable"] == var
            if not tmask.any():
                continue
            test_pred.loc[tmask] = reg_mod.predict_variable_error(art, te[tmask])
            if tmask.sum() >= 5:
                art.metrics["test"] = reg_mod._evaluate(
                    te.loc[tmask, "abs_error"], test_pred[tmask])
    event_te = (pv.build_event_frame(te, test_pred, p90_error, bust_threshold, hbf)
               if not te.empty else pd.DataFrame())

    clf_art = clf_mod.train_bust_classifier(event_tr, event_va)
    report.classifier_metrics = dict(clf_art.metrics)
    if not event_te.empty and "y_bust" in event_te:
        proba_te = clf_mod.predict_bust_probability(clf_art, event_te)
        report.classifier_metrics["test"] = clf_mod._evaluate(event_te["y_bust"], proba_te)

    proba_va = (clf_mod.predict_bust_probability(clf_art, event_va)
               if len(event_va) else np.array([]))
    risk_cuts = compute_risk_bands(proba_va)
    thresholds = Thresholds(
        bust_threshold=bust_threshold, p90_error=p90_error, risk_band_cuts=risk_cuts,
        notes=[f"bust_threshold = 90th pct of event-grain ensemble-mean abs error, "
              f"pooled train years {train_years} ({len(train_c)} cycles)",
              f"risk bands from {len(proba_va)} validation events"],
    )
    report.thresholds = {"bust_threshold": bust_threshold, "p90_error": p90_error,
                         "risk_band_cuts": risk_cuts}

    for var, art in artifacts.items():
        registry.save_regressor(rid, var, art.model, art.feature_columns)
    registry.save_classifier(rid, clf_art.model, clf_art.feature_columns)
    registry.save_thresholds(rid, thresholds)
    registry.save_historical_bust_freq(rid, hbf)
    registry.save_metrics(rid, {"regressors": report.regressor_metrics,
                               "classifier": report.classifier_metrics})
    registry.save_manifest(rid, {
        "run_id": rid, "pooled_train_years": train_years, "test_year": test_year,
        "split_cycles": report.split_cycles, "modelled_variables": report.modelled_variables,
        "skipped_variables": report.skipped_variables,
    })

    report.status = "success"
    report.made_current = False
    report.promotion_note = ("not evaluated: full_retrain_pooled never promotes - "
                             "this is a comparison tool")
    report.seconds = (pd.Timestamp.now() - t0).total_seconds()
    return report
