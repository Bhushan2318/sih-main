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
import pickle
import shutil
import subprocess
import sys
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
    # Existence is not validity. Real crash 2026-09-21: this process was killed mid-write
    # during a disk squeeze, leaving paired_2000.parquet truncated at 1.59 GB of 3.65 GB;
    # every later run reused it unread, and the pooled run died nine hours on in
    # pooled_split, after re-caching seventeen other years for nothing.
    if path.exists():
        if _readable_parquet(path):
            _report_finite_repair(path)
            return path
        path.unlink()
    lo, hi = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
    paired, _ = _build_paired_in_chunks(init_date_min=lo, init_date_max=hi)
    if paired.empty:
        raise ValueError(f"no paired forecast+observation data for {year} in the store")
    # Written under a temporary name and renamed into place, so an interrupted write
    # leaves no file rather than a plausible-looking partial one - the same convention
    # parquet_store already uses for a batch.
    tmp = path.with_name(path.name + ".tmp")
    paired.to_parquet(tmp, index=False)
    del paired
    tmp.replace(path)
    _report_finite_repair(path)
    return path


def _report_finite_repair(path: Path) -> None:
    n = _ensure_finite_cache(path)
    if n:
        print(f"[pooled] {path.name}: {n} inf laf_spread_ratio values rewritten as missing",
              file=sys.stderr, flush=True)


# Columns whose inf has one known meaning, and what it becomes. laf_spread_ratio: pool
# spread over an own spread of exactly zero (members agreeing), which the feature code
# now yields as NaN - see compute_time_lagged_ensemble. Inf anywhere else is not
# understood, so it is refused rather than rewritten.
_INF_MEANS_MISSING = ("laf_spread_ratio",)
_FINITE_MARKER = b"sanket_finite_checked"


def _ensure_finite_cache(path: Path) -> int:
    """Rewrite a cached year so that `_INF_MEANS_MISSING` columns carry NaN where they
    carried +-inf, and refuse (ValueError, file untouched) inf in any other float column.
    Returns how many values were rewritten. Streams one row group at a time and replaces
    the file atomically; a checked file is marked in its footer, so every later call is a
    footer read.

    Why in place rather than a rebuild: the feature code was fixed to produce NaN where it
    produced inf (x/0, x > 0 - the only source of inf), so the repaired file is exactly
    what a rebuild would write, for minutes instead of ~50 per year. Real failure
    2026-09-21: 1.2-1.4 M inf per cached year (rainfall, soil moisture, humidity), and
    XGBoost refused humidity_pct outright."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    src = pq.ParquetFile(path)
    meta = dict(src.schema_arrow.metadata or {})
    if _FINITE_MARKER in meta:
        return 0
    floats = [f.name for f in src.schema_arrow if pa.types.is_floating(f.type)]
    unknown = sorted({c for i in range(src.metadata.num_row_groups)
                      for c in floats if c not in _INF_MEANS_MISSING
                      and (pc.sum(pc.is_inf(src.read_row_group(i, columns=[c])[c])).as_py() or 0)})
    if unknown:
        raise ValueError(f"{path.name}: inf in {unknown}, which have no known meaning for "
                         f"inf - refusing to rewrite; fix the feature code that produced it")

    schema = src.schema_arrow.with_metadata({**meta, _FINITE_MARKER: b"1"})
    tmp = path.with_name(path.name + ".finite.tmp")
    n_fixed = 0
    try:
        with pq.ParquetWriter(tmp, schema, compression="snappy") as writer:
            for i in range(src.metadata.num_row_groups):
                tb = src.read_row_group(i)
                for c in _INF_MEANS_MISSING:
                    if c not in tb.column_names:
                        continue
                    col = tb[c]
                    bad = pc.is_inf(col)
                    n_fixed += pc.sum(bad).as_py() or 0
                    fixed = pc.if_else(bad, pa.scalar(None, col.type), col)
                    tb = tb.set_column(tb.column_names.index(c), c, fixed)
                writer.write_table(tb.replace_schema_metadata(schema.metadata))
        del src
        if pq.ParquetFile(tmp).metadata.num_rows != pq.ParquetFile(path).metadata.num_rows:
            raise RuntimeError(f"{path.name}: row count changed while repairing inf")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return n_fixed


def _readable_parquet(path: Path) -> bool:
    """Whether the file has an intact Parquet footer, i.e. was written to completion."""
    import pyarrow.parquet as pq
    try:
        pq.ParquetFile(path).metadata
        return True
    except Exception:  # noqa: BLE001 - any unreadable footer means rebuild it
        return False


MAX_VAL_CYCLES = 365


# How many training cycles one variable's regressor (and each OOF fold model) fits on.
# Measured 2026-09-21 on the real caches, one variable: XGBoost's QuantileDMatrix peaked
# at 17.32 GB of commit for one year and 26.11 GB for two - thread count made no
# difference (25.89 GB at nthread=4) and the parquet iterator alone stays flat at ~8.6 GB
# - so seventeen years at every-day density (6,193 cycles) cannot fit this machine's
# 83 GB commit limit on either device. 2,000 is about a third of that pool. Every year and
# season stays in the sample; thresholds, validation, test and classifier events are
# unaffected and still use every cycle.
MAX_FIT_CYCLES = 2000


def fit_cycles(train_cycles: set, cap: "int | None" = None) -> set:
    """The training cycles the per-variable fits use: all of them up to `cap`, otherwise
    a fixed-seed uniform sample of `cap`. Random, not every k-th day: `assign_folds`
    numbers sorted cycles `i % 3`, so a regular stride would put every kept cycle in one
    OOF fold and leave that fold's model nothing to train on."""
    cap = MAX_FIT_CYCLES if cap is None else cap
    if len(train_cycles) <= cap:
        return set(train_cycles)
    ordered = sorted(train_cycles)
    keep = np.random.default_rng(42).choice(len(ordered), size=cap, replace=False)
    return {ordered[i] for i in keep}


def fit_chunks(train_cycles: set, cap: "int | None" = None) -> list:
    """Every training cycle, split into disjoint chunks of at most `cap` - the staged fit
    boosts them one after another into one booster, so the whole pool is used while no
    single QuantileDMatrix holds more than `cap` cycles. A fixed-seed shuffle, not
    contiguous runs of days, so every chunk spans every year, season and OOF fold."""
    cap = MAX_FIT_CYCLES if cap is None else cap
    if len(train_cycles) <= cap:
        return [set(train_cycles)]
    ordered = sorted(train_cycles)
    k = -(-len(ordered) // cap)
    perm = np.random.default_rng(42).permutation(len(ordered))
    return [{ordered[i] for i in part} for part in np.array_split(perm, k)]


def _split_rounds(total: int, k: int) -> list:
    """`total` boosting rounds over `k` chunks, as evenly as possible."""
    base, extra = divmod(total, k)
    return [base + (1 if i < extra else 0) for i in range(k)]


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
    train, val = pre_cycles[:a], pre_cycles[a:]
    # Validation is sized in absolute cycles, not as a share of a pool that grows with
    # every year added. At seventeen years 15% is 1,093 cycles - 229 million rows, and
    # pyarrow could not materialise that frame at all (one 28.4 GB allocation on a 23.7 GB
    # machine, the crash on 2026-09-21). MAX_VAL_CYCLES is still nearly double the 193
    # cycles the three-year runs validated on, so the signal is not weaker; what a bigger
    # share would have bought is redundancy, and every surplus cycle is worth more in
    # training. Nothing is discarded - the surplus is chronologically earlier than the
    # cycles kept, so it joins training without leaking the validation window.
    if len(val) > MAX_VAL_CYCLES:
        train = train + val[:-MAX_VAL_CYCLES]
        val = val[-MAX_VAL_CYCLES:]
    return set(train), set(val), test_cycles


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
    read_cols = list(dict.fromkeys(_THIN_STATS_COLUMNS + event_cols + ["init_date"]))

    # Two bounded passes, never one concatenated frame. Real crash 2026-09-21: at 17
    # pooled years `thin` is ~0.8 billion rows; casting one year's region_id to str asked
    # for an 8.07 GiB <U29 array, and compute_historical_bust_frequency would then have
    # copied the whole frame and walked it with itertuples(). Both are fine at the three
    # years this was written for and impossible at seventeen. Percentiles need every
    # value, so pass 1 keeps abs_error alone (float32, per variable); the rest is counting,
    # which sums per year in pass 2.
    per_var: dict = {}
    event_frames = []
    for year in train_years:
        df = pd.read_parquet(cached_paths[year], columns=read_cols)
        df = df[df["init_date"].isin(train_cycles)]
        if df.empty:
            continue
        # Native dtype, never a cast: rounding abs_error to float32 moved the 75th
        # percentile enough to flip borderline rows and changed hbf in the 3rd decimal.
        err = df["abs_error"].to_numpy()
        codes, cats = _as_codes(df["variable"])
        for i, cat in enumerate(cats):
            vals = err[(codes == i) & ~np.isnan(err)]
            if vals.size:
                per_var.setdefault(str(cat), []).append(vals)
        event_frames.append(_event_mean_error(df))
        del df, err, codes
    if not per_var:
        return {}, {}, {}

    thr_large, p90_error = {}, {}
    for var, chunks in per_var.items():
        vals = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        chunks.clear()
        thr_large[var] = float(np.percentile(vals, 75.0))
        p90_error[var] = float(np.percentile(vals, 90.0))
        del vals
    del per_var

    # An error is "large" against its own variable's 75th percentile; a variable with no
    # threshold gets inf, so it is never large - matching thr.get(var, np.inf) before.
    # A NaN abs_error compares False, and still counts in its group's denominator.
    large: dict = {}
    total: dict = {}
    for year in train_years:
        df = pd.read_parquet(
            cached_paths[year],
            columns=list(dict.fromkeys(_THIN_STATS_COLUMNS + ["init_date"])))
        df = df[df["init_date"].isin(train_cycles)]
        if df.empty:
            continue
        codes, cats = _as_codes(df["variable"])
        by_code = np.array([thr_large.get(str(c), np.inf) for c in cats])
        is_large = df["abs_error"].to_numpy() > by_code[codes]
        # Counted over integer codes with bincount, never over strings. Converting one
        # year's region_id to objects is itself an 8.27 GiB <U29 array (the 2026-09-21
        # crash, and then the same crash again in the first version of this rewrite):
        # pandas materialises one value per row, and there are 76.5 million of them.
        r_codes, r_cats = _codes_with_missing(df["region_id"])
        s_codes, s_cats = _codes_with_missing(df["season"])
        n_seasons = len(s_cats)
        key = r_codes.astype(np.int64) * n_seasons + s_codes
        n_keys = len(r_cats) * n_seasons
        tot = np.bincount(key, minlength=n_keys)
        lrg = np.bincount(key[is_large], minlength=n_keys)
        for idx in np.nonzero(tot)[0]:
            k = (r_cats[idx // n_seasons], s_cats[idx % n_seasons])
            large[k] = large.get(k, 0) + int(lrg[idx])
            total[k] = total.get(k, 0) + int(tot[idx])
        del df, codes, is_large, r_codes, s_codes, key, tot, lrg

    hbf = {k: float(large[k] / total[k]) for k in total if total[k]}
    event_err = pd.concat(event_frames, ignore_index=True)
    bust_threshold = compute_error_thresholds(event_err, percentile=90.0)
    return hbf, p90_error, bust_threshold


def _drop_stale_caches(cache_dir: Path, years: list) -> list:
    """Delete any cached year narrower than its peers, so it is rebuilt with the columns
    the current code produces.

    A cache file is only valid for the feature set that existed when it was written. Real
    crash 2026-09-21: paired_2016 and paired_2017 were built before the district
    descriptors landed - 27 columns against the other sixteen years' 44 - and
    `_feature_columns_for` reads the schema of one year to decide what to read from all of
    them. The run died on `No match for FieldRef.Name(area_km2)` after eight hours of
    caching. Comparing schemas costs one footer read per year."""
    import pyarrow.parquet as pq
    schemas = {}
    for year in years:
        path = cache_dir / f"paired_{year}.parquet"
        if not path.exists():
            continue
        try:
            schemas[year] = set(pq.ParquetFile(path).schema_arrow.names)
        except Exception:  # noqa: BLE001 - unreadable is handled by cache_year itself
            continue
    if len(schemas) < 2:
        return []
    widest = max(len(n) for n in schemas.values())
    stale = sorted(y for y, names in schemas.items() if len(names) < widest)
    for year in stale:
        (cache_dir / f"paired_{year}.parquet").unlink()
    return stale


def _codes_with_missing(col: "pd.Series"):
    """Codes and their category names as strings, with missing values given their own
    slot named "nan" - which is what `.astype(str)` called them before."""
    codes, cats = _as_codes(col)
    names = [str(c) for c in cats] + ["nan"]
    codes = np.where(codes < 0, len(names) - 1, codes)
    return codes, names


def _as_codes(col: "pd.Series"):
    """Integer codes plus their categories, for a categorical or a plain column alike -
    so a per-variable lookup never materialises one string per row."""
    if isinstance(col.dtype, pd.CategoricalDtype):
        return col.cat.codes.to_numpy(), list(col.cat.categories)
    cats, codes = np.unique(col.to_numpy(), return_inverse=True)
    return codes, list(cats)


def _feature_columns_for(cached_paths: dict, years: list) -> list:
    """Which columns `reg_mod.feature_columns` would pick for this variable's frame,
    without reading any row data - schema only, plus the one feature attached after
    caching (`historical_bust_frequency_region_season`, computed globally in
    `pooled_stats` rather than known at cache time)."""
    import pyarrow.parquet as pq
    names = set(pq.ParquetFile(cached_paths[years[0]]).schema_arrow.names)
    names.add("historical_bust_frequency_region_season")
    return reg_mod.feature_columns(pd.DataFrame(columns=sorted(names)))


def attach_hbf_column(df: "pd.DataFrame", hbf: dict,
                      out_col: str = "historical_bust_frequency_region_season") -> "pd.DataFrame":
    """Attach the per-(region, season) historical bust frequency feature, vectorized in
    category-code space - never materialising a per-row string array, at any row count.

    History: the original `[hbf.get(k, np.nan) for k in key]` list comprehension was
    replaced (2026-09-15, v10) by a merge on a `region_id.astype(str)` key frame, which
    fixed the boxed-list allocation but not the underlying problem - `.astype(str)` on a
    Categorical still builds a genuinely dense fixed-width unicode array, one entry per
    row. That merge version worked in `pooled_stats` (a thin, freshly-read 2-3 column
    frame) but real crash 2026-09-17: called here on the FULL wide per-year training
    frame - already resident with every feature column - the same `.astype(str)` needed
    an 8.27 GiB contiguous allocation for one 76.5M-row year and hit `ArrayMemoryError`,
    the same "free but not contiguous" signature as every earlier crash in this file.

    `region_id`/`season` are already Categorical (contracts.py), which means the actual
    per-row data is already a small int codes array plus a tiny categories index - the
    string array the merge approach built was pure waste. Building a `region x season`
    lookup table (at most a few thousand cells - 666 districts x a handful of seasons,
    independent of row count) and indexing it with the existing `.cat.codes` arrays does
    the identical lookup with zero additional string materialisation, at any scale."""
    out = np.full(len(df), np.nan, dtype=np.float64)
    if not hbf:
        df[out_col] = out
        return df

    def _codes_and_categories(col: "pd.Series"):
        if isinstance(col.dtype, pd.CategoricalDtype):
            return col.cat.codes.to_numpy(), col.cat.categories
        # Not categorical (e.g. a hand-built test frame) - the fallback path is the one
        # place this still costs a string array, sized to the number of DISTINCT values,
        # not the row count, so it stays cheap even here.
        cat = col.astype(str).astype("category")
        return cat.cat.codes.to_numpy(), cat.cat.categories

    region_codes, region_cats = _codes_and_categories(df["region_id"])
    season_codes, season_cats = _codes_and_categories(df["season"])
    region_pos = {r: i for i, r in enumerate(region_cats)}
    season_pos = {s: i for i, s in enumerate(season_cats)}

    lut = np.full((len(region_cats), len(season_cats)), np.nan, dtype=np.float64)
    for (r, s), v in hbf.items():
        ri, si = region_pos.get(r), season_pos.get(s)
        if ri is not None and si is not None:
            lut[ri, si] = v

    valid = (region_codes >= 0) & (season_codes >= 0)
    out[valid] = lut[region_codes[valid], season_codes[valid]]
    df[out_col] = out
    return df


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
        self.batches = 0  # batches handed to XGBoost; 0 means this chunk had no rows
        # No cache_prefix: QuantileDMatrix builds its quantile sketch batch by batch and
        # never writes batches to disk itself - passing one is a QuantileDMatrix-specific
        # ValueError, unlike the plain external-memory DMatrix this class is modelled on.
        super().__init__()

    def next(self, input_data) -> int:
        if self._i == len(self._paths):
            return 0
        # The variable filter goes into the read, so the other seven variables' rows are
        # dropped row group by row group and never become one full-year frame. Real
        # failure 2026-09-21: reading the whole year and filtering in pandas, 17 years x
        # every QuantileDMatrix pass, drove the worker past 43 GB of commit before its
        # first boosting round. Measured on real 2009+2010: peak 29.87 GB -> 9.38 GB,
        # 69 s -> 10 s, identical rows and abs_error sums.
        df = pd.read_parquet(self._paths[self._i], columns=self._read_cols,
                             filters=[("variable", "==", self._variable)])
        df = df[(df["variable"] == self._variable) & (df["init_date"].isin(self._cycles))]
        self._i += 1
        if df.empty:
            return 1
        # No .copy() before this assignment: boolean-mask filtering above already
        # produced a new frame, and an extra .copy() here forces pandas to consolidate
        # its blocks into one contiguous array per dtype - a second full-sized allocation
        # on top of the filtered frame already in memory. Measured real crash 2026-09-13,
        # ArrayMemoryError on a single cached year's own frame.
        df = attach_hbf_column(df, self._hbf)
        X = reg_mod._prep_X(df, self._feature_cols)
        input_data(data=X, label=df["abs_error"].to_numpy())
        self.batches += 1
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


def _fit_booster(cached_paths: dict, train_years: list, variable: str, cycle_chunks: list,
                 cols: list, hbf: dict, cache_dir: Path, device: str,
                 n_estimators: "int | None" = None):
    """One booster boosted over `cycle_chunks` in turn: each chunk gets its own
    QuantileDMatrix, freed before the next is built, and its share of `n_estimators`
    rounds continues the same booster. One chunk is the ordinary single fit. Returns
    `(booster or None, rows seen)`; None when there are too few rows to fit at all."""
    n_estimators = n_estimators or reg_mod.XGB_PARAMS["n_estimators"]
    chunks = [c for c in cycle_chunks if c]
    params = _xgb_train_params(device)
    booster, n_rows = None, 0
    rounds_left = n_estimators
    for i, cycles in enumerate(chunks):
        it = _YearDataIter(cached_paths, train_years, variable, cycles, cols, hbf, cache_dir)
        try:
            dtrain = xgb.QuantileDMatrix(it, enable_categorical=True)
        except xgb.core.XGBoostError:
            # A chunk whose cycles hold no rows for this variable: XGBoost raises on an
            # iterator that never produced a batch rather than returning an empty matrix.
            # Skip it; its rounds go to the chunks that remain.
            if it.batches:
                raise
            del it
            continue
        rows = int(dtrain.num_row())
        if rows < reg_mod.MIN_ROWS and len(chunks) == 1:
            del it, dtrain
            gc.collect()
            return None, rows
        n_rows += rows
        # Remaining rounds over remaining chunks, so a skipped chunk costs no rounds.
        rounds = _split_rounds(rounds_left, len(chunks) - i)[0]
        rounds_left -= rounds
        booster = xgb.train(params, dtrain, num_boost_round=rounds, xgb_model=booster)
        # Freed before the next chunk's matrix is built: two resident at once is exactly
        # the memory this staging exists to avoid.
        del it, dtrain
        gc.collect()
    if booster is None or n_rows < reg_mod.MIN_ROWS:
        return None, n_rows
    return booster, n_rows


def _as_chunks(train_cycles) -> list:
    """`train_cycles` as a list of chunks: a set is one chunk (the ordinary single fit);
    a list is already the staged fit's chunks (`fit_chunks`)."""
    return list(train_cycles) if isinstance(train_cycles, list) else [set(train_cycles)]


def train_variable_regressor_pooled(cached_paths: dict, train_years: list, variable: str,
                                    train_cycles, val_df: "pd.DataFrame",
                                    hbf: dict, cache_dir: Path,
                                    device: str = "cpu") -> "reg_mod.RegressorArtifact | None":
    """The pooled-training equivalent of `regressors.train_variable_regressor`: same
    params, same features, fit via an external-memory `QuantileDMatrix` instead of a
    single in-memory `.fit()` so `train_years` is never all resident at once."""
    cols = _feature_columns_for(cached_paths, train_years)
    booster, n_train = _fit_booster(cached_paths, train_years, variable,
                                    _as_chunks(train_cycles), cols, hbf, cache_dir, device)
    if booster is None:
        return None
    model = _booster_to_sklearn(booster, xgb.XGBRegressor)

    va = val_df[val_df["variable"] == variable]
    del booster  # see oof_fold_models for why: real fragmentation crashes
    gc.collect()
    # No train-split metric here, unlike the single-frame path: recomputing it would mean
    # a second full external-memory pass over every pooled year just to score rows the
    # model already fit on. Held-out val/test - what the promotion gate actually reads -
    # are unaffected.
    metrics = {}
    if len(va):
        va = attach_hbf_column(va.copy(), hbf)
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
                    train_cycles, hbf: dict, fold_of: dict, cache_dir: Path,
                    device: str = "cpu"):
    """fold id -> (fitted, sklearn-wrapped booster, feature columns), each excluding its
    own fold's cycles. Used only to compute out-of-fold predictions for the training
    events the classifier trains on - never saved as an artifact."""
    chunks = _as_chunks(train_cycles)
    if sum(len(c) for c in chunks) < 2:
        return {}
    n_splits = len(set(fold_of.values()))
    cols = _feature_columns_for(cached_paths, train_years)
    models = {}
    for fold in range(n_splits):
        fold_chunks = [{c for c in chunk if fold_of[c] != fold} for chunk in chunks]
        booster, _ = _fit_booster(cached_paths, train_years, variable, fold_chunks, cols,
                                  hbf, cache_dir, device)
        if booster is None:
            continue
        models[fold] = (_booster_to_sklearn(booster, xgb.XGBRegressor), cols)
        # Explicit cleanup, not left to Python's own GC timing: real repeated crashes
        # 2026-09-14, a small (~900 MB) pyarrow malloc failing after ~1.5-2 hours of a
        # process that had been running fine - the signature of fragmentation, not a
        # leak, from many short-lived DataIter/QuantileDMatrix/Booster objects (each
        # wrapping native pyarrow/XGBoost C++ allocations Python's cyclic GC does not
        # prioritise) accumulating across dozens of fits without being freed promptly.
        # _fit_booster frees each chunk's matrix itself.
        del booster
        gc.collect()
    return models


_YEAR_EVENTS_WORKER_SCRIPT = (Path(__file__).resolve().parents[2] / "scripts"
                              / "_build_pooled_year_events_worker.py")


def _run_year_events_subprocess(cached_path: Path, train_cycles: set, hbf: dict,
                                p90_error: dict, bust_threshold: dict,
                                fold_models: dict, fold_of: dict, columns) -> "pd.DataFrame":
    """One year's raw-frame load + OOF-predict + `build_event_frame` in a fresh process -
    see _build_pooled_year_events_worker.py's docstring for why: real crash 2026-09-15
    (v9), ArrayMemoryError inside pandas' own groupby machinery on a 76.7M-row year,
    the same fragmentation signature the per-variable training subprocess fix (below)
    already solved for the training phase. Only the small, event-reduced result needs
    to survive back into the parent.

    Real crash 2026-09-17/18: process isolation alone was not enough - the identical
    ~586 MiB groupby allocation failed again, twice, each ~1.8 hours into a real 3-year
    pool (fixed in the worker itself by freeing fold_models before the call - see its
    docstring). Retrying once here too, same reasoning as `_run_variable_subprocess`'s
    retry: a fresh OS process is a genuinely different memory state, not a hope, and at
    ~1.8 hours to reach this point, losing the whole job to one allocation is a far
    worse trade than one retry costing a few extra seconds of subprocess startup."""
    job = {"cached_path": cached_path, "train_cycles": train_cycles, "hbf": hbf,
          "p90_error": p90_error, "bust_threshold": bust_threshold,
          "fold_models": fold_models, "fold_of": fold_of, "columns": columns}
    errors = []
    for attempt in range(2):
        result = _run_worker_subprocess(_YEAR_EVENTS_WORKER_SCRIPT, job)
        if not result.get("error"):
            return result["event_frame"]
        errors.append(result["error"])
    raise RuntimeError(
        f"year-events worker failed on both attempts:\n" + "\n---\n".join(errors))


# Forecast dates per batch when one year's event frame is built. Measured on real 2015:
# the whole year's needed columns peak at 41.6 GB of commit in pandas, and the worker died
# on a 1.14 GiB allocation on both attempts (2026-09-22). 100 dates is about a quarter.
EVENTS_BATCH_CYCLES = 100


def _cycle_batches(cycles: set, max_per_batch: int) -> list:
    """`cycles` in date order, in consecutive runs of at most `max_per_batch`."""
    ordered = sorted(cycles)
    return [ordered[i:i + max_per_batch] for i in range(0, len(ordered), max_per_batch)]


def year_event_frame(cached_path: Path, train_cycles: set, hbf: dict, p90_error: dict,
                     bust_threshold: dict, fold_models: dict, fold_of: dict, columns,
                     max_cycles_per_batch: int = EVENTS_BATCH_CYCLES) -> "pd.DataFrame":
    """One cached year's training events with out-of-fold regressor predictions,
    built one batch of forecast dates at a time and concatenated.

    Exact, not an approximation: EVENT_KEYS include init_date and build_event_frame only
    groups within an event, so a batch of dates yields exactly its own events - the same
    argument that makes the per-year split exact."""
    import pyarrow.parquet as pq

    present = pd.to_datetime(pq.read_table(cached_path, columns=["init_date"])
                             .column("init_date").unique().to_pandas())
    mine = set(present) & {pd.Timestamp(c) for c in train_cycles}
    read_cols = sorted(columns) if columns else None
    frames = []
    for batch in _cycle_batches(mine, max_cycles_per_batch):
        df = pd.read_parquet(cached_path, columns=read_cols,
                             filters=[("init_date", "in", list(batch))])
        if df.empty:
            continue
        df = attach_hbf_column(df, hbf)
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
        frames.append(_events_float32(pv.build_event_frame(
            df, oof, p90_error, bust_threshold, hbf, copy_input=False)))
        del df, oof
        gc.collect()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _events_float32(df: "pd.DataFrame") -> "pd.DataFrame":
    """Float columns of an event frame as float32 - the precision XGBoost trains and
    predicts in anyway, so no model sees a different number. Integer labels and
    categoricals are untouched. Halves the frames the parent holds for the classifier."""
    floats = df.select_dtypes(include=["float64"]).columns
    if len(floats):
        df[floats] = df[floats].astype(np.float32)
    return df


def val_event_frame(spill_dir: Path, val_pred, hbf: dict, p90_error: dict,
                    bust_threshold: dict,
                    max_cycles_per_batch: int = EVENTS_BATCH_CYCLES) -> "pd.DataFrame":
    """Validation events from the spilled validation frame (see _build_val_frame_worker),
    one batch of forecast dates at a time. `val_pred` is indexed by `_va_row`, the shared
    row numbering the spill carries. Same exactness argument as year_event_frame."""
    import pyarrow.dataset as ds

    val_pred = np.asarray(val_pred, dtype=float)
    dataset = ds.dataset(spill_dir, format="parquet", partitioning="hive")
    cycles = set(pd.to_datetime(dataset.to_table(columns=["init_date"])
                                .column("init_date").unique().to_pandas()))
    frames = []
    for batch in _cycle_batches(cycles, max_cycles_per_batch):
        df = pd.read_parquet(spill_dir, filters=[("init_date", "in", list(batch))])
        if df.empty:
            continue
        df = df.set_index("_va_row").sort_index()
        pred = pd.Series(val_pred[df.index.to_numpy()], index=df.index)
        frames.append(_events_float32(pv.build_event_frame(
            df, pred, p90_error, bust_threshold, hbf, copy_input=False)))
        del df, pred
        gc.collect()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def test_event_frame(cached_path: Path, test_cycles: set, hbf: dict, p90_error: dict,
                     bust_threshold: dict, artifacts: dict, columns,
                     max_cycles_per_batch: int = EVENTS_BATCH_CYCLES) -> tuple:
    """The held-out year's events and each variable's test metrics, one batch of
    forecast dates at a time - same reasoning and same exactness as year_event_frame.
    Metrics are not averaged across batches: each variable's targets and predictions are
    kept (one value per row of that variable, the dtypes the whole-year path used) and
    scored once at the end, so they equal the whole-year computation."""
    import pyarrow.parquet as pq

    present = pd.to_datetime(pq.read_table(cached_path, columns=["init_date"])
                             .column("init_date").unique().to_pandas())
    mine = set(present) & {pd.Timestamp(c) for c in test_cycles}
    read_cols = sorted(columns) if columns else None
    frames, y_parts, p_parts = [], {}, {}
    for batch in _cycle_batches(mine, max_cycles_per_batch):
        df = pd.read_parquet(cached_path, columns=read_cols,
                             filters=[("init_date", "in", list(batch))])
        if df.empty:
            continue
        df = attach_hbf_column(df, hbf)
        pred = pd.Series(np.nan, index=df.index, dtype=float)
        for var, art in artifacts.items():
            tmask = df["variable"] == var
            if not tmask.any():
                continue
            p = reg_mod.predict_variable_error(art, df[tmask])
            pred.loc[tmask] = p
            y_parts.setdefault(var, []).append(df.loc[tmask, "abs_error"].to_numpy())
            p_parts.setdefault(var, []).append(np.asarray(p, dtype=float))
        frames.append(_events_float32(pv.build_event_frame(
            df, pred, p90_error, bust_threshold, hbf, copy_input=False)))
        del df, pred
        gc.collect()
    metrics = {}
    for var, ys in y_parts.items():
        y = np.concatenate(ys)
        if len(y) >= 5:
            metrics[var] = reg_mod._evaluate(y, np.concatenate(p_parts[var]))
    events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return events, metrics


def build_pooled_train_events(cached_paths: dict, train_years: list, train_cycles: set,
                              hbf: dict, p90_error: dict, bust_threshold: dict,
                              fold_models: dict, fold_of: dict,
                              columns: "set | None" = None) -> "pd.DataFrame":
    """`event_tr`, assembled one cached year at a time, each year's own raw-frame work
    (load, attach OOF predictions, reduce to event grain) run in a fresh subprocess -
    see _run_year_events_subprocess. Concatenating the (small) per-year event frames
    afterward is exactly `pv.build_event_frame` on the full multi-year `tr` would
    produce, since EVENT_KEYS never crosses a year boundary."""
    frames = []
    for year in train_years:
        frame = _run_year_events_subprocess(
            cached_paths[year], train_cycles, hbf, p90_error, bust_threshold,
            fold_models, fold_of, columns)
        if frame is not None and not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _run_worker_subprocess(script: Path, job: dict) -> dict:
    """Run `script` with `job` pickled as its stdin file, writing the child's stdout and
    stderr to files on disk rather than capturing them in memory. Real crash 2026-09-15
    (v11): `subprocess.run(..., capture_output=True)` buffers the entire child output as
    Python strings with no upper bound, and the PARENT process's own private memory
    ballooned to 35 GB (almost no working set - virtual, not resident) after dispatching
    only two variables' worth of subprocess calls, driving free system memory to ~250
    MB before it had to be killed by hand. Whatever was producing that much output, a
    file on disk has no such ceiling; only the small pickled result dict comes back into
    this process."""
    with tempfile.TemporaryDirectory() as td:
        job_path, out_path = Path(td) / "job.pkl", Path(td) / "out.pkl"
        stdout_path, stderr_path = Path(td) / "stdout.log", Path(td) / "stderr.log"
        with open(job_path, "wb") as f:
            pickle.dump(job, f)
        with open(stdout_path, "wb") as out_f, open(stderr_path, "wb") as err_f:
            proc = subprocess.run(
                [sys.executable, str(script), "--job", str(job_path), "--out", str(out_path)],
                stdout=out_f, stderr=err_f)
        if not out_path.exists():
            tail = stderr_path.read_text(errors="replace")[-4000:] if stderr_path.exists() else ""
            return {"error": f"worker produced no output, rc={proc.returncode}: {tail}"}
        with open(out_path, "rb") as f:
            return pickle.load(f)


_WORKER_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "_train_pooled_variable_worker.py"
_VAL_FRAME_WORKER_SCRIPT = (Path(__file__).resolve().parents[2] / "scripts"
                            / "_build_val_frame_worker.py")


def _run_val_frame_subprocess(cached: dict, val_years: list, val_cycles: set,
                              columns: set, hbf: dict, out_dir: Path) -> dict:
    """Build the validation frame in a throwaway process that spills it to `out_dir`.

    The parent must be lean when it spawns a per-variable training worker: that worker
    needs one ~9.19 GB allocation for XGBoost's quantised matrix over seventeen years,
    and a parent holding this frame sat at 16.8 GB of 23.7 GB, so all eight variables
    failed that malloc (2026-09-21). Freeing the frame in-process is not enough - the
    read arena is not returned to the OS - so it is never allocated here at all."""
    job = {"cached": cached, "val_years": val_years, "val_cycles": val_cycles,
           "columns": columns, "hbf": hbf, "out_dir": str(out_dir)}
    return _run_worker_subprocess(_VAL_FRAME_WORKER_SCRIPT, job)


def _read_va_var(spill_dir: Path, variable: str) -> "pd.DataFrame":
    """One variable's validation rows, read straight from its own partition."""
    part = Path(spill_dir) / f"variable={variable}"
    if not part.exists():
        return pd.DataFrame()
    df = pd.read_parquet(part)
    df["variable"] = variable
    return df.set_index("_va_row")


def _run_variable_subprocess(cached: dict, train_years: list, variable: str,
                             train_cycles: set, va_var: "pd.DataFrame", hbf: dict,
                             cache_dir: Path, device: str, fold_of: dict) -> dict:
    """Train one variable's regressor + its OOF fold models in a brand-new process, and
    return the result as a plain dict. See _train_pooled_variable_worker.py's docstring
    for why this is a subprocess and not a function call: the process exit is what
    actually reclaims the native (pyarrow/XGBoost) memory this does, which repeated
    `gc.collect()` calls in a long-lived process could not - real crashes 2026-09-14.

    Real crash 2026-09-16: humidity_pct's worker died with a hard OS-level kill -
    STATUS_STACK_BUFFER_OVERRUN (0xC0000409) once, 0xFFFFFFFF another time - specifically
    when `train_years` contained exactly one of {2016, 2017} without the other. Both are
    native crashes below Python (the worker's own `except Exception` never runs; no
    stderr is written before the OS kills it), so there is nothing to catch here, only to
    retry around. A `humidity_pct: worker produced no output` in skipped_variables used
    to mean the whole variable silently dropped from that run - it is now missing from
    the feature set AND the bust-label definition (an event busts if any of ~8 variables
    exceeds its own p90), which is a real, unflagged degradation of the run, not a
    graceful skip. Retry once on the same device (a one-off native fault, e.g. a
    transient CUDA/driver hiccup, need not repeat); if it fails twice and the device was
    CUDA, retry once more on CPU - if the crash is specific to the CUDA path for this
    variable/year combination, CPU sidesteps it entirely rather than losing the variable.
    Every attempt is recorded in the returned dict's `skipped`/`error` message even on
    eventual success, so a run that needed a fallback is visible, not indistinguishable
    from one that never had a problem."""
    def attempt(dev: str) -> dict:
        job = {"cached": cached, "train_years": train_years, "variable": variable,
              "train_cycles": train_cycles, "va_var": va_var, "hbf": hbf,
              "cache_dir": cache_dir, "device": dev, "fold_of": fold_of}
        return _run_worker_subprocess(_WORKER_SCRIPT, job)

    # A caught exception (the worker pickles `error`) is a failed attempt too, not only a
    # hard crash with no pickle. Real 2026-09-21: a MemoryError the worker caught came
    # back as a result, was taken as finished, and the variable was dropped unretried.
    # Each attempt is printed as it ends - to the job's stderr log - so a retry or a skip
    # is visible while the run is still going, not only in the report hours later.
    attempts_log = []
    devices_to_try = [device, device] + (["cpu"] if device != "cpu" else [])
    result = None
    for i, dev in enumerate(devices_to_try):
        result = attempt(dev)
        if "artifact" in result and not result.get("error"):
            print(f"[pooled] {variable} attempt {i + 1} device={dev} ok"
                  + (f" (skipped: {result['skipped']})" if result.get("skipped") else ""),
                  file=sys.stderr, flush=True)
            if i > 0:
                note = (f"succeeded on attempt {i + 1} (device={dev}) after: "
                        f"{' | '.join(attempts_log)}")
                if result.get("skipped"):
                    result["skipped"] = f"{result['skipped']} [{note}]"
            return result
        err = str(result.get("error"))
        print(f"[pooled] {variable} attempt {i + 1} device={dev} FAILED: {err[-1500:]}",
              file=sys.stderr, flush=True)
        attempts_log.append(f"attempt {i + 1} device={dev}: {err[-1500:]}")

    msg = f"worker failed on every attempt - {' | '.join(attempts_log)}"
    return {"artifact": None, "val_pred": None, "fold_models": {},
           "skipped": msg, "error": msg}


def _variable_checkpoint_path(cache_dir: Path, variable: str, train_years: list,
                              train_cycles, val_cycles: set) -> Path:
    """Where one variable's finished worker result is kept between runs.

    At seventeen years a variable takes about an hour, and the results used to live only
    in the parent's memory: a failure in any later stage threw every finished variable
    away (2026-09-21, four finished CUDA variables lost with the run). The key covers
    everything the result depends on - the pool, the exact train and validation cycles
    (validation row numbering follows from them), and the model parameters - so a
    changed split or changed params never picks up a stale result."""
    import hashlib
    h = hashlib.sha256()
    h.update(repr(sorted(train_years)).encode())
    if isinstance(train_cycles, list):  # staged fit: the chunking itself is part of the key
        h.update(b"staged")
        for chunk in train_cycles:
            h.update(repr(sorted(str(c) for c in chunk)).encode())
    else:
        h.update(repr(sorted(str(c) for c in train_cycles)).encode())
    h.update(repr(sorted(str(c) for c in val_cycles)).encode())
    h.update(repr(sorted(reg_mod.XGB_PARAMS.items())).encode())
    return Path(cache_dir) / "_variable_checkpoints" / f"{variable}_{h.hexdigest()[:16]}.pkl"


def _load_variable_checkpoint(path: Path) -> "dict | None":
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            result = pickle.load(f)
    except Exception:  # noqa: BLE001 - an unreadable checkpoint just means retrain
        return None
    return result if result.get("artifact") is not None else None


def _save_variable_checkpoint(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(result, f)
    tmp.replace(path)


_VAL_EVENTS_WORKER_SCRIPT = (Path(__file__).resolve().parents[2] / "scripts"
                             / "_build_pooled_val_events_worker.py")


def _run_val_events_subprocess(spill_dir: Path, val_pred, hbf: dict, p90_error: dict,
                               bust_threshold: dict) -> "pd.DataFrame":
    """Validation events in a fresh process - see _build_pooled_val_events_worker.py.
    Retried once, like the year-events worker: a fresh process is a different memory
    state, and this runs hours into a seventeen-year job."""
    job = {"spill_dir": str(spill_dir), "val_pred": np.asarray(val_pred, dtype=float),
           "hbf": hbf, "p90_error": p90_error, "bust_threshold": bust_threshold}
    errors = []
    for _ in range(2):
        result = _run_worker_subprocess(_VAL_EVENTS_WORKER_SCRIPT, job)
        if not result.get("error"):
            return result["event_frame"]
        errors.append(str(result["error"])[-3000:])
    raise RuntimeError("val-events worker failed on both attempts:\n"
                       + "\n---\n".join(errors))


_FINALIZE_WORKER_SCRIPT = (Path(__file__).resolve().parents[2] / "scripts"
                           / "_finalize_pooled_run_worker.py")
_TEST_EVENTS_WORKER_SCRIPT = (Path(__file__).resolve().parents[2] / "scripts"
                              / "_build_pooled_test_events_worker.py")


def _run_test_events_subprocess(cached_path: Path, test_cycles: set, hbf: dict,
                                p90_error: dict, bust_threshold: dict, artifacts: dict,
                                columns) -> tuple:
    """The held-out test year's raw-frame load + predict + `build_event_frame`, in a
    fresh process - same reasoning as _run_year_events_subprocess, applied preemptively
    since the test year carries the identical full-year row-count risk. Returns
    `(event_frame, test_metrics)`; `test_metrics` is `{variable: metrics_dict}`, merged
    into each artifact's own `.metrics["test"]` by the caller since the artifact objects
    living in the parent are not the same objects the subprocess touched."""
    job = {"cached_path": cached_path, "test_cycles": test_cycles, "hbf": hbf,
          "p90_error": p90_error, "bust_threshold": bust_threshold,
          "artifacts": artifacts, "columns": columns}
    result = _run_worker_subprocess(_TEST_EVENTS_WORKER_SCRIPT, job)
    if result.get("error"):
        raise RuntimeError(f"test-events worker failed:\n{result['error']}")
    return result["event_frame"], result["test_metrics"]


def _cuda_available() -> bool:
    """Whether a CUDA device can be used, without making PyTorch a requirement.

    XGBoost trains on CUDA without torch; torch is only the probe here. Importing it
    unconditionally crashed pooled training with ModuleNotFoundError on any machine
    without torch - CI's core install, any XGBoost-only environment - before the
    "no GPU, everything on CPU" branch below could run. No torch means no probe, which
    is the same answer as no GPU.

    `POOLED_FORCE_CPU=1` overrides this to False regardless of the real answer. Added
    2026-09-18: a real 3-year pool's parent process was observed at 37.8 GB private
    bytes with a near-zero working set (Get-Process, live, mid-run) - the exact
    "committed but not resident" signature of severe Windows memory pressure - right
    before the year-events worker failed on the same ~584 MiB allocation for the fourth
    time running, on two different (fresh-process, non-fragmented-by-definition)
    attempts. GPU/CUDA paths have been the recurring source of trouble all session (the
    CNN's own cuBLAS non-determinism, earlier). This is a genuinely different
    configuration to test that hypothesis, not another retry of the identical one that
    already failed four times - not a confirmed diagnosis.
    """
    import os
    if os.environ.get("POOLED_FORCE_CPU") == "1":
        return False
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


# Forecast dates per batch when a finished run's validation events are rebuilt for SHAP.
# Smaller than EVENTS_BATCH_CYCLES because this can run beside a training job.
FINALIZE_BATCH_CYCLES = 20


def finalize_for_serving(run_id: str, cache_dir: Path,
                         max_cycles_per_batch: int = FINALIZE_BATCH_CYCLES) -> dict:
    """Make a finished pooled run servable: write `shap_summary.parquet` and the
    manifest's `shap_method`/`paired_rows`, the way `train_pipeline.full_retrain` does.

    The region panel's "what drove this prediction" reads the classifier's rows of that
    summary. Without it a pooled run would serve no explanation at all, silently.

    Rebuilds the validation events from the cached years, one batch of forecast dates
    at a time, using the run's own saved regressors and thresholds. It refuses
    (ValueError) unless those events reproduce the run's saved validation ROC-AUC, so the
    explanation is provably of the events the model was scored on and not a near miss.
    Regressors are explained on up to SHAP_ROWS_PER_GROUP validation rows per (region,
    lead) and the classifier on every validation event, as in full_retrain."""
    import json as _json

    import pyarrow.parquet as pq

    from app.ml import explain as explain_mod
    from app.ml import registry
    from app.ml.train_pipeline import _CLF_CATEGORICAL, _REG_CATEGORICAL

    rd = registry.run_dir(run_id)
    manifest = _json.loads((rd / "manifest.json").read_text())
    train_years = list(manifest["pooled_train_years"])
    test_year = int(manifest["test_year"])
    cached = {y: Path(cache_dir) / f"paired_{y}.parquet" for y in train_years + [test_year]}
    missing = [str(p) for p in cached.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"cached years this run was trained on are missing: {missing}")

    regs = registry.load_regressors(run_id)
    clf, clf_cols = registry.load_classifier(run_id)
    thr = registry.load_thresholds(run_id)
    hbf = registry.load_historical_bust_freq(run_id)
    saved = registry.load_metrics(run_id) or {}
    saved_auc = (((saved.get("classifier") or {}).get("val")) or {}).get("roc_auc")
    if clf is None or thr is None or not regs or saved_auc is None:
        raise ValueError(f"{run_id} lacks a classifier, thresholds, regressors or a saved "
                         f"validation ROC-AUC - nothing to finalize against")

    _, val_c, _ = pooled_split(cached, test_year)
    needed = set(_feature_columns_for(cached, train_years)) | set(fe.EVENT_KEYS) | {
        "variable", "forecast_value", "observed_value", "ensemble_spread",
        "abs_error", "region_id", "season"}
    needed.discard("historical_bust_frequency_region_season")
    read_cols = sorted(needed)

    # Which file holds which cycle is read from the files, not inferred from the year in
    # the name - the same rule year_event_frame follows.
    want = {pd.Timestamp(c) for c in val_c}
    events, samples = [], {v: [] for v in regs}
    for year in train_years:
        present = pd.to_datetime(pq.read_table(cached[year], columns=["init_date"])
                                 .column("init_date").unique().to_pandas())
        mine = want & set(present)
        for batch in _cycle_batches(mine, max_cycles_per_batch):
            df = pd.read_parquet(cached[year], columns=read_cols,
                                 filters=[("init_date", "in", list(batch))])
            if df.empty:
                continue
            df = attach_hbf_column(df.reset_index(drop=True), hbf)
            pred = pd.Series(np.nan, index=df.index, dtype=float)
            for var, (model, cols) in regs.items():
                vmask = df["variable"] == var
                if not vmask.any():
                    continue
                sub = df.loc[vmask]
                pred.loc[vmask] = model.predict(reg_mod._prep_X(sub, cols))
                samples[var].append(explain_mod._stratified_sample(
                    sub, ("region_id", "lead_time_days"), explain_mod.SHAP_ROWS_PER_GROUP))
            events.append(_events_float32(pv.build_event_frame(
                df, pred, thr.p90_error, thr.bust_threshold, hbf, copy_input=False)))
            del df, pred
            gc.collect()
    event_va = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    del events

    art = clf_mod.ClassifierArtifact(model=clf, feature_columns=clf_cols, metrics={},
                                     n_train=0, n_val=len(event_va),
                                     train_bust_rate=float("nan"))
    got = clf_mod._evaluate(event_va["y_bust"], clf_mod.predict_bust_probability(art, event_va))
    if abs(got["roc_auc"] - float(saved_auc)) > 1e-6:
        raise ValueError(f"{run_id}: rebuilt validation events score validation ROC-AUC "
                         f"{got['roc_auc']:.6f}, but the run recorded {float(saved_auc):.6f} - "
                         f"refusing to explain events that are not the ones it was scored on")

    frames = []
    for var, parts in samples.items():
        sub = pd.concat(parts) if parts else pd.DataFrame()
        if len(sub) >= 20:
            model, cols = regs[var]
            frames.append(explain_mod.explain_model(
                model, sub, cols, _REG_CATEGORICAL, model_name=f"regressor::{var}",
                max_rows_per_group=explain_mod.SHAP_ROWS_PER_GROUP))
    if len(event_va) >= 20:
        frames.append(explain_mod.explain_model(clf, event_va, clf_cols, _CLF_CATEGORICAL,
                                                model_name="classifier"))
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise ValueError(f"{run_id}: nothing could be explained")
    shap_summary = pd.concat(frames, ignore_index=True)
    tmp = rd / "shap_summary.parquet.tmp"
    shap_summary.to_parquet(tmp, index=False)
    tmp.replace(rd / "shap_summary.parquet")

    shap_methods = shap_summary.groupby("model")["method"].first().to_dict()
    manifest.update({
        "shap_method": shap_methods.get("classifier", next(iter(shap_methods.values()))),
        "shap_methods": shap_methods,
        "shap_regressor_rows_per_group": explain_mod.SHAP_ROWS_PER_GROUP,
        # Rows the pool held across its training years, read from the cache footers.
        "paired_rows": int(sum(pq.ParquetFile(cached[y]).metadata.num_rows
                               for y in train_years)),
        "finalized_validation_roc_auc": got["roc_auc"],
    })
    registry.save_manifest(run_id, manifest)
    return {"run_id": run_id, "validation_events": len(event_va),
            "validation_roc_auc": got["roc_auc"], "shap_method": manifest["shap_method"]}


FIT_MODES = ("sample", "staged")


def full_retrain_pooled(train_years: list, test_year: int, cache_dir: Path,
                        run_id: str | None = None, fit_mode: str = "sample") -> "TrainReport":
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
    for stale_year in _drop_stale_caches(cache_dir, all_years):
        print(f"pooled cache for {stale_year} was narrower than its peers - rebuilding",
              flush=True)
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
    va_spill = cache_dir / "_va_by_variable"
    va_built = _run_val_frame_subprocess(cached, val_years, val_c, needed, hbf, va_spill)
    if va_built.get("error"):
        report.status = "failed"
        report.error = f"validation frame build failed: {va_built['error'][-2000:]}"
        return report
    n_va = int(va_built["n_rows"])

    variables = sorted(pd.read_parquet(cached[train_years[0]], columns=["variable"])
                       ["variable"].unique())
    fold_of = assign_folds(train_c)
    # "sample": each regressor fits on MAX_FIT_CYCLES of the training cycles.
    # "staged": on every training cycle, boosted chunk by chunk (fit_chunks), each chunk
    # no bigger than the sample - same peak memory, the whole pool seen.
    if fit_mode not in FIT_MODES:
        raise ValueError(f"fit_mode must be one of {FIT_MODES}, got {fit_mode!r}")
    if fit_mode == "staged":
        fit_c = fit_chunks(train_c)
        n_fit = sum(len(c) for c in fit_c)
        report.split_cycles["fit_chunks"] = len(fit_c)
    else:
        fit_c = fit_cycles(train_c)
        n_fit = len(fit_c)
    report.split_cycles["fit"] = n_fit
    print(f"[pooled] regressors fit ({fit_mode}) on {n_fit} of {len(train_c)} training cycles"
          + (f" in {len(fit_c)} chunks" if fit_mode == "staged" else ""),
          file=sys.stderr, flush=True)

    # Split the variables across GPU and CPU devices, both running against the same `va`
    # and `hbf`. Each half returns its own partial results; the halves touch disjoint
    # variables and disjoint rows of `va` (via vmask), so merging afterward rather than
    # writing into shared dicts/Series avoids relying on pandas' assignment being
    # thread-safe - kept even though the groups no longer run concurrently (below),
    # because it is still the simplest way to merge two independently-produced results.
    def _train_group(var_subset: list, device: str):
        # Each variable trains in its own subprocess (see _run_variable_subprocess) so
        # fragmentation from one variable's fits can never carry over into the next -
        # the previous in-process loop here is exactly what crashed repeatedly on
        # 2026-09-14 after 1-3 hours, always at a later point as smaller fixes landed,
        # never fixed by them because the real cause was cross-variable accumulation.
        g_artifacts, g_skipped, g_fold_models = {}, {}, {}
        g_val_pred = pd.Series(np.nan, index=pd.RangeIndex(n_va), dtype=float)
        for var in var_subset:
            ckpt = _variable_checkpoint_path(cache_dir, var, train_years, fit_c, val_c)
            result = _load_variable_checkpoint(ckpt)
            if result is not None:
                print(f"[pooled] {var} reused from checkpoint {ckpt.name}",
                      file=sys.stderr, flush=True)
            else:
                va_var = _read_va_var(va_spill, var)
                result = _run_variable_subprocess(
                    cached, train_years, var, fit_c, va_var, hbf, cache_dir, device, fold_of)
                del va_var
                if result.get("artifact") is not None:
                    _save_variable_checkpoint(ckpt, result)
            art = result["artifact"]
            if art is None:
                g_skipped[var] = result.get("skipped") or "regressor training returned None or too few rows"
                continue
            g_artifacts[var] = art
            if result["val_pred"] is not None:
                g_val_pred.loc[result["val_pred"].index] = result["val_pred"]
            g_fold_models[var] = result["fold_models"]
        return g_artifacts, g_skipped, g_val_pred, g_fold_models

    # Every variable on one device - CUDA when there is one, with _run_variable_subprocess
    # falling back to CPU per variable after two CUDA failures. Real failure 2026-09-21,
    # 17-year pool: the four variables trained on CUDA finished; every variable given to
    # the CPU half drove its worker to 60-62 GB of virtual memory and into Windows' commit
    # limit (System event 2004 at 16:28, 17:37 and 18:43 UTC). The halves only existed to
    # run concurrently, and that was removed on 2026-09-17 (below).
    device = "cuda" if _cuda_available() else "cpu"

    artifacts: dict = {}
    val_pred = pd.Series(np.nan, index=pd.RangeIndex(n_va), dtype=float)
    fold_models: dict = {}
    # Sequential, deliberately - NOT a ThreadPoolExecutor running both groups at once
    # anymore. Real crash 2026-09-17: each variable's worker is its own OS subprocess
    # (_run_variable_subprocess) reading a full year's frame - several GB depending on
    # the variable - and running two of those at once (one GPU-thread variable, one
    # CPU-thread variable) alongside whatever the parent still holds (`va`, `hbf`) pushed
    # this ~24 GB machine over the edge repeatedly: the SAME ~4.29 GB malloc failed on
    # different variables across different runs (temperature_c, wind_direction_deg,
    # rainfall_mm), consistent with "whichever pair happened to be concurrent at the peak
    # moment" rather than one variable being the problem. Concurrency here bought GPU/CPU
    # overlap; it cost reliably fitting in memory. Given this project's own rule (a model
    # that gets a number is worse than no number if it silently dropped 5 of 8 variables
    # getting there), correctness wins over the wall-clock saving.
    results = [_train_group(variables, device)]

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

    # The classifier trains on events from the same bounded cycle sample the regressors
    # fit on (in staged mode too). Real crash 2026-09-22: every training cycle's events
    # at seventeen years (~39 M rows, float64) left the parent unable to allocate 306 MB.
    # Out-of-fold predictions stay honest either way - a cycle's own fold model never
    # trained on it.
    clf_c = fit_cycles(train_c)
    report.split_cycles["classifier"] = len(clf_c)
    event_tr = build_pooled_train_events(
        cached, train_years, clf_c, hbf, p90_error, bust_threshold, fold_models, fold_of,
        columns=needed)
    del fold_models  # only needed for event_tr's out-of-fold predictions, above
    gc.collect()
    # Built in a worker, batched by forecast date - see _run_val_events_subprocess.
    event_va = (_run_val_events_subprocess(va_spill, val_pred.to_numpy(), hbf, p90_error,
                                           bust_threshold)
                if n_va else pd.DataFrame())
    shutil.rmtree(va_spill, ignore_errors=True)
    gc.collect()

    # Now safe to build: event_tr is built, fold_models is gone, and only the small
    # saved regressor artifacts (not the pooled training data) are needed to score the
    # held-out test year. Runs in its own subprocess - see _run_test_events_subprocess -
    # since the held-out year is read in full and carries the same row-count risk that
    # crashed the (now-fixed) training-year path.
    event_te, test_metrics = _run_test_events_subprocess(
        cached[test_year], test_c, hbf, p90_error, bust_threshold, artifacts, needed)
    for var, m in test_metrics.items():
        if var in artifacts:
            artifacts[var].metrics["test"] = m

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

    # Servable, not just saved - see finalize_for_serving. In its own process: it reads the
    # validation year again, and this parent has held a seventeen-year job's worth of arena.
    # The model directory is passed, not inherited: the worker must read the run this
    # process just wrote, whatever its own environment would resolve MODEL_DIR to.
    fin = _run_worker_subprocess(_FINALIZE_WORKER_SCRIPT,
                                 {"run_id": rid, "cache_dir": str(cache_dir),
                                  "model_dir": str(registry.MODEL_DIR)})
    if fin.get("error"):
        report.status = "failed"
        report.error = (f"models saved as {rid}, but finalize_for_serving failed: "
                        f"{str(fin['error'])[-2000:]}")
        return report
    print(f"[pooled] finalized {rid} for serving: {fin['summary']}", file=sys.stderr, flush=True)

    report.status = "success"
    report.made_current = False
    report.promotion_note = ("not evaluated: full_retrain_pooled never promotes - "
                             "this is a comparison tool")
    report.seconds = (pd.Timestamp.now() - t0).total_seconds()
    return report
