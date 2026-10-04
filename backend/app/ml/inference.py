from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # loaded with the model (registry), not with this module
    import xgboost as xgb

from app.config import settings
from app.features import engineering as fe
from app.features import pivot as pv
from app.features.history import forecast_history
from app.ml import precomputed
from app.ml import registry
from app.ml.thresholds import Thresholds
from app.storage import parquet_store


def run_feature_version(manifest: dict) -> int:
    """The feature version a run was trained with, so it is scored with the same
    arithmetic. A run that predates recorded versions is version 1 - see
    contracts.FEATURE_VERSION."""
    from app import contracts
    return int((manifest or {}).get("feature_version", contracts.LEGACY_FEATURE_VERSION))


@dataclass
class ModelState:

    run_id: str
    regressors: dict
    classifier: xgb.XGBClassifier
    classifier_columns: list
    thresholds: Thresholds
    historical_bust_freq: dict
    # Training-split mean |forecast jump| per (region, variable); empty for a run trained
    # before C1, whose models never saw jump_rel_climatology.
    jump_climatology: dict = field(default_factory=dict)
    shap_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    manifest: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    # Label version 2: the run's training-period bias table (app/features/bias.py), applied
    # to every frame it scores. None for a run trained before it - scored exactly as before.
    bias_table: Optional[pd.DataFrame] = None
    # Set instead of `regressors` on the serving box, which loads the models' names and not
    # the models (see _models_stay_off_this_box).
    regressor_names: list = field(default_factory=list)

    @property
    def variables(self) -> list:
        return sorted(self.regressors) if self.regressors else sorted(self.regressor_names)


@dataclass
class ScoredCycle:

    run_id: str
    init_date: pd.Timestamp
    events: pd.DataFrame
    per_variable: pd.DataFrame
    n_rows_scored: int


class CycleNotPrecomputed(RuntimeError):
    """A cycle with no CI-precomputed answer, requested on a box that must not score it."""

    def __init__(self, init_date):
        self.init_date = pd.Timestamp(init_date).date()
        super().__init__(
            f"Cycle {self.init_date} has no precomputed answer on this server, and scoring "
            "it here would exceed the server's 512 MB memory limit. Choose one of the cycles "
            "listed by /api/replay/cycles."
        )


_lock = threading.Lock()
_state_cache: tuple | None = None
_score_cache: "dict[tuple, ScoredCycle]" = {}
_SCORE_CACHE_MAX = 24


# The label columns of a SHAP summary are a handful of distinct values repeated across
# every row - one per model, region group, feature and method. Held as `object` they are
# two million Python strings; dictionary-encoded they are two million small integers and a
# short lookup. run_20260922T043925Z is 11.2 MB on disk, 2,101,395 rows, and 604 MB
# resident once read the obvious way, on a box that is killed at 512 MB.
#
# DO NOT "simplify" this to pd.read_parquet followed by astype("category"). That is WORSE
# than doing nothing at all: it builds every string and then throws them away, so the peak
# - which is what kills a container - goes UP while the steady-state number goes down and
# the diff reads as a tidy-up. Measured 2026-09-23 in separate processes on two machines:
#
#                                     peak        resident
#   pd.read_parquet                +410 / +462 MB   604 MB
#   read_parquet then astype       +461 / +524 MB    42 MB   <- worse at the peak
#   read_table(read_dictionary=)   +319 / +232 MB    42 MB
#
# Two figures because two machines disagree on the magnitude and agree on the direction;
# one number invites someone to re-measure once, get something between them, and conclude
# the comment is wrong. Separate processes are not optional either: measuring both in one
# process gave +374 for the fix against +197 for the bare read - backwards - because the
# second read reused the first's freed pages.
#
# This changes representation only. It deliberately does NOT filter to the classifier's
# rows, though that would be smaller again: top_factors_for takes a `model` argument, so
# a caller may legitimately ask for a regressor's explanation, and a loader that silently
# dropped those rows would answer such a caller with nothing.
#
# The one exception is the serving box, and it is deliberate (2026-09-27): the site only ever
# asks for the classifier's factors (the district panel; pinned by
# tests/test_shap_summary_memory.py), so the box reads the classifier's rows alone, in
# batches. Measured on run_20260922T043925Z (macOS RSS): +247 MB resident for the whole
# summary, +195 MB for a whole-file read filtered afterwards - Arrow keeps the decode buffers
# of each million-row group - and +73 MB for the classifier's 632,795 rows read 64k at a time.
_SHAP_LABEL_COLUMNS = ("model", "group_region_id", "feature", "method")
_SHAP_BATCH_ROWS = 65_536


def _read_shap_summary(path, only_model: Optional[str] = None) -> pd.DataFrame:
    """The run's SHAP summary, in the smallest representation that loses nothing - or, with
    `only_model`, that model's rows alone, read a batch at a time."""
    if not path.exists():
        return pd.DataFrame()
    # read_dictionary, not astype("category") afterwards: converting after the fact
    # materialises every string first and then throws them away, so the PEAK is worse
    # than doing nothing even though the resident frame is smaller - measured +461 MB
    # against the bare read's +408. Asking Arrow for dictionary-encoded columns means the
    # strings are never built. Measured +257 MB peak, 42 MB resident.
    import pyarrow.parquet as pq

    present = [c for c in _SHAP_LABEL_COLUMNS
               if c in pq.ParquetFile(path).schema_arrow.names]
    if only_model is None:
        df = pq.read_table(path, read_dictionary=present).to_pandas()
    else:
        df = _read_one_models_rows(path, present, only_model)
    if "mean_abs_shap" in df.columns:
        df["mean_abs_shap"] = df["mean_abs_shap"].astype("float32")
    if "group_lead_time_days" in df.columns:
        df["group_lead_time_days"] = df["group_lead_time_days"].astype("int16")
    return df


def _read_one_models_rows(path, dictionary_columns: list, model: str) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    f = pq.ParquetFile(path, read_dictionary=dictionary_columns)
    parts = []
    for batch in f.iter_batches(batch_size=_SHAP_BATCH_ROWS):
        m = batch.column("model")
        if pa.types.is_dictionary(m.type):
            m = m.dictionary_decode()
        parts.append(batch.filter(pc.equal(m, model)))
    if not parts:
        return pd.DataFrame(columns=f.schema_arrow.names)
    return pa.Table.from_batches(parts, schema=parts[0].schema).to_pandas()


def _models_stay_off_this_box() -> bool:
    """True on the serving box, which loads the models' names and not the models.

    The box never scores - it refuses to (settings.serving_read_only, score_cycle) and CI
    scores for it - so the XGBoost models would sit unused: measured +73 MB with xgboost for
    the eight regressors (macOS RSS, 2026-09-27). With them go their SHAP rows. On a
    fresh box-mode server building the opening screen and Replay at once, peak RSS went from
    518-527 MB to 363-386 MB, every one of the 689 screens identical (tests/test_api.py).
    Only reached when the box builds live - a deploy without its prebuilt bundle.
    """
    return settings.serving_read_only


def load_model_state(run_id: Optional[str] = None) -> Optional[ModelState]:
    global _state_cache
    rid = run_id or registry.current_run_id()
    if rid is None:
        return None

    # Keyed on the mode as well: a state loaded without its models must never be handed to
    # a caller that scores. A full state is a superset, so it serves the box as it is.
    lean = _models_stay_off_this_box()
    with _lock:
        if _state_cache and _state_cache[0] == rid and (lean or not _state_cache[1]):
            return _state_cache[2]

    thr = registry.load_thresholds(rid)
    if lean:
        names = registry.regressor_names(rid)
        has_clf = (registry.run_dir(rid) / "classifier.json").exists()
        if not names or not has_clf or thr is None:
            return None
        regressors, clf, clf_cols, names_only = {}, None, [], names
        freq, jump = {}, {}
        shap_summary = _read_shap_summary(registry.run_dir(rid) / "shap_summary.parquet",
                                          only_model="classifier")
    else:
        regressors = registry.load_regressors(rid)
        clf, clf_cols = registry.load_classifier(rid)
        if not regressors or clf is None or thr is None:
            return None
        names_only = []
        freq = registry.load_historical_bust_freq(rid)
        jump = registry.load_jump_climatology(rid)
        shap_summary = _read_shap_summary(registry.run_dir(rid) / "shap_summary.parquet")
        bias_table = registry.load_bias_table(rid)

    import json
    def _read(name):
        p = registry.run_dir(rid) / name
        return json.loads(p.read_text()) if p.exists() else {}

    state = ModelState(
        run_id=rid,
        regressors=regressors,
        classifier=clf,
        classifier_columns=clf_cols,
        thresholds=thr,
        historical_bust_freq=freq,
        jump_climatology=jump,
        shap_summary=shap_summary,
        manifest=_read("manifest.json"),
        metrics=_read("metrics.json"),
        regressor_names=names_only,
        bias_table=None if lean else bias_table,
    )
    with _lock:
        _state_cache = (rid, lean, state)
    return state


def invalidate_caches() -> None:
    global _state_cache
    with _lock:
        _state_cache = None
        _score_cache.clear()
    try:
        from app.services import replay_service
        replay_service.invalidate()
        from app.services import ensemble_service
        ensemble_service.invalidate()
        from app.services import response_cache
        response_cache.invalidate()
    except Exception:  # noqa: BLE001 - never let cache cleanup break an ingest
        pass


def score_latest_cycle(state: Optional[ModelState] = None) -> Optional[ScoredCycle]:
    return score_cycle(state, init_date=None)


def available_cycles() -> list[pd.Timestamp]:
    """Every scoreable forecast cycle, newest first.

    Read from Parquet footers rather than by scanning a column of every forecast row. At
    district resolution that scan was 19.4 million values to learn 73 dates - measured at
    +253 MB against a 512 MB box that is killed rather than throttled - where the footers
    answer the same question for +2 MB. See parquet_store.distinct_forecast_init_dates.
    """
    dates = parquet_store.distinct_forecast_init_dates()
    return sorted((pd.Timestamp(d) for d in dates), reverse=True)


_MAX_LEAD_DAYS = 10
_OBS_PAD_DAYS = 3

_SCORING_COLUMNS = [
    "region_id", "variable", "valid_date", "value", "value_type",
    "init_date", "lead_time_days", "ensemble_member_id", "verification_status",
]


def build_scoring_frame(
    state: ModelState, target_init: pd.Timestamp, *, as_of_init: bool = False
) -> pd.DataFrame:
    """Every feature row of one cycle, as the models are about to see it.

    `as_of_init=True` asks what the model would have said at 00 UTC on `target_init`.
    `forecast_error_lag` is the only input built from observations - the realised error at
    the previous lead of the same forecast, i.e. data from after the forecast was issued
    (engineering.build_training_frame) - so it is blanked, which is the state every live
    cycle is scored in. The observed values stay: they are what Replay checks the forecast
    against, not something the model is told.

    This recreates issue-time conditions for a replay. It is not a fix for the leak: the
    model still learned from the lag in training (docs/known-issues.md).
    """
    fc_rows = parquet_store.read_dataset(
        value_types=["forecast"], init_dates=[target_init.date()], columns=_SCORING_COLUMNS,
    )
    fc_rows = fe.drop_beyond_archive_leads(fc_rows)
    if fc_rows.empty:
        return pd.DataFrame()
    observed = parquet_store.read_dataset(
        value_types=["observed"],
        columns=_SCORING_COLUMNS,
        valid_date_min=(target_init - pd.Timedelta(days=_OBS_PAD_DAYS)).date(),
        valid_date_max=(target_init + pd.Timedelta(days=_MAX_LEAD_DAYS + _OBS_PAD_DAYS)).date(),
    )

    subset = pd.concat([fc_rows, observed], ignore_index=True)
    del fc_rows, observed
    # Earlier cycles for the jumpiness features, read one at a time and reduced to ensemble
    # means as they are read. Cycle dates come from Parquet footers, never a column scan.
    history = forecast_history(target_init, parquet_store.distinct_forecast_init_dates())

    frame = fe.build_training_frame(
        subset,
        historical_bust_freq=state.historical_bust_freq or None,
        require_observed=False,
        forecast_history=history,
        jump_climatology=state.jump_climatology or None,
        feature_version=run_feature_version(getattr(state, "manifest", None)),
    )
    del subset, history
    bias_table = getattr(state, "bias_table", None)
    if bias_table is not None:
        from app.features.bias import apply_bias
        frame = apply_bias(frame, bias_table)
    if as_of_init and "forecast_error_lag" in frame.columns:
        frame["forecast_error_lag"] = np.nan
    return frame


def score_cycle(
    state: Optional[ModelState] = None,
    init_date: "Optional[pd.Timestamp | str]" = None,
    as_of_init: bool = False,
) -> Optional[ScoredCycle]:
    state = state or load_model_state()
    if state is None:
        return None

    fp = parquet_store.store_fingerprint()
    if init_date is None:
        latest = parquet_store.latest_forecast_init_date()
        if latest is None:
            return None
        target_init = pd.Timestamp(latest).normalize()
    else:
        target_init = pd.Timestamp(init_date).normalize()

    cache_key = (state.run_id, str(target_init), fp, as_of_init)
    with _lock:
        hit = _score_cache.get(cache_key)
        if hit is not None:
            return hit

    # Scored in CI, read here. At 666 districts scoring a cycle peaks at 1,406 MB against a
    # box killed at 512; reading the answer is 38 MB. Checked before any store read, because
    # the point is not to touch the forecast rows at all. A cycle with no artifact - Replay
    # asks for arbitrary historical ones - falls through and scores as before. See
    # app/ml/precomputed.py. Never for an as-of-init score: those artifacts were scored with
    # forecast_error_lag filled in, which is the very thing as_of_init leaves out.
    if not as_of_init:
        ready = precomputed.read_scored_cycle(state.run_id, target_init)
        if ready is not None:
            with _lock:
                _score_cache[cache_key] = ready
            return ready
    # On the serving box the fall-through below is the 1,406 MB path, and Replay and the
    # ensemble endpoint accept any date. Refuse, rather than score and be killed.
    if settings.serving_read_only:
        raise CycleNotPrecomputed(target_init)

    frame = build_scoring_frame(state, target_init, as_of_init=as_of_init)
    if frame.empty:
        return None

    # A state the serving box loaded - its models' names but not the models - would predict
    # nothing and return None: an empty map that reads as "no cycle to score". Say what
    # actually happened instead.
    if getattr(state, "regressor_names", None) and not getattr(state, "regressors", None):
        raise RuntimeError(
            f"model state {state.run_id} was loaded without its models (the serving box's "
            "load) and cannot score; load it off the box")
    pred = pd.Series(np.nan, index=frame.index, dtype=float)
    for var, (model, cols) in state.regressors.items():
        mask = frame["variable"] == var
        if not mask.any():
            continue
        X = _prep(frame.loc[mask], cols, categorical_features(model))
        pred.loc[mask] = model.predict(X)
    frame["pred_err"] = pred

    scored = frame.loc[frame["pred_err"].notna()].copy()
    if scored.empty:
        return None

    events = pv.build_event_frame(
        scored, scored["pred_err"], state.thresholds.p90_error,
        bust_threshold=None, historical_bust_freq=state.historical_bust_freq or None,
    )
    X_evt = _prep(events, state.classifier_columns,
                  categorical_features(state.classifier))
    events["bust_probability"] = state.classifier.predict_proba(X_evt)[:, 1]
    events["risk_band"] = [state.thresholds.band_for(p) for p in events["bust_probability"]]

    events["dominant_variable"] = _dominant_variable(events, state.thresholds.bust_threshold)

    per_variable = _per_variable_table(scored, state)

    result = ScoredCycle(
        run_id=state.run_id,
        init_date=target_init,
        events=events,
        per_variable=per_variable,
        n_rows_scored=len(scored),
    )
    with _lock:
        if len(_score_cache) >= _SCORE_CACHE_MAX:
            _score_cache.pop(next(iter(_score_cache)))
        _score_cache[cache_key] = result
    return result


_FALLBACK_CATEGORICAL = {"state_id", "season"}


def categorical_features(model) -> set:
    """Which of a model's features are categorical, according to that model.

    Read from the booster's own `feature_types` rather than from a constant in this
    module. The constant describes what the *current* code trains with; a loaded model
    may have been trained under an earlier feature contract, and it is the model's
    contract that has to be honoured at scoring time.

    This is not hypothetical. C4 swapped the district-identity feature from `region_id`
    to `state_id`; against a model trained before that, a hardcoded `{"state_id",
    "season"}` sent `region_id` down the numeric-coercion branch and silently destroyed
    it - see tests/test_inference_feature_contract.py for what that did to the scores.

    Falls back to the current contract only when a model carries no type information,
    which is the best that can be done for one that never recorded it.
    """
    booster = getattr(model, "get_booster", None)
    if booster is None:
        return set(_FALLBACK_CATEGORICAL)
    try:
        b = booster()
        names, types = b.feature_names, b.feature_types
    except Exception:  # noqa: BLE001 - a model that cannot describe itself gets the default
        return set(_FALLBACK_CATEGORICAL)
    if not names or not types or len(names) != len(types):
        return set(_FALLBACK_CATEGORICAL)
    return {n for n, t in zip(names, types) if t == "c"}


def _prep(df: pd.DataFrame, cols: list, categorical: "set | None" = None) -> pd.DataFrame:
    categorical = set(_FALLBACK_CATEGORICAL if categorical is None else categorical)
    X = pd.DataFrame(index=df.index)
    for c in cols:
        if c in df.columns:
            X[c] = df[c]
        else:
            X[c] = np.nan

    # Which columns arrived carrying real data. Anything that is all-null on the way in
    # is missing data, which is legitimate - wind stops at day 5 and soil moisture at
    # day 3 in the reforecast archive - and must not be confused with damage done below.
    had_data = {c for c in cols if c in df.columns and df[c].notna().any()}

    destroyed = []
    for c in cols:
        if c in categorical:
            X[c] = X[c].astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        if c in had_data and X[c].isna().all():
            destroyed.append(c)

    if destroyed:
        # Refuse rather than patch (CONTRIBUTING.md rule 3). These columns held real values and
        # are now entirely NaN, which only happens when a non-numeric column was coerced -
        # i.e. the model expects it as categorical and we did not treat it as one. Scoring
        # on it would not fail, it would quietly produce confident nonsense, so the only
        # safe move is to stop and say which feature and why.
        raise ValueError(
            f"feature contract mismatch: {', '.join(sorted(destroyed))} "
            f"had data but became entirely NaN after type coercion. The model expects "
            f"{'these' if len(destroyed) > 1 else 'this'} as categorical but "
            f"{'they were' if len(destroyed) > 1 else 'it was'} not in the categorical "
            f"set {sorted(categorical)}. This usually means the model artifact predates "
            f"a change to the feature pipeline - retrain, or score with the model whose "
            f"contract matches this code."
        )
    return X


def _dominant_variable(events: pd.DataFrame, bust_threshold: dict) -> list:
    ratio_cols = {}
    for var, thr in bust_threshold.items():
        col = f"pred_err_{var}"
        if col in events.columns and thr:
            ratio_cols[var] = events[col] / thr
    if not ratio_cols:
        return [None] * len(events)
    ratios = pd.DataFrame(ratio_cols, index=events.index)
    if bool(ratios.isna().to_numpy().all()):
        return [None] * len(events)
    return ratios.idxmax(axis=1, skipna=True).where(ratios.notna().any(axis=1)).tolist()


def _is_retired_city_point_cycle(init: pd.Timestamp) -> bool:
    """Was this cycle ingested by the live feed while it sampled 36 city points?

    Until 07d9835 the live feed took the nearest grid point to each of 36 cities, and the
    rows found their district by name or by the point's polygon. It has written district
    area means, keyed by district id, since - and the observations in the store for those
    earlier days were re-pulled as district area means too. Pairing one of those cycles
    anew would verify a point forecast against a polygon mean.

    Told apart by the feed's own files and how their rows were placed. The method alone
    does not say point or area: the reforecast district store's area means found their
    districts by point_in_polygon too.
    """
    # Undeduplicated: only which files and methods occur matters, not which row wins.
    fc = parquet_store.read_dataset(value_types=["forecast"], init_dates=[init.date()],
                                    columns=["source_file", "region_resolution_method"],
                                    dedupe=False)
    if fc.empty:
        return False
    live = fc["source_file"].astype(str).str.startswith("gefs_operational_forecast")
    return bool((live & (fc["region_resolution_method"].astype(str) != "region_id")).any())


def refresh_observations(scored: ScoredCycle) -> ScoredCycle:
    """`scored` with its observed values re-read from the store, every prediction as issued.

    A live cycle is scored in the CI run that ingests it, before any of its days have
    happened, and score_cycle returns an existing artifact as it stands - so every later
    run packaged the first answer again and a cycle never verified. This is not a re-score:
    that would fill forecast_error_lag from the very observations the cycle is being
    checked against, and change after the fact what the model is shown to have said.

    Observations are read over the window build_scoring_frame reads and paired by the same
    rule (engineering.observed_means), so they are exactly what a fresh score would pair; a
    cell still unobserved is NaN, never zero. A cycle from the retired city-point feed is
    returned as it was (_is_retired_city_point_cycle).
    """
    init = pd.Timestamp(scored.init_date).normalize()
    if _is_retired_city_point_cycle(init):
        return scored
    observed = parquet_store.read_dataset(
        value_types=["observed"],
        columns=_SCORING_COLUMNS,
        valid_date_min=(init - pd.Timedelta(days=_OBS_PAD_DAYS)).date(),
        valid_date_max=(init + pd.Timedelta(days=_MAX_LEAD_DAYS + _OBS_PAD_DAYS)).date(),
    )
    ob = fe.observed_means(observed[observed["region_id"].notna()])
    keys = ["region_id", "valid_date", "variable"]

    def _keyed(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"region_id": df["region_id"].astype(str),
                             "valid_date": pd.to_datetime(df["valid_date"]),
                             "variable": df["variable"].astype(str)})

    ob = pd.concat([_keyed(ob), ob["observed_value"]], axis=1)
    pv = scored.per_variable.copy()
    paired = _keyed(pv).merge(ob, on=keys, how="left", validate="many_to_one")
    pv["observed_value"] = pd.to_numeric(paired["observed_value"], errors="coerce").to_numpy()

    # The event frame's actual_err_<var> is |forecast mean - observed| (pivot.build_event_frame),
    # and the forecast mean is per_variable's predicted_value; restated from the refreshed
    # observations so the two tables describe the same ones. No classifier input reads it.
    fc = pv["predicted_value"].to_numpy(dtype=float)
    if "bias_correction" in pv.columns:
        # Label version 2: the label is on the corrected forecast, so the restated error is
        # too. A label variable whose cell had no bias has no corrected error (NaN).
        from app.features.bias import label_variable_mask
        label = label_variable_mask(pv["variable"])
        fc = np.where(label, fc - pv["bias_correction"].to_numpy(dtype=float), fc)
    err = pv.assign(region_id=pv["region_id"].astype(str), variable=pv["variable"].astype(str),
                    actual_err=np.abs(fc - pv["observed_value"].to_numpy(dtype=float)))
    err = err.pivot(index=["region_id", "lead_time_days"], columns="variable",
                    values="actual_err")
    ev = scored.events.drop(columns=[c for c in scored.events.columns
                                     if c.startswith("actual_err_")])
    at = err.reindex(pd.MultiIndex.from_arrays(
        [ev["region_id"].astype(str), ev["lead_time_days"].astype(int)]))
    for var in at.columns:
        ev[f"actual_err_{var}"] = at[var].to_numpy()

    return ScoredCycle(run_id=scored.run_id, init_date=scored.init_date, events=ev,
                       per_variable=pv, n_rows_scored=scored.n_rows_scored)


def _per_variable_table(scored: pd.DataFrame, state: ModelState) -> pd.DataFrame:
    # predicted_value is GEFS's own forecast mean - what the site shows - even for a run
    # that reads the bias-corrected forecast (label version 2); the bias removed is kept
    # beside it so a restated error can be the corrected one (refresh_observations).
    shown = "forecast_value_raw" if "forecast_value_raw" in scored.columns else "forecast_value"
    extra = ({"bias_correction": ("bias_correction", "mean")}
             if "bias_correction" in scored.columns else {})
    g = (
        scored.groupby(["region_id", "lead_time_days", "variable", "valid_date"],
                       observed=True)
        .agg(
            predicted_value=(shown, "mean"),
            observed_value=("observed_value", "mean"),
            predicted_error=("pred_err", "mean"),
            ensemble_spread=("ensemble_spread", "mean"),
            ensemble_member_count=("ensemble_member_count", "max"),
            **extra,
        )
        .reset_index()
    )
    p90 = g["variable"].map(state.thresholds.p90_error)
    g["confidence"] = g["predicted_error"].div(p90).rsub(1.0).clip(0.0, 1.0)
    g["bust_threshold"] = g["variable"].map(state.thresholds.bust_threshold)
    return g


def model_validation_metrics(state: ModelState) -> dict:
    metrics = state.metrics or {}
    regs = {}
    for var, m in (metrics.get("regressors") or {}).items():
        chosen = m.get("test") or m.get("val") or m.get("train") or {}
        regs[var] = {
            "mae": chosen.get("mae"),
            "rmse": chosen.get("rmse"),
            "r2": chosen.get("r2"),
            "baseline_mae": chosen.get("baseline_mae_predict_mean"),
            "split": "test" if m.get("test") else ("val" if m.get("val") else "train"),
            "n": chosen.get("n"),
        }
    clf = metrics.get("classifier") or {}
    chosen_clf = clf.get("test") or clf.get("val") or clf.get("train") or {}
    return {
        "regressors": regs,
        "classifier": {
            **{k: chosen_clf.get(k) for k in
               ("n", "bust_rate", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier")},
            "split": "test" if clf.get("test") else ("val" if clf.get("val") else "train"),
        },
    }
