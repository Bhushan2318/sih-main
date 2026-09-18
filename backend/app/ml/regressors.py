from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

MIN_ROWS = 30

NUMERIC_FEATURES = [
    "lead_time_days", "forecast_value", "month",
    "ensemble_spread", "ensemble_member_count",
    "pressure_rate_of_change", "moisture_rate_of_change",
    "forecast_error_lag", "historical_bust_frequency_region_season",
    "jump_abs_change", "jump_std", "jump_sign_flips", "jump_rel_climatology",
    # C2: time-lagged ensemble (see app.features.engineering.LAF_FEATURES).
    "laf_pool_mean", "laf_pool_std", "laf_pool_size", "laf_spread_ratio",
    # C4: district descriptors, replacing region_id as a raw feature (see
    # app.features.engineering.DISTRICT_DESCRIPTOR_FEATURES) - a district the model never
    # saw labelled in training still has a latitude, a size and a distance from the
    # country's edge, so these generalise where a 666-level categorical cannot.
    "centroid_lat", "centroid_lon", "area_km2", "border_distance_km", "elevation_mean",
    # C3: global daily MJO state (see app.features.engineering.MJO_FEATURES), attached by
    # an as-of join on init_date - identical for every district issued the same day.
    "mjo_rmm1", "mjo_rmm2", "mjo_amplitude",
]
# state_id (~36 levels) replaces region_id (666 levels, only ~34 ever labelled) as the
# district-identity feature - see NUMERIC_FEATURES' C4 comment above.
CATEGORICAL_FEATURES = ["state_id", "season"]
CONCURRENT_PREFIX = "fc_"

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_lambda=2.0,
    reg_alpha=0.5,
    objective="reg:squarederror",
    tree_method="hist",
    enable_categorical=True,
    n_jobs=0,
    random_state=42,
)


@dataclass
class RegressorArtifact:
    variable: str
    model: xgb.XGBRegressor
    feature_columns: list
    metrics: dict
    n_train: int
    n_val: int


def feature_columns(df: pd.DataFrame) -> list:
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cols += [c for c in df.columns if c.startswith(CONCURRENT_PREFIX)]
    cols += [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return cols


def _prep_X(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    X = df[cols].copy()
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for c in X.columns:
        if c not in CATEGORICAL_FEATURES:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 2 else float("nan"),
        "n": int(len(y_true)),
        "baseline_mae_predict_mean": float(np.mean(np.abs(y_true - np.mean(y_true)))),
    }


def train_variable_regressor(
    train_df: pd.DataFrame, val_df: pd.DataFrame, variable: str
) -> RegressorArtifact | None:
    tr = train_df[train_df["variable"] == variable]
    va = val_df[val_df["variable"] == variable]
    if len(tr) < MIN_ROWS:
        return None
    cols = feature_columns(tr)
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(_prep_X(tr, cols), tr["abs_error"].to_numpy())

    metrics = {"train": _evaluate(tr["abs_error"], model.predict(_prep_X(tr, cols)))}
    if len(va) >= 5:
        metrics["val"] = _evaluate(va["abs_error"], model.predict(_prep_X(va, cols)))
    return RegressorArtifact(variable, model, cols, metrics, len(tr), len(va))


def oof_predict(train_df: pd.DataFrame, variable: str, n_splits: int = 3) -> pd.Series:
    """Out-of-fold predictions, grouped by init_date so no forecast cycle spans a fold."""
    tr = train_df[train_df["variable"] == variable].copy()
    if len(tr) < MIN_ROWS:
        return pd.Series(np.nan, index=tr.index)
    cols = feature_columns(tr)
    groups = tr["init_date"].astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups) if n_groups > 1 else 2
    oof = pd.Series(np.nan, index=tr.index, dtype=float)

    if n_groups < 2:
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(_prep_X(tr, cols), tr["abs_error"].to_numpy())
        oof[:] = model.predict(_prep_X(tr, cols))
        return oof

    gkf = GroupKFold(n_splits=splits)
    for tr_idx, te_idx in gkf.split(tr, tr["abs_error"], groups):
        sub_tr, sub_te = tr.iloc[tr_idx], tr.iloc[te_idx]
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(_prep_X(sub_tr, cols), sub_tr["abs_error"].to_numpy())
        oof.iloc[te_idx] = m.predict(_prep_X(sub_te, cols))
    return oof


def predict_variable_error(artifact: RegressorArtifact, df: pd.DataFrame) -> np.ndarray:
    return artifact.model.predict(_prep_X(df, artifact.feature_columns))
