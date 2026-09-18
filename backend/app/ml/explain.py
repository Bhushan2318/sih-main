from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    import shap  # type: ignore
    _SHAP_OK = True
except Exception:  # noqa: BLE001
    _SHAP_OK = False

# Rows explained per (region_id, lead_time_days) for the per-variable regressors. Measured
# 2026-09-11 on the 2017 district models: TreeSHAP runs ~2,830 rows/s on a 300-tree
# regressor and ~13,100 rows/s on the classifier. Explaining the full validation frame cost
# ~86 min for the eight regressors (14.5 M rows) and ~0.5 min for the classifier (366 k
# events). 25 rows per group takes the regressors to ~8 min. The classifier - the model
# the region panel serves - is explained in full.
SHAP_ROWS_PER_GROUP = 25


def _stratified_sample(df: pd.DataFrame, group_cols, n: int | None, seed: int = 0) -> pd.DataFrame:
    """At most `n` rows per group, reproducibly. None keeps every row."""
    gcols = [c for c in group_cols if c in df.columns]
    if n is None or not gcols or df.empty:
        return df
    order = np.random.default_rng(seed).permutation(len(df))
    shuffled = df.iloc[order]
    keep = shuffled.groupby(gcols, observed=True, sort=False).cumcount().to_numpy() < n
    return shuffled[keep].sort_index()


def _prep(df: pd.DataFrame, cols: list, categorical: list) -> pd.DataFrame:
    X = df[cols].copy()
    for c in cols:
        if c in categorical:
            X[c] = X[c].astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _shap_values(model, X: pd.DataFrame) -> np.ndarray | None:
    if not _SHAP_OK:
        return None
    try:
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X, check_additivity=False)
        if isinstance(vals, list):
            vals = vals[-1]
        return np.asarray(vals)
    except Exception as exc:  # noqa: BLE001
        # Still a fallback, no longer a silent one: the served explanation changes kind here.
        log.error("SHAP failed (%s: %s); falling back to feature importance",
                  type(exc).__name__, str(exc)[:200])
        return None


def explain_model(
    model,
    val_df: pd.DataFrame,
    feature_columns: list,
    categorical: list,
    model_name: str,
    group_cols=("region_id", "lead_time_days"),
    max_rows_per_group: int | None = None,
) -> pd.DataFrame:
    if val_df.empty:
        return pd.DataFrame()
    val_df = _stratified_sample(val_df, group_cols, max_rows_per_group)
    X = _prep(val_df, feature_columns, categorical)
    sv = _shap_values(model, X)

    if sv is not None and sv.shape == X.shape:
        method = "shap"
        contrib = np.abs(sv)
    else:
        method = "feature_importance_fallback"
        fi = np.asarray(getattr(model, "feature_importances_", np.zeros(len(feature_columns))), float)
        contrib = np.tile(fi, (len(X), 1))

    contrib_df = pd.DataFrame(contrib, columns=feature_columns, index=val_df.index)
    rows = []

    overall = contrib_df.mean(axis=0)
    for feat, v in overall.items():
        rows.append(dict(model=model_name, group_region_id="__all__",
                         group_lead_time_days=-1, feature=feat,
                         mean_abs_shap=float(v), method=method))

    gcols = [c for c in group_cols if c in val_df.columns]
    if gcols:
        group_keys = [val_df[c].to_numpy() for c in gcols]
        for keys, idx in contrib_df.groupby(group_keys, observed=True).groups.items():
            keys = keys if isinstance(keys, tuple) else (keys,)
            gr = dict(zip(gcols, keys))
            means = contrib_df.loc[idx, feature_columns].mean(axis=0)
            for feat, v in means.items():
                rows.append(dict(
                    model=model_name,
                    group_region_id=str(gr.get("region_id", "__all__")),
                    group_lead_time_days=int(gr.get("lead_time_days", -1))
                    if gr.get("lead_time_days") is not None else -1,
                    feature=feat, mean_abs_shap=float(v), method=method,
                ))
    return pd.DataFrame(rows)


def top_factors_for(shap_summary: pd.DataFrame, region_id: str, lead_time_days: int,
                    model: str | None = None, k: int = 5) -> list:
    df = shap_summary
    if model:
        df = df[df["model"] == model]
    sel = df[(df["group_region_id"] == str(region_id)) &
             (df["group_lead_time_days"] == int(lead_time_days))]
    if sel.empty:
        sel = df[df["group_region_id"] == "__all__"]
    sel = sel.sort_values("mean_abs_shap", ascending=False).head(k)
    return [
        {"feature": r.feature, "importance": round(r.mean_abs_shap, 5), "method": r.method}
        for r in sel.itertuples()
    ]
