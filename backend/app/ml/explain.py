"""SHAP explanations, precomputed at train time.

For every regressor and the classifier we run shap.TreeExplainer once on the validation
split and aggregate mean|SHAP| per feature, and per (region_id, lead_time_days) group.
If SHAP is unavailable or errors, we fall back to XGBoost's feature_importances_ and set
method='feature_importance_fallback' so the UI never implies SHAP ran when it didn't.

Output: one tidy DataFrame written to shap_summary.parquet with columns
  model, group_region_id, group_lead_time_days, feature, mean_abs_shap, method
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # shap pulls in llvmlite/numba; degrade gracefully if the wheel is unhappy
    import shap  # type: ignore
    _SHAP_OK = True
except Exception:  # noqa: BLE001
    _SHAP_OK = False


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
        if isinstance(vals, list):  # older API, classifier -> [neg, pos]
            vals = vals[-1]
        return np.asarray(vals)
    except Exception:  # noqa: BLE001
        return None


def explain_model(
    model,
    val_df: pd.DataFrame,
    feature_columns: list,
    categorical: list,
    model_name: str,
    group_cols=("region_id", "lead_time_days"),
) -> pd.DataFrame:
    if val_df.empty:
        return pd.DataFrame()
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

    # overall
    overall = contrib_df.mean(axis=0)
    for feat, v in overall.items():
        rows.append(dict(model=model_name, group_region_id="__all__",
                         group_lead_time_days=-1, feature=feat,
                         mean_abs_shap=float(v), method=method))

    # per (region_id, lead_time_days) - group by external arrays so feature columns that
    # are themselves named region_id / lead_time_days don't collide
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
    """Read-side helper: top-k features for a region/lead, newest run's summary."""
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
