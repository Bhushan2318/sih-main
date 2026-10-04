"""Platt scaling for the bust classifier: probabilities that mean what they say.

The served classifier trained with scale_pos_weight = neg/pos and was never recalibrated;
its lowest reliability bin predicted 0.158 against ~0.07 observed on test. New runs drop the
weighting and fit p' = sigmoid(a * logit(p) + b) on the validation year. With a > 0 it is
monotone, so rankings and ROC-AUC are unchanged; a fit with a <= 0 would reverse them and
is refused (the probabilities pass through as they are).

SHAP attributions stay on the classifier's own margin: calibration rescales that margin by
`a` and shifts it by `b`, so the order and relative size of every contribution is unchanged.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-6


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def fit_platt(proba, y) -> "dict | None":
    """{'method': 'platt', 'a', 'b'} fitted by maximum likelihood, or None when the fit
    is unusable (one class only, or a non-positive slope that would reverse the ranking)."""
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return None
    x = _logit(proba).reshape(-1, 1)
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(x, y)
    a, b = float(m.coef_[0, 0]), float(m.intercept_[0])
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0:
        return None
    return {"method": "platt", "a": a, "b": b}


def apply(calibrator: "dict | None", proba) -> np.ndarray:
    """Calibrated probabilities; unchanged when there is no calibrator."""
    p = np.asarray(proba, dtype=float)
    if not calibrator:
        return p
    if calibrator.get("method") != "platt":
        raise ValueError(f"unknown calibrator {calibrator.get('method')!r}")
    z = calibrator["a"] * _logit(p) + calibrator["b"]
    return 1.0 / (1.0 + np.exp(-z))
