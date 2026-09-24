"""Small, shared circular-statistics helpers.

Angles are stored in degrees.  A direction of 0 and 360 is the same bearing, so every
error, mean and spread involving ``wind_direction_deg`` must live here rather than being
reimplemented around the training path.  Keeping these primitives dependency-free also lets
feature engineering, event construction and the held-out threshold path use exactly the
same definitions.
"""
from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from app.ingestion.canonical_schema import CIRCULAR_VARIABLES as _CANONICAL_CIRCULAR

CIRCULAR_VARIABLES = frozenset(
    v.value if isinstance(v, Enum) else str(v) for v in _CANONICAL_CIRCULAR
)
_RESULTANT_EPS = 1e-12


def is_circular(variable) -> bool:
    name = variable.value if isinstance(variable, Enum) else str(variable)
    return name in CIRCULAR_VARIABLES


def normalize_degrees(angle):
    """Canonical [0, 360) bearing, including 360 -> 0 without float wrap residue."""
    out = np.mod(np.asarray(angle, dtype=float), 360.0)
    return np.where(out >= 360.0 - 1e-12, 0.0, out)


def circular_abs_error(forecast, observed):
    """Smallest angular separation in degrees, in ``[0, 180]``.

    NaN/Inf inputs stay non-finite; canonical-value validation is responsible for
    quarantining those before this function is used.
    """
    a = np.asarray(forecast, dtype=float)
    b = np.asarray(observed, dtype=float)
    delta = np.mod(a - b, 360.0)
    return np.minimum(delta, 360.0 - delta)


def circular_mean(values, weights=None):
    """Unit-vector circular mean in degrees.

    Returns NaN when no finite observation exists or the resultant vector is
    effectively zero.  The latter is genuinely undefined (for example, members at
    0 and 180 degrees); reporting an arbitrary atan2(0, 0) bearing would invent signal.
    """
    x = np.asarray(values, dtype=float)
    if weights is None:
        w = np.isfinite(x).astype(float)
    else:
        w = np.asarray(weights, dtype=float)
        w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)
    total = np.sum(w)
    if not np.isfinite(total) or total <= 0:
        return np.nan
    rad = np.radians(x)
    sin_sum = np.nansum(np.sin(rad) * w)
    cos_sum = np.nansum(np.cos(rad) * w)
    resultant = float(np.hypot(sin_sum, cos_sum))
    if not np.isfinite(resultant) or resultant <= _RESULTANT_EPS * max(total, 1.0):
        return np.nan
    return float(normalize_degrees(np.degrees(np.arctan2(sin_sum, cos_sum))))


def circular_standard_deviation(values, center=None) -> float:
    """RMS angular deviation from a circular mean, in degrees.

    This is the circular ensemble-spread counterpart to the ordinary sample standard
    deviation.  It remains in ``[0, 180]`` even for opposing members, unlike unwrapping
    and applying a linear standard deviation.  Fewer than two finite members have no
    spread and return NaN.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    if center is None:
        center = circular_mean(x)
    if not np.isfinite(center):
        return float("nan")
    deviation = circular_abs_error(x, center)
    return float(np.sqrt(np.mean(deviation ** 2)))
