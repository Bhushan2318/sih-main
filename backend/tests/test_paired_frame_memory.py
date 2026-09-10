"""The paired frame has to fit in memory, and at district grain it did not.

Measured 2026-12-15 cycle, 661 districts: 208,155 paired rows at 88.7 MB, which is
31.6 GB across 365 cycles. The box has 16 GB and so does a CI runner, so
`_build_paired_in_chunks` - which chunks the read and then concatenates - hit a hard wall
rather than a risk.

61% of that was four low-cardinality string columns held as Python objects: variable (8
distinct), value_type (2), verification_status (2), ensemble_member_id (5). region_id and
season were already categorical, so the pattern was established and these were missed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app import contracts
from app.ml.train_pipeline import _downcast_paired

# Distinct values taken from the real store, not invented.
_VARIABLES = ["temperature_c", "humidity_pct", "rainfall_mm", "pressure_hpa",
              "atmospheric_moisture_kgm2", "soil_moisture_pct", "wind_speed_ms",
              "wind_direction_deg"]
_MEMBERS = ["gec00", "gep01", "gep02", "gep03", "gep04"]


def _frame(n: int = 20_000) -> pd.DataFrame:
    """A paired frame shaped like the real one. Values are arbitrary; the dtypes and the
    cardinality are the point."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: contracts.example_column(c, n)
                       for c in contracts.PAIRED_ROW_COLUMNS})
    df["variable"] = rng.choice(_VARIABLES, n)
    df["value_type"] = "forecast"
    df["verification_status"] = "final"
    df["ensemble_member_id"] = rng.choice(_MEMBERS, n)
    return df


def test_downcast_shrinks_the_frame_materially():
    """At least halves it.

    The real gain is larger - 88.7 MB -> 20.7 MB, 4.3x, measured on the 2017-12-15 cycle
    at 661 districts - but a synthetic fixture cannot honestly reproduce that ratio,
    because pandas charges for an object column by the actual string, and these fixtures
    are shorter than the real values. Asserting 4.3x here would be fitting the test to a
    number this frame cannot produce. What is pinned is that the reduction is material;
    the real figure lives in the docstring and the commit that measured it."""
    before = _frame()
    mb_before = before.memory_usage(deep=True).sum() / 1e6
    after = _downcast_paired(before.copy())
    mb_after = after.memory_usage(deep=True).sum() / 1e6
    assert mb_after < mb_before / 2, (
        f"expected at least a halving, got {mb_before:.1f} -> {mb_after:.1f} MB")


def test_downcast_preserves_every_value():
    """Cheaper storage, identical data. A downcast that quietly changed a value would
    move every metric downstream."""
    before = _frame(5_000)
    after = _downcast_paired(before.copy())
    for col in contracts.PAIRED_ROW_COLUMNS:
        b, a = before[col], after[col]
        if b.dtype.kind == "f":
            # float64 -> float32 keeps ~7 significant digits, far more than a
            # meteorological value carries.
            assert np.allclose(b.to_numpy(dtype="float64"),
                               a.to_numpy(dtype="float64"),
                               rtol=1e-6, equal_nan=True), col
        else:
            assert list(b.astype(str)) == list(a.astype(str)), col


def test_the_low_cardinality_strings_become_categories():
    after = _downcast_paired(_frame())
    for col in ("variable", "value_type", "verification_status", "ensemble_member_id"):
        assert isinstance(after[col].dtype, pd.CategoricalDtype), f"{col} is still object"


def test_a_downcast_frame_still_satisfies_the_contract():
    """The frozen contract must accept the cheaper dtypes, or the trainer and the
    contract disagree about the same frame."""
    contracts.validate_paired_frame(_downcast_paired(_frame(1_000)))


def test_downcast_is_idempotent():
    once = _downcast_paired(_frame(1_000))
    twice = _downcast_paired(once.copy())
    assert (once.dtypes.astype(str) == twice.dtypes.astype(str)).all()
