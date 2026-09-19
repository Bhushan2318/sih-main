"""C3 - MJO (Madden-Julian Oscillation), via NOAA PSL's OMI index.

BOM's canonical RMM index (http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt)
returns HTTP 403 with "The Bureau of Meteorology website does not support web scraping"
for any automated request - verified 2026-09-18, not assumed. NOAA PSL's OMI index
(https://psl.noaa.gov/mjo/mjoindex/omi.1x.txt) is a real, no-auth, government-hosted
alternative, updated daily since 1991; PSL's own documentation gives the transform to the
standard RMM convention: RMM1 = OMI(PC2), RMM2 = -OMI(PC1) (correlation > 0.93 with BOM's
RMM, per PSL). MISO has no verified public real-time source and is not built here.

No discrete 1-8 MJO phase is computed: phase-boundary conventions vary across sources and
could not be independently verified given BOM's access block. mjo_rmm1/mjo_rmm2/
mjo_amplitude already carry the same information a model can use directly.

The MJO index is global, not per-district - one row per calendar day, not per region_id -
so it is attached by an as-of (backward) join on init_date, never valid_date: only the
most recent MJO reading ON OR BEFORE the day a forecast was issued could have been known
at issue time. A reading older than MJO_ASOF_TOLERANCE_DAYS is treated as unknown, not
used as if it were current - the same "missing never becomes zero" discipline as C1/C2.

Every expected value below is computed by hand in the comment beside it. The input
numbers are small, invented and labelled as such.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.features import engineering as fe
from scripts import fetch_mjo_index as fmi


def test_omi_to_rmm_swaps_and_flips_per_psls_documented_convention():
    # PSL: "the sign of OMI PC1 and the PC ordering should be reversed, so that
    # OMI(PC2) is analogous to RMM(PC1) and -OMI(PC1) is analogous to RMM(PC2)."
    rmm1, rmm2 = fmi._omi_to_rmm(pc1=3.0, pc2=4.0)
    assert rmm1 == pytest.approx(4.0)     # RMM1 = PC2
    assert rmm2 == pytest.approx(-3.0)    # RMM2 = -PC1


def test_amplitude_is_invariant_to_the_swap_and_flip():
    # sqrt(3^2 + 4^2) = 5, and a reflection/swap does not change a vector's length.
    rmm1, rmm2 = fmi._omi_to_rmm(pc1=3.0, pc2=4.0)
    amp = math.hypot(rmm1, rmm2)
    assert amp == pytest.approx(5.0)


def test_parsing_a_real_shaped_line():
    """One real line from the fetched file (2026-09-18), not invented."""
    df = fmi.parse("1991 1 1 0.23390 -0.69290 0.73132\n")
    row = df.iloc[0]
    assert row["date"] == pd.Timestamp("1991-01-01")
    assert row["mjo_rmm1"] == pytest.approx(-0.69290)     # PC2
    assert row["mjo_rmm2"] == pytest.approx(-0.23390)     # -PC1
    # the file's own amplitude column, sqrt(pc1^2+pc2^2) = sqrt(0.2339^2+0.6929^2):
    assert row["mjo_amplitude"] == pytest.approx(
        math.hypot(0.23390, -0.69290), rel=1e-4)


# ----------------------------------------------------------- attach, hand-computed

_MJO = pd.DataFrame({
    "date": pd.to_datetime(["2017-11-01", "2017-11-02", "2017-11-05"]),
    "mjo_rmm1": [1.0, 2.0, 5.0],
    "mjo_rmm2": [-1.0, -2.0, -5.0],
    "mjo_amplitude": [1.414214, 2.828427, 7.071068],
})


def _paired(init_dates):
    return pd.DataFrame({
        "region_id": ["A"] * len(init_dates),
        "init_date": pd.to_datetime(init_dates),
    })


def test_an_exact_day_match_is_used():
    out = fe.attach_mjo_index(_paired(["2017-11-02"]), _MJO)
    assert out.loc[0, "mjo_rmm1"] == pytest.approx(2.0)
    assert out.loc[0, "mjo_rmm2"] == pytest.approx(-2.0)


def test_the_most_recent_prior_day_is_used_when_there_is_no_exact_match():
    # 11-04 has no MJO row; the nearest one ON OR BEFORE it is 11-02, not 11-05.
    out = fe.attach_mjo_index(_paired(["2017-11-04"]), _MJO)
    assert out.loc[0, "mjo_rmm1"] == pytest.approx(2.0)


def test_a_future_reading_is_never_used_even_if_it_is_the_nearest_in_absolute_time():
    """Causality: a two-point MJO history, 11-01 and 11-05. Querying 11-04 - one day
    before 11-05, three days after 11-01, so 11-05 is nearer in absolute time - must
    still return 11-01: 11-05 could not have been known yet on 11-04."""
    sparse = pd.DataFrame({
        "date": pd.to_datetime(["2017-11-01", "2017-11-05"]),
        "mjo_rmm1": [1.0, 5.0], "mjo_rmm2": [-1.0, -5.0],
        "mjo_amplitude": [1.414214, 7.071068],
    })
    out = fe.attach_mjo_index(_paired(["2017-11-04"]), sparse)
    assert out.loc[0, "mjo_rmm1"] == pytest.approx(1.0)     # 11-01, not the nearer 11-05


def test_a_reading_older_than_the_tolerance_is_unknown_not_stale():
    # _MJO's last reading is 2017-11-05; querying tolerance+1 days after it must miss.
    too_late = _paired(["2017-11-01"])
    too_late["init_date"] = pd.Timestamp("2017-11-05") + pd.Timedelta(
        days=fe.MJO_ASOF_TOLERANCE_DAYS + 1)
    out = fe.attach_mjo_index(too_late, _MJO)
    assert math.isnan(out.loc[0, "mjo_rmm1"])
    assert math.isnan(out.loc[0, "mjo_amplitude"])


def test_a_reading_within_the_tolerance_boundary_is_used():
    # Exactly tolerance days after the last reading (2017-11-05) must still match it.
    at_boundary = _paired(["2017-11-01"])
    at_boundary["init_date"] = pd.Timestamp("2017-11-05") + pd.Timedelta(
        days=fe.MJO_ASOF_TOLERANCE_DAYS)
    out = fe.attach_mjo_index(at_boundary, _MJO)
    assert out.loc[0, "mjo_rmm1"] == pytest.approx(5.0)


def test_a_date_before_any_mjo_history_is_unknown():
    out = fe.attach_mjo_index(_paired(["2017-10-01"]), _MJO)
    assert math.isnan(out.loc[0, "mjo_rmm1"])


def test_every_row_of_the_same_cycle_gets_the_same_global_value():
    """MJO is global, not per-district - two districts issuing forecasts the same day
    see the identical reading."""
    two = pd.DataFrame({
        "region_id": ["A", "B"],
        "init_date": pd.to_datetime(["2017-11-02", "2017-11-02"]),
    })
    out = fe.attach_mjo_index(two, _MJO)
    assert out["mjo_rmm1"].nunique() == 1


# --------------------------------------------------------------------- wired into frames

def _canonical(region="A", variable="temperature_c",
              init="2017-11-02", valid="2017-11-03"):
    """PLUMBING FIXTURE - invented numbers, never used to produce a metric."""
    fc = pd.DataFrame([{
        "region_id": region, "variable": variable, "value_type": "forecast",
        "valid_date": pd.Timestamp(valid), "init_date": pd.Timestamp(init),
        "lead_time_days": 2, "ensemble_member_id": "m0", "value": 30.0,
    }])
    ob = pd.DataFrame([{
        "region_id": region, "variable": variable, "value_type": "observed",
        "valid_date": pd.Timestamp(valid), "value": 31.0,
    }])
    return pd.concat([fc, ob], ignore_index=True)


def test_the_training_frame_carries_the_mjo_features(monkeypatch):
    monkeypatch.setattr(fe, "load_mjo_index", lambda: _MJO)
    frame = fe.build_training_frame(_canonical())
    for col in fe.MJO_FEATURES:
        assert col in frame.columns
    assert frame.loc[0, "mjo_rmm1"] == pytest.approx(2.0)   # 2017-11-02, exact match


def test_regressor_and_classifier_both_see_the_mjo_features(monkeypatch):
    from app.features import pivot as pv
    from app.ml import regressors as reg_mod

    monkeypatch.setattr(fe, "load_mjo_index", lambda: _MJO)
    frame = fe.build_training_frame(_canonical())
    for col in fe.MJO_FEATURES:
        assert col in reg_mod.feature_columns(frame)

    pred = pd.Series(1.0, index=frame.index)
    ev = pv.build_event_frame(frame, pred, {"temperature_c": 5.0}, {"temperature_c": 3.0})
    for col in fe.MJO_FEATURES:
        assert col in ev.columns
        assert col in reg_mod.feature_columns(ev) or col in pv.classifier_feature_columns(ev)
