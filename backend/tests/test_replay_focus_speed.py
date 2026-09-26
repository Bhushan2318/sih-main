"""Replay must build its forecast-vs-observed charts in one pass, not one pass per district.

`_build_focus` called `_focus_for_region` for each of the 666 districts, and each call
filtered the whole per-variable table again, converting every region id to a string on
the way. Measured 2026-09-26 on the published bundle: 4.98 of get_replay's 5.25 s for the
live cycle - whose outcome is not known yet, so all 666 passes found nothing - and 6.2 of
6.4 s for Kerala 2018. Render's free tier has a fraction of that laptop's CPU.

The table below is a SHAPE fixture: random values in the per-variable layout at the real
size (666 districts x 10 lead days x 8 variables). It exercises plumbing and timing only
and never reaches anything that produces a metric. Equivalence with the old behaviour on
real data is pinned in test_ml.py (test_focus_options_match_a_direct_filter).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from app.ml.inference import ScoredCycle
from app.ml.thresholds import Thresholds
from app.services import replay_service

_VARS = ["atmospheric_moisture_kgm2", "humidity_pct", "pressure_hpa", "rainfall_mm",
         "soil_moisture_pct", "temperature_c", "wind_direction_deg", "wind_speed_ms"]


class _State:
    thresholds = Thresholds(bust_threshold={v: 1.0 for v in _VARS},
                            p90_error={v: 2.0 for v in _VARS},
                            risk_band_cuts={"medium": 0.45, "high": 0.81})


def _shape_cycle(verified: bool) -> ScoredCycle:
    rng = np.random.default_rng(0)
    regions = [f"IN-XX-R{i:03d}" for i in range(666)]
    leads = range(1, 11)
    ev = pd.DataFrame([(r, d, pd.Timestamp("2018-08-13") + pd.Timedelta(days=d - 1))
                       for r in regions for d in leads],
                      columns=["region_id", "lead_time_days", "valid_date"])
    ev["region_id"] = ev["region_id"].astype("category")
    ev["bust_probability"] = rng.random(len(ev))
    ev["risk_band"] = "low"
    ev["dominant_variable"] = "rainfall_mm"
    pv = pd.DataFrame([(r, d, v) for r in regions for d in leads for v in _VARS],
                      columns=["region_id", "lead_time_days", "variable"])
    pv["region_id"] = pv["region_id"].astype("category")
    pv["valid_date"] = pd.Timestamp("2018-08-13") + pd.to_timedelta(pv["lead_time_days"] - 1, "D")
    pv["predicted_value"] = rng.random(len(pv))
    pv["observed_value"] = rng.random(len(pv)) if verified else np.nan
    pv["ensemble_spread"] = rng.random(len(pv))
    return ScoredCycle(run_id="run_SHAPE", init_date=pd.Timestamp("2018-08-13"),
                       events=ev, per_variable=pv, n_rows_scored=len(ev))


def _timed(sc):
    t = time.perf_counter()
    out = replay_service._build_focus(sc, _State(), None)
    return time.perf_counter() - t, out


def test_an_unverified_cycle_builds_no_charts_and_does_not_search_for_them():
    """The live cycle: nothing observed yet, so there is nothing to chart - and finding
    that out must not cost a pass over every district."""
    secs, (default, options) = _timed(_shape_cycle(verified=False))
    assert default is None and options == []
    assert secs < 0.5, f"{secs:.2f}s to find that nothing is verified"


def test_a_verified_cycle_charts_every_district_in_one_pass():
    secs, (default, options) = _timed(_shape_cycle(verified=True))
    assert len(options) == 666
    assert default is options[0]
    assert all(len(o.points) == 10 for o in options)
    assert secs < 1.5, f"{secs:.2f}s for 666 districts"
