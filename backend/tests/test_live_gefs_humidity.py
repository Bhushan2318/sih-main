"""Live GEFS humidity is built the way the reforecast's was.

The reforecast archive has no 2 m RH; scripts/fetch_gefs_reforecast_sample.py averages
specific humidity, temperature and surface pressure over the district and the day, and
derives RH from those means (Bolton 1980, app/utils/humidity.py). The live feed read GEFS's
own RH product and averaged it - a different estimator for the same air, fed to models
trained on the first.

The live 0.25 deg file (pgrb2s) carries RH but not SPFH at 2 m; only the 0.5 deg b-file has
SPFH. So each cell's q is recovered from GEFS's RH, T and p with the exact inverse of the
formula the reforecast uses. Measured 2026-10-04 on gefs.20261002/00 gep01 f012 and f024
(3,720 points where the two grids coincide): Bolton RH from GEFS's own SPFH minus GEFS's RH
product averaged +0.035 and -0.010 %RH, so the inversion returns GEFS's q (median ratio
error -0.13% and -0.05%); below 0 C, where GEFS uses ice, it differs by -0.4 to -0.7 %RH.

SYNTHETIC, LABELLED: step values are generated to check the arithmetic of the reduction.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from app.live import gefs
from app.utils import humidity


def test_q_from_rh_inverts_rh_from_q_exactly():
    t = np.array([275.0, 290.0, 305.0, 310.0])
    p = np.array([60000.0, 85000.0, 100000.0, 101300.0])
    q = np.array([0.002, 0.008, 0.016, 0.022])
    rh = humidity.rh_from_specific_humidity(q, t, p)
    assert np.all(rh < 100)
    assert np.allclose(humidity.specific_humidity_from_rh(rh, t, p), q, rtol=1e-12)


def _steps(n_region, member="gec00", t=None, rh=None, p=None):
    """Every 3-hourly step of a 00Z cycle, with per-step district values."""
    rng = np.random.default_rng(0)
    out = {}
    for fh in range(3, gefs.MAX_LEAD_H + 1, 3):
        tk = t if t is not None else rng.uniform(285, 310, n_region)
        r = rh if rh is not None else rng.uniform(20, 95, n_region)
        pp = p if p is not None else rng.uniform(80000, 101000, n_region)
        out[(member, fh)] = {
            "t2m_c": tk, "rh2m_pct": r, "psfc_hpa": pp,
            "q2m_kgkg": humidity.specific_humidity_from_rh(r, tk, pp),
            "mslp_hpa": np.full(n_region, 101000.0), "pwat_kgm2": np.full(n_region, 40.0),
            "u10": np.full(n_region, 1.0), "v10": np.full(n_region, 1.0),
            "soilw_vol_pct": np.full(n_region, 0.3),
        }
        if fh % 6 == 0:
            out[(member, fh)]["apcp_mm"] = np.zeros(n_region)
    return out


def test_daily_rh_comes_from_the_daily_mean_q_t_and_p():
    n = len(gefs._districts())
    steps = _steps(n)
    frame = gefs._reduce_to_daily(steps, date(2026, 10, 2), "00", ["gec00"], "s3")
    day1 = frame[frame.lead_day == 1].reset_index(drop=True)
    hours = list(range(3, 25, 3))
    q = np.mean([steps[("gec00", h)]["q2m_kgkg"] for h in hours], axis=0)
    t = np.mean([steps[("gec00", h)]["t2m_c"] for h in hours], axis=0)
    p = np.mean([steps[("gec00", h)]["psfc_hpa"] for h in hours], axis=0)
    rh_mean = np.mean([steps[("gec00", h)]["rh2m_pct"] for h in hours], axis=0)
    got = day1["rh2m_pct"].to_numpy()
    ids = gefs._districts()["region_id"].to_numpy()
    want = humidity.rh_from_specific_humidity(q, t, p)
    pos = {r: i for i, r in enumerate(ids)}
    want = np.array([want[pos[r]] for r in day1["region_id"]])
    old = np.array([rh_mean[pos[r]] for r in day1["region_id"]])
    assert np.allclose(got, want, rtol=1e-10)
    assert np.abs(got - old).max() > 0.1, "not the mean of the RH product"
    assert "q2m_kgkg" not in frame.columns


def test_constant_air_keeps_its_rh():
    n = len(gefs._districts())
    steps = _steps(n, t=np.full(n, 300.0), rh=np.full(n, 70.0), p=np.full(n, 95000.0))
    frame = gefs._reduce_to_daily(steps, date(2026, 10, 2), "00", ["gec00"], "s3")
    assert np.allclose(frame["rh2m_pct"], 70.0, atol=1e-9)


def test_without_q_there_is_no_rh_rather_than_the_old_estimator():
    n = len(gefs._districts())
    steps = _steps(n)
    for v in steps.values():
        v.pop("q2m_kgkg")
    frame = gefs._reduce_to_daily(steps, date(2026, 10, 2), "00", ["gec00"], "s3")
    assert "rh2m_pct" not in frame.columns or frame["rh2m_pct"].isna().all()
    assert frame["t2m_c"].notna().all()
