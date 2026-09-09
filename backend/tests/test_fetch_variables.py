"""Selecting messages by level, and keeping predictor-only fields out of the paired store.

The reforecast publishes 30 variable files; the pipeline reads 9. The unused ones include
the fields that actually carry predictability — Z500 for the synoptic pattern, CAPE and
CIN for convection, which is what drives rainfall busts.

Two things have to be true before they can be added:

*Selection by level.* Z500 lives inside hgt_pres_abv700mb, which holds 1,440 messages
across 18 pressure levels; only 80 of them are 500 mb. Because the fetch reads by byte
range off the .idx, taking just those 80 costs ~16 MB rather than the file's 296 MB. The
existing level filter is hardcoded to two special cases and cannot express "500 mb".

*Predictor-only fields.* These have no observed counterpart — there is no ERA5 "CAPE
observation" to verify a CAPE forecast against — so they cannot join the paired store,
where every row is a forecast matched to an observation. They belong in the grid the
convolutional model reads. Letting them into the tabular path would create rows that can
never pair and would silently dilute every count derived from it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fetch_gefs_v", BACKEND / "scripts" / "fetch_gefs_reforecast_sample.py")
fetch = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fetch
_spec.loader.exec_module(fetch)


def _rec(msg, level, fcst):
    return fetch.IdxRecord(msg, msg * 1000, (msg + 1) * 1000, "HGT", level, fcst)


def _multi_level_day():
    """One lead day of a file laid out like hgt_pres_abv700mb: every level repeated at
    every 3-hourly step."""
    levels = ["1000 mb", "700 mb", "500 mb", "300 mb"]
    recs, msg = [], 1
    for hour in range(3, 27, 3):
        for lvl in levels:
            recs.append(_rec(msg, lvl, f"{hour} hour fcst"))
            msg += 1
    return recs


def test_level_filter_picks_only_the_wanted_level():
    spec = {"short_name": "gh", "max_lead_h": 240, "accum": False, "level": "500 mb"}
    got = fetch.select_for_day(_multi_level_day(), spec, 1)
    assert got, "nothing selected"
    assert all(r.level.strip() == "500 mb" for r in got)
    assert len(got) == 8, "eight 3-hourly steps in a lead day"


def test_without_a_level_filter_every_level_is_taken():
    spec = {"short_name": "gh", "max_lead_h": 240, "accum": False}
    got = fetch.select_for_day(_multi_level_day(), spec, 1)
    assert len(got) == 32, "8 steps x 4 levels when nothing filters"


def test_a_level_that_is_not_present_selects_nothing():
    """Better to select nothing, and have the completeness check refuse the cycle, than
    to quietly fall back to a different level."""
    spec = {"short_name": "gh", "max_lead_h": 240, "accum": False, "level": "250 mb"}
    assert fetch.select_for_day(_multi_level_day(), spec, 1) == []


def test_the_existing_wind_and_soil_filters_still_work():
    """These were special-cased before the general filter existed; they must not change."""
    wind = [_rec(1, "10 m above ground", "3 hour fcst"),
            _rec(2, "80 m above ground", "3 hour fcst")]
    spec = {"short_name": "u10", "max_lead_h": 120, "accum": False,
            "level_key": "heightAboveGround", "level_val": 10.0}
    got = fetch.select_for_day(wind, spec, 1)
    assert len(got) == 1 and got[0].level.strip() == "10 m above ground"


# --------------------------------------------------- predictor-only fields

def test_the_new_predictor_fields_are_declared_grid_only():
    for v in ("hgt_pres_abv700mb", "cape_sfc", "cin_sfc"):
        assert v in fetch.VAR_SPEC, f"{v} not declared"
        assert fetch.VAR_SPEC[v].get("grid_only") is True, \
            f"{v} has no observed counterpart and must not enter the paired store"


def test_z500_is_selected_by_level_not_by_whole_file():
    """The whole file is 296 MB; the 500 mb messages are ~16 MB of it."""
    assert fetch.VAR_SPEC["hgt_pres_abv700mb"].get("level") == "500 mb"


def test_the_original_nine_stay_pairable():
    """Everything that had an observed counterpart before must still have one."""
    for v in ("tmp_2m", "apcp_sfc", "pres_msl", "spfh_2m", "pwat_eatm", "soilw_bgrnd"):
        assert not fetch.VAR_SPEC[v].get("grid_only"), f"{v} must stay pairable"


def test_grid_only_fields_are_excluded_from_the_tabular_columns():
    """The canonical column list is what reaches the paired store."""
    tabular = [v for v, s in fetch.VAR_SPEC.items() if not s.get("grid_only")]
    assert "cape_sfc" not in tabular and "hgt_pres_abv700mb" not in tabular
    assert "tmp_2m" in tabular
