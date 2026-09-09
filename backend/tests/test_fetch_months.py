"""Month selection for the daily fetch.

GitHub Actions kills a job at 6 hours. A full year at daily density is ~842 GB, which is
too tight for one job even at CI's ~80 MB/s, so the year is matrixed by month. This is
the argument that carves it up.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fetch_gefs_m", BACKEND / "scripts" / "fetch_gefs_reforecast_sample.py")
fetch = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fetch
_spec.loader.exec_module(fetch)


def test_one_month_of_daily_dates():
    got = fetch.init_dates_for([2017], stride=1, months=[11])
    assert len(got) == 30, "November has 30 days"
    assert got[0] == "2017-11-01" and got[-1] == "2017-11-30"


def test_a_month_range():
    got = fetch.init_dates_for([2017], stride=1, months=list(range(11, 13)))
    assert len(got) == 61, "November 30 + December 31"


def test_no_month_filter_is_the_whole_year():
    assert len(fetch.init_dates_for([2017], stride=1)) == 365


def test_leap_year_february():
    assert len(fetch.init_dates_for([2016], stride=1, months=[2])) == 29


def test_months_apply_to_the_seasonal_pattern_too():
    """The filter is on dates, not on how they were generated, so it composes with the
    17-a-year pattern rather than only with stride."""
    got = fetch.init_dates_for([2017], stride=0, months=[7])
    assert got and all(d.startswith("2017-07") for d in got)


def test_ockhi_window_is_inside_november_and_december():
    """Cyclone Ockhi ran 29 Nov - 6 Dec 2017. Every initialisation that was supposed to
    see it coming has to be in the fetched set."""
    got = set(fetch.init_dates_for([2017], stride=1, months=[11, 12]))
    for d in ("2017-11-25", "2017-11-28", "2017-11-29", "2017-12-01", "2017-12-06"):
        assert d in got, d


def test_parse_months_accepts_the_forms_the_workflow_sends():
    assert fetch.parse_months("11") == [11]
    assert fetch.parse_months("1-3") == [1, 2, 3]
    assert fetch.parse_months("1,6,12") == [1, 6, 12]
    assert fetch.parse_months("") is None
    assert fetch.parse_months(None) is None


def test_a_month_outside_the_calendar_is_refused():
    with pytest.raises(ValueError):
        fetch.parse_months("13")
