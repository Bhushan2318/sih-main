"""A cycle is complete or it is refused. There is no third option.

Rule 3 of the working agreement: an incomplete cycle is rejected, not partially
ingested. This matters more on the daily path than it did on the seasonal one - 366
cycles x 45 (variable, member) downloads is ~16,500 range-GET sequences against a public
bucket, and at that volume a transient failure is a certainty rather than a risk.

The failure being guarded is quiet. A cycle that lost three of its five members still
writes a parquet part and a grid bundle; `--resume` then sees both files and marks that
date done forever. The ensemble spread for those rows is computed over two members
instead of five, which does not look wrong - it looks like a calm day.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fetch_gefs", BACKEND / "scripts" / "fetch_gefs_reforecast_sample.py")
fetch = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fetch
_spec.loader.exec_module(fetch)


ALL_MEMBERS = list(fetch.ALL_MEMBERS)
ALL_VARS = list(fetch.VAR_SPEC)


def _results(missing: set[tuple[str, str]] | None = None) -> dict:
    """The `results` dict `build` assembles: (variable, member) -> rows. A failed
    download contributes an empty frame, which is exactly what pull_one_file returns."""
    missing = missing or set()
    out = {}
    for m in ALL_MEMBERS:
        for v in ALL_VARS:
            out[(v, m)] = (pd.DataFrame() if (v, m) in missing
                           else pd.DataFrame({"region_id": ["IN-MH-NAGPUR"], v: [1.0]}))
    return out


def test_a_complete_cycle_is_accepted():
    ok, why = fetch.cycle_is_complete(_results(), ALL_MEMBERS)
    assert ok, why


def test_a_cycle_missing_one_member_is_refused():
    """Four members instead of five is a 20% narrower ensemble spread on every row of
    that cycle, and spread is one of the model's real inputs."""
    missing = {(v, "p03") for v in ALL_VARS}
    ok, why = fetch.cycle_is_complete(_results(missing), ALL_MEMBERS)
    assert not ok
    assert "p03" in why


def test_a_cycle_missing_one_variable_is_refused():
    missing = {(ALL_VARS[0], m) for m in ALL_MEMBERS}
    ok, why = fetch.cycle_is_complete(_results(missing), ALL_MEMBERS)
    assert not ok
    assert ALL_VARS[0] in why


def test_a_cycle_missing_a_single_file_is_refused():
    """One file of forty-five. The resulting rows look entirely normal."""
    ok, why = fetch.cycle_is_complete(_results({(ALL_VARS[2], "p01")}), ALL_MEMBERS)
    assert not ok


def test_refusal_names_what_was_missing():
    """The report has to be actionable: at 366 cycles, 'a cycle failed' is not enough
    to know whether to retry it or whether the archive simply lacks that date."""
    ok, why = fetch.cycle_is_complete(_results({(ALL_VARS[1], "p02")}), ALL_MEMBERS)
    assert not ok
    assert ALL_VARS[1] in why and "p02" in why


def test_an_empty_cycle_is_refused_not_crashed():
    ok, why = fetch.cycle_is_complete({}, ALL_MEMBERS)
    assert not ok


def test_completeness_is_relative_to_the_members_asked_for():
    """A deliberate single-member run is complete with one member. The check is against
    what was requested, not against a hardcoded five."""
    subset = ["c00"]
    results = {(v, "c00"): pd.DataFrame({"region_id": ["x"], v: [1.0]}) for v in ALL_VARS}
    ok, why = fetch.cycle_is_complete(results, subset)
    assert ok, why
