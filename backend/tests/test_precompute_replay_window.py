"""Replay has to cover all 666 districts too, which means precomputing its cycles.

`score_cycle` reads a precomputed artifact when one exists and otherwise scores live. That
made the national map cheap, and left Replay exactly as expensive as before: it offers the
newest `replay_service._MAX_CYCLES` cycles and scores whichever one is picked, on request.
On a 666-district store that is the 1,406 MB path this was built to avoid, reached by a
click rather than by a page load.

The window is small, so the fix is to precompute it rather than to cap what Replay shows:
10 cycles at 3.96 MB is ~40 MB, against a data bundle that is already 52 MB and a model
tarball of 11.9 MB.

The count is read from `replay_service._MAX_CYCLES` rather than restated. Two constants
that must agree and are written down twice eventually disagree, and the failure is silent
- Replay would offer a cycle nothing had precomputed, and the box would score it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.ml import precomputed
from app.services import replay_service


def test_the_precompute_window_is_replays_own_constant():
    """Not "10" written twice. If Replay's cap moves, the packaged set moves with it."""
    from scripts import package_for_deploy

    assert package_for_deploy.PRECOMPUTE_CYCLES is replay_service._MAX_CYCLES or \
        package_for_deploy.PRECOMPUTE_CYCLES == replay_service._MAX_CYCLES
    assert package_for_deploy.PRECOMPUTE_CYCLES >= 1


def test_every_cycle_replay_offers_gets_precomputed(tmp_path, monkeypatch):
    """The set that is written must be the set Replay can ask for - no fewer."""
    from scripts import package_for_deploy

    cycles = [pd.Timestamp("2018-12-31") - pd.Timedelta(days=i) for i in range(25)]
    written: list = []

    monkeypatch.setattr(package_for_deploy, "_available_cycles", lambda: cycles)
    monkeypatch.setattr(package_for_deploy, "_score_and_write",
                        lambda state, init: (written.append(init), 4_000_000)[1])

    n, total = package_for_deploy.precompute_cycles(state=object())

    assert n == replay_service._MAX_CYCLES
    assert written == cycles[:replay_service._MAX_CYCLES], (
        "must precompute the NEWEST cycles, which are the ones Replay offers")
    assert total == 4_000_000 * replay_service._MAX_CYCLES


def test_a_store_with_fewer_cycles_than_the_window_is_not_an_error(tmp_path, monkeypatch):
    """A fresh store has one or two cycles. Precompute what exists and move on."""
    from scripts import package_for_deploy

    monkeypatch.setattr(package_for_deploy, "_available_cycles",
                        lambda: [pd.Timestamp("2018-12-31")])
    monkeypatch.setattr(package_for_deploy, "_score_and_write", lambda state, init: 1000)
    n, _ = package_for_deploy.precompute_cycles(state=object())
    assert n == 1


def test_one_cycle_failing_does_not_lose_the_others(tmp_path, monkeypatch):
    """A cycle that cannot be scored must not take the rest of the window with it.

    Refuse rather than patch applies to the cycle, not to the batch: the one that failed
    simply has no artifact and will be scored live, which is the pre-existing behaviour.
    """
    from scripts import package_for_deploy

    cycles = [pd.Timestamp("2018-12-31") - pd.Timedelta(days=i) for i in range(4)]
    monkeypatch.setattr(package_for_deploy, "_available_cycles", lambda: cycles)

    def flaky(state, init):
        if init == cycles[1]:
            raise RuntimeError("no scoreable rows for this cycle")
        return 1000

    monkeypatch.setattr(package_for_deploy, "_score_and_write", flaky)
    n, total = package_for_deploy.precompute_cycles(state=object())
    assert n == 3, "the three that scored are still packaged"
    assert total == 3000


def test_the_score_cache_is_not_held_across_cycles(tmp_path, monkeypatch):
    """Ten cycles held at once is ten frames of 6,660 events in the packaging process.

    inference._score_cache keeps up to 24, so without clearing between cycles the runner
    accumulates the whole window. It has 16 GB and would survive it, but the same loop is
    the one that would run against a larger window later.
    """
    from app.ml import inference
    from scripts import package_for_deploy

    cycles = [pd.Timestamp("2018-12-31") - pd.Timedelta(days=i) for i in range(3)]
    sizes: list = []

    monkeypatch.setattr(package_for_deploy, "_available_cycles", lambda: cycles)

    def one(state, init):
        sizes.append(len(inference._score_cache))
        return 1000

    monkeypatch.setattr(package_for_deploy, "_score_and_write", one)
    package_for_deploy.precompute_cycles(state=object())
    assert max(sizes) <= 1, f"score cache grew across cycles: {sizes}"
