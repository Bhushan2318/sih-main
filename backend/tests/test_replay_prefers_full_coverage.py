"""Replay must open on a cycle that covers the country, not a 36-district one.

Measured on the live site 2026-09-23, after the district migration shipped:

  /api/replay/cycles  offered 10 cycles
  /api/replay         defaulted to 2026-09-15 -> 36 districts per lead day

The store is cumulative, so it holds one 666-district cycle and nine older ones from when
the live feed sampled 36 city points. `list_cycles` sorted on `verified` first, and the
verified cycles are all the old sparse ones - the full-coverage cycle is today's, whose
outcome has not happened yet. So the flagship feature opened on a map of India with 36
districts drawn on it, which reads as broken rather than as young.

Coverage has to outrank everything else. A cycle that cannot show the country cannot show
a bust, whatever else is true about it - and the "outcome not yet known" case is already
handled in the UI with an explanation, where a near-empty map is not.

Coverage is compared in tiers rather than exactly, so that a cycle missing a handful of
districts is not ranked below one with six more; within a tier the existing ordering
(verified, growth, peak) still decides.
"""

from __future__ import annotations

import pytest

from app.services import replay_service


class _Cycle:
    """The fields of ReplayCycleSummary that the ordering reads."""

    def __init__(self, init_date, n_regions, verified, growth=0.0, peak=0.5,
                 verified_lead_days=0, relative_error=None):
        self.init_date = init_date
        self.n_regions = n_regions
        self.verified = verified
        self.medium_range_growth = growth
        self.peak_bust_probability = peak
        self.verified_lead_days = verified_lead_days
        self.peak_region_relative_error = relative_error

    def __repr__(self):
        return f"<{self.init_date} n={self.n_regions} verified={self.verified}>"


def test_full_coverage_outranks_a_verified_sparse_cycle():
    """The exact situation on the live site: nine verified 36s and one unverified 666."""
    sparse = [_Cycle(f"2026-09-{d:02d}", 36, True, verified_lead_days=10)
              for d in range(12, 21)]
    full = _Cycle("2026-09-23", 666, False)
    ordered = replay_service.order_cycles(sparse + [full])
    assert ordered[0] is full, (
        f"Replay would open on {ordered[0]}, a sparse cycle, because it is verified")


def test_within_the_same_coverage_verified_still_wins():
    """Coverage decides between tiers, not inside one. The old ordering still applies."""
    unverified = _Cycle("2026-09-23", 666, False, growth=0.9, peak=0.99)
    verified = _Cycle("2026-09-22", 666, True, growth=0.1, peak=0.6,
                      verified_lead_days=8)
    ordered = replay_service.order_cycles([unverified, verified])
    assert ordered[0] is verified


def test_a_few_missing_districts_does_not_demote_a_cycle():
    """660 and 666 are the same map. Tiers stop a 6-district difference deciding this."""
    interesting = _Cycle("2026-09-22", 660, True, growth=0.8, peak=0.99,
                         verified_lead_days=9)
    dull = _Cycle("2026-09-23", 666, True, growth=0.0, peak=0.51, verified_lead_days=1)
    ordered = replay_service.order_cycles([dull, interesting])
    assert ordered[0] is interesting, "a 6-district edge outranked a far better cycle"


def test_ordering_survives_an_empty_list():
    assert replay_service.order_cycles([]) == []


def test_ordering_survives_a_cycle_with_no_peak_probability():
    """A cycle the model could not score a peak for must not break the sort."""
    a = _Cycle("2026-09-23", 666, True, peak=None, verified_lead_days=3)
    b = _Cycle("2026-09-22", 666, True, peak=0.9, verified_lead_days=3)
    ordered = replay_service.order_cycles([a, b])
    assert ordered[0] is b
    assert set(ordered) == {a, b}


def test_a_badly_missed_verified_cycle_outranks_a_barely_missed_one():
    """Within the same coverage tier and both verified, actual error should decide -
    not just how confident the model was or how much its risk grew with lead time.

    Confirmed missing before this fix: order_cycles's key had no error term at all, so
    a cycle whose forecast was nearly right could outrank one that missed badly, purely
    on peak_bust_probability or medium_range_growth.
    """
    barely_missed = _Cycle("2026-09-20", 666, True, growth=0.5, peak=0.9,
                            verified_lead_days=10, relative_error=0.1)
    badly_missed = _Cycle("2026-09-19", 666, True, growth=0.1, peak=0.6,
                           verified_lead_days=10, relative_error=2.4)
    ordered = replay_service.order_cycles([barely_missed, badly_missed])
    assert ordered[0] is badly_missed, (
        f"Replay would open on {ordered[0]}, which missed by only "
        f"{ordered[0].peak_region_relative_error}x its threshold, ahead of a cycle "
        f"that missed by {badly_missed.peak_region_relative_error}x"
    )


def test_relative_error_only_matters_within_a_verified_tier():
    """An unverified cycle (relative_error=None) must not be able to win on error alone -
    'badly missed' presupposes the outcome is known."""
    unverified_with_stale_error = _Cycle("2026-09-23", 666, False, peak=0.5,
                                          relative_error=None)
    verified_mild_miss = _Cycle("2026-09-22", 666, True, peak=0.5,
                                 verified_lead_days=5, relative_error=0.2)
    ordered = replay_service.order_cycles(
        [unverified_with_stale_error, verified_mild_miss])
    assert ordered[0] is verified_mild_miss


def test_every_cycle_is_kept_not_filtered():
    """Sparse cycles stay selectable - they are real runs and their data is real.

    Demoting them is the fix; hiding them would throw away the only cycles whose outcome
    is currently known.
    """
    cycles = [_Cycle("2026-09-15", 36, True), _Cycle("2026-09-23", 666, False)]
    assert len(replay_service.order_cycles(cycles)) == 2
