"""No observed day on both sides of any split, in the pooled retrain.

A cycle issued on day d verifies Day 1-10 against the observations of days d .. d+9. So
two cycles less than 10 days apart are checked against some of the same observed days,
and the later one's label is partly known to a model trained on the earlier one.

Before this:
  - pooled_split put the first validation cycle the day after the last training cycle,
    and the first test cycle the day after the last validation cycle;
  - assign_folds numbered sorted daily cycles i % 3, so a cycle's two neighbours - which
    share 9 of its 10 verified days - always sat in the other folds, and every fold model
    had effectively seen the days it was asked to predict out-of-fold.

Inputs are hand-built cycle calendars (tiny parquet files with an init_date column).
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.ml import pooled_training as pt

GAP = pd.Timedelta(days=pt.SPLIT_EMBARGO_DAYS)


def _year(tmp_path, year):
    p = tmp_path / f"p{year}.parquet"
    pd.DataFrame({"init_date": pd.date_range(f"{year}-01-01", f"{year}-12-31")}).to_parquet(
        p, index=False)
    return p


def _verified_days(cycles):
    return {c + pd.Timedelta(days=k) for c in cycles for k in range(10)}


def test_the_embargo_is_the_longest_lead():
    assert pt.SPLIT_EMBARGO_DAYS == 10       # Day 10 verifies init + 9


def test_no_observed_day_is_shared_between_train_val_and_test(tmp_path):
    cached = {y: _year(tmp_path, y) for y in (2014, 2015, 2016, 2017)}
    tr, va, te = pt.pooled_split(cached, test_year=2017)
    assert tr and va and te
    assert not _verified_days(tr) & _verified_days(va)
    assert not _verified_days(va) & _verified_days(te)
    assert not _verified_days(tr) & _verified_days(te)
    assert max(tr) + GAP <= min(va)


def test_the_embargo_costs_only_the_cycles_next_to_a_boundary(tmp_path):
    cached = {y: _year(tmp_path, y) for y in (2014, 2015, 2016, 2017)}
    tr, va, te = pt.pooled_split(cached, test_year=2017)
    pre = 365 + 365 + 366
    dropped = pre - len(tr) - len(va)
    assert 0 < dropped <= 2 * (pt.SPLIT_EMBARGO_DAYS - 1)


def test_folds_are_contiguous_blocks_of_days():
    cycles = set(pd.date_range("2000-01-01", "2002-12-31"))
    fold_of = pt.assign_folds(cycles)
    s = sorted(cycles)
    changes = sum(fold_of[a] != fold_of[b] for a, b in zip(s, s[1:]))
    assert changes <= len(s) / pt.FOLD_BLOCK_DAYS + 1, "folds must not alternate daily"
    assert set(fold_of.values()) == {0, 1, 2}


def test_a_fold_model_never_trains_on_a_cycle_sharing_an_observed_day_with_its_fold():
    cycles = set(pd.date_range("2000-01-01", "2002-12-31"))
    fold_of = pt.assign_folds(cycles)
    for fold in (0, 1, 2):
        held = {c for c, f in fold_of.items() if f == fold}
        train = pt.fold_training_cycles(cycles, fold_of, fold)
        assert train and not train & held
        assert not _verified_days(train) & _verified_days(held)


def test_oof_fold_models_fit_on_the_embargoed_cycles(monkeypatch, tmp_path):
    cycles = set(pd.date_range("2000-01-01", "2000-12-31"))
    fold_of = pt.assign_folds(cycles)
    seen = {}

    def fake_fit(cached, years, variable, chunks, cols, hbf, cache_dir, device, **_):
        seen[len(seen)] = set().union(*chunks)
        return None, None
    monkeypatch.setattr(pt, "_fit_booster", fake_fit)
    monkeypatch.setattr(pt, "_feature_columns_for", lambda cached, years, **_: ["x"])
    pt.oof_fold_models({}, [2000], "temperature_c", cycles, {}, fold_of, tmp_path)
    for fold, used in seen.items():
        held = {c for c, f in fold_of.items() if f == fold}
        assert not _verified_days(used) & _verified_days(held)
