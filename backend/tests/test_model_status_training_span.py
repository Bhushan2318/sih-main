"""The Model page's training span must come from whichever manifest shape the run wrote.

The single-year pipeline writes `split_cycles.train_dates`; the pooled 17-year run writes
`split_cycles.train_years` and `test_year` instead. Only the first was read, so the live
pooled model reported `first_train_date: null` and the page showed nothing but the serving
store's 2016+ range. The pooled manifest below is copied from the real
run_20260922T043925Z manifest.json, not invented.
"""
from __future__ import annotations

from app.api.routers.model_status import _training_data

POOLED_MANIFEST = {
    "run_id": "run_20260922T043925Z",
    "pooled_train_years": list(range(2000, 2017)),
    "test_year": 2017,
    "split_cycles": {
        "train": 5828, "val": 365, "test": 365,
        "train_years": list(range(2000, 2017)), "test_year": 2017,
        "fit": 2000, "classifier": 2000,
    },
    "paired_rows": 1298937012,
}


def test_pooled_manifest_reports_its_training_years():
    td = _training_data(POOLED_MANIFEST)
    assert td["first_train_year"] == 2000
    assert td["last_train_year"] == 2016
    assert td["test_year"] == 2017
    # Only years are recorded; a day would be invented, so none is reported.
    assert td["first_train_date"] is None


def test_single_year_manifest_still_reads_train_dates():
    td = _training_data({"split_cycles": {
        "train": 3, "val": 1, "test": 1,
        "train_dates": ["2017-01-04", "2017-02-11", "2017-03-09"],
    }})
    assert td["first_train_date"] == "2017-01-04"
    assert td["first_train_year"] == 2017
    assert td["last_train_year"] == 2017


def test_manifest_without_split_dates_says_nothing():
    td = _training_data({})
    assert td["first_train_date"] is None
    assert td["first_train_year"] is None
    assert td["test_year"] is None
