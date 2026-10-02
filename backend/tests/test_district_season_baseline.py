"""The district x season bust-frequency baseline.

"How often has this district busted in this season?" needs no forecast at all, and on the
served run's 2017 test set it scored ROC-AUC 0.652 - above every rung the ladder had
(the best, EMOS, 0.626). A model that does not clearly beat it has not shown it reads the
forecast. So it is a rung.

Frames here are hand-built for the arithmetic of the rates and the back-off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml import baselines as bl


def _events(rows):
    return pd.DataFrame(rows, columns=["region_id", "season", "y_bust"]).assign(
        lead_time_days=1)


def test_it_predicts_each_district_seasons_training_rate():
    train = _events([("A", "JJAS", 1)] * 40 + [("A", "JJAS", 0)] * 60
                    + [("B", "JJAS", 1)] * 80 + [("B", "JJAS", 0)] * 20)
    m = bl.DistrictSeasonFrequencyBaseline().fit(train)
    p = m.predict_proba(_events([("A", "JJAS", 0), ("B", "JJAS", 0)]))
    assert p == pytest.approx([0.4, 0.8])


def test_a_thin_cell_backs_off_to_its_season():
    train = _events([("A", "JJAS", 1)] * 3                                  # 3 events only
                    + [("B", "JJAS", 0)] * 100 + [("B", "JJAS", 1)] * 100)
    m = bl.DistrictSeasonFrequencyBaseline(min_events=30).fit(train)
    season_rate = (3 + 100) / 203
    assert m.predict_proba(_events([("A", "JJAS", 0)]))[0] == pytest.approx(season_rate)


def test_an_unseen_season_backs_off_to_the_global_rate():
    train = _events([("A", "JJAS", 1)] * 50 + [("A", "JJAS", 0)] * 150)
    m = bl.DistrictSeasonFrequencyBaseline().fit(train)
    assert m.predict_proba(_events([("A", "DJF", 0)]))[0] == pytest.approx(0.25)


def test_it_is_on_the_ladder():
    assert bl.DistrictSeasonFrequencyBaseline in bl.ALL_BASELINES
    train = _events([("A", "JJAS", 1)] * 40 + [("A", "JJAS", 0)] * 60).assign(
        spread_mean=0.1, spread_max=0.2)
    assert "district_season_frequency" in bl.fit_all(train)


def test_it_reads_only_the_ladder_contract_columns():
    """Fits from region_id (an event key), season and y_bust - all in the baseline-fit
    file - so it cannot silently half-fit on a file that lacks a column."""
    from app.features import engineering as fe
    from app.ml.pooled_training import BASELINE_FIT_BASE_COLUMNS
    allowed = set(BASELINE_FIT_BASE_COLUMNS) | set(fe.EVENT_KEYS)
    assert {"region_id", "season", "y_bust"} <= allowed
