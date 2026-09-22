"""The cases the model got most wrong, picked from its own held-out rows.

Every other panel in this project argues that the model works. This one exists to show
where it does not, which is only worth showing if it is picked honestly: the worst cases
by confidence, not a convenient sample.

Fixtures here are hand-built frames with hand-checked expected answers, per CLAUDE.md -
a toy frame cannot express the volume bugs this repo keeps finding, but ranking and
tie-breaking is exactly the kind of logic a toy frame *can* pin.
"""
import pandas as pd

from app.ml.misses import worst_misses


def _events():
    """Six held-out rows, plus one row from another split that must be ignored."""
    return pd.DataFrame({
        "region_id": ["IN-A-1", "IN-A-2", "IN-A-3", "IN-B-1", "IN-B-2", "IN-B-3", "IN-C-9"],
        "valid_date": pd.to_datetime(
            ["2018-01-01"] * 6 + ["2018-02-02"]).date.tolist(),
        "lead_time_days": [1, 2, 3, 4, 5, 6, 7],
        "y_bust": [1, 1, 1, 0, 0, 0, 1],
        # Missed busts: it happened, and the model said it would not.
        # False alarms: it did not happen, and the model said it would.
        "model_proba": [0.05, 0.20, 0.44, 0.95, 0.70, 0.51, 0.01],
        "split": ["test"] * 6 + ["val"],
        "actual_err_temperature_c": [9.0, 1.0, 1.0, 0.1, 0.1, 0.1, 99.0],
        "actual_err_rainfall_mm": [2.0, 40.0, 2.0, 0.1, 0.1, 0.1, 1.0],
    })


THR = {"temperature_c": 4.45, "rainfall_mm": 13.58}


def test_missed_busts_are_ranked_by_how_wrong_the_model_was():
    out = worst_misses(_events(), THR, split="test", k=2)
    got = [(m["region_id"], m["bust_probability"]) for m in out["missed_busts"]]
    # It busted in all three, and 0.05 is the most confidently wrong.
    assert got == [("IN-A-1", 0.05), ("IN-A-2", 0.20)]


def test_false_alarms_are_ranked_the_other_way():
    out = worst_misses(_events(), THR, split="test", k=2)
    got = [(m["region_id"], m["bust_probability"]) for m in out["false_alarms"]]
    assert got == [("IN-B-1", 0.95), ("IN-B-2", 0.70)]


def test_a_missed_bust_names_the_variable_that_actually_busted():
    out = worst_misses(_events(), THR, split="test", k=3)
    by_region = {m["region_id"]: m for m in out["missed_busts"]}
    # 9.0 against a 4.45 threshold is 2.02x; rainfall's 2.0 against 13.58 is below it.
    assert by_region["IN-A-1"]["variable"] == "temperature_c"
    assert by_region["IN-A-1"]["actual_error"] == 9.0
    assert by_region["IN-A-1"]["threshold"] == 4.45
    # The second row busted on rainfall instead, so the panel must not say temperature.
    assert by_region["IN-A-2"]["variable"] == "rainfall_mm"


def test_only_the_named_split_is_used():
    out = worst_misses(_events(), THR, split="test", k=10)
    regions = {m["region_id"] for m in out["missed_busts"]}
    # IN-C-9 is a val row: a far worse miss, and it must not appear.
    assert "IN-C-9" not in regions


def test_k_caps_each_list():
    out = worst_misses(_events(), THR, split="test", k=1)
    assert len(out["missed_busts"]) == 1
    assert len(out["false_alarms"]) == 1


def test_an_empty_frame_gives_empty_lists_rather_than_raising():
    empty = _events().iloc[0:0]
    out = worst_misses(empty, THR, split="test", k=3)
    assert out["missed_busts"] == []
    assert out["false_alarms"] == []


def test_a_row_with_no_variable_over_threshold_reports_no_variable():
    """y_bust can be 1 while every actual_err_ column is NaN - the label is built from
    the paired frame, and a column can be missing for that district and lead."""
    ev = _events().head(1).copy()
    ev["actual_err_temperature_c"] = float("nan")
    ev["actual_err_rainfall_mm"] = float("nan")
    out = worst_misses(ev, THR, split="test", k=1)
    assert out["missed_busts"][0]["variable"] is None
    assert out["missed_busts"][0]["actual_error"] is None
