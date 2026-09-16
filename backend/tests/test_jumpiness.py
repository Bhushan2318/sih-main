"""C1 - forecast jumpiness: how much the forecast for one valid date moved between
consecutive initialisations.

Every expected value below is computed by hand in the comment beside it, not by
re-running the code under test. The input numbers are small, invented and labelled as
such: these tests pin arithmetic and causality, and nothing here reaches a metric.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.features import engineering as fe

VALID = pd.Timestamp("2017-11-10")


def _fc_rows(region, variable, valid, init_to_members):
    """Canonical forecast rows. `init_to_members` maps init date -> member values.
    ARITHMETIC FIXTURE - invented numbers, never used to produce a metric."""
    rows = []
    for init, members in init_to_members.items():
        init = pd.Timestamp(init)
        lead = (pd.Timestamp(valid) - init).days + 1          # valid = init + (lead - 1)
        for i, v in enumerate(members):
            rows.append({
                "region_id": region, "variable": variable, "value_type": "forecast",
                "valid_date": pd.Timestamp(valid), "init_date": init,
                "lead_time_days": lead, "ensemble_member_id": f"m{i}", "value": v,
            })
    return pd.DataFrame(rows)


def _jump(fc, **kw):
    out = fe.compute_jumpiness(fe.forecast_trajectories(fc), **kw)
    return out.set_index("init_date").sort_index()


# Ensemble means 30, 32, 31, 35 on inits 6, 7, 8, 9 November, all for 10 November.
TEMPS = {"2017-11-06": [29, 31], "2017-11-07": [31, 33],
         "2017-11-08": [30, 32], "2017-11-09": [34, 36]}


def test_ensemble_mean_is_the_trajectory_value():
    traj = fe.forecast_trajectories(_fc_rows("A", "temperature_c", VALID, TEMPS))
    got = traj.set_index("init_date")["fc_mean"].sort_index().tolist()
    assert got == [30.0, 32.0, 31.0, 35.0]


def test_absolute_change_between_the_two_most_recent_cycles():
    j = _jump(_fc_rows("A", "temperature_c", VALID, TEMPS))
    # 9 Nov: |35 - 31| = 4.   8 Nov: |31 - 32| = 1.   7 Nov: |32 - 30| = 2.
    assert j.loc["2017-11-09", "jump_abs_change"] == pytest.approx(4.0)
    assert j.loc["2017-11-08", "jump_abs_change"] == pytest.approx(1.0)
    assert j.loc["2017-11-07", "jump_abs_change"] == pytest.approx(2.0)


def test_the_first_cycle_has_no_jump_rather_than_a_zero_one():
    """Missing never becomes zero (CLAUDE.md rule 3)."""
    j = _jump(_fc_rows("A", "temperature_c", VALID, TEMPS))
    first = j.loc["2017-11-06"]
    assert math.isnan(first["jump_abs_change"])
    assert math.isnan(first["jump_std"])
    assert math.isnan(first["jump_sign_flips"])


def test_std_across_the_last_k_cycles():
    j = _jump(_fc_rows("A", "temperature_c", VALID, TEMPS), window=5)
    # 9 Nov, window holds 35, 31, 32, 30: mean 32, deviations 3, -1, 0, -2,
    # squares sum 14, /(4-1) = 4.6667, sqrt = 2.160247.
    assert j.loc["2017-11-09", "jump_std"] == pytest.approx(math.sqrt(14 / 3))
    # 8 Nov, window holds 31, 32, 30: mean 31, squares 0+1+1 = 2, /2 = 1, sqrt = 1.
    assert j.loc["2017-11-08", "jump_std"] == pytest.approx(1.0)
    # 7 Nov has only two cycles - too few for a spread that is not just |change|/sqrt2.
    assert math.isnan(j.loc["2017-11-07", "jump_std"])


def test_the_window_is_honoured():
    j = _jump(_fc_rows("A", "temperature_c", VALID, TEMPS), window=3)
    # 9 Nov, window of 3 holds 35, 31, 32: mean 32.6667, deviations 2.3333, -1.6667,
    # -0.6667, squares 5.4444 + 2.7778 + 0.4444 = 8.6667, /2 = 4.3333, sqrt = 2.081666.
    assert j.loc["2017-11-09", "jump_std"] == pytest.approx(math.sqrt(26 / 6))
    # changes inside that window: +4 (31->35), -1 (32->31): one flip.
    assert j.loc["2017-11-09", "jump_sign_flips"] == 1


def test_sign_flip_count():
    j = _jump(_fc_rows("A", "temperature_c", VALID, TEMPS), window=5)
    # chronological 30 -> 32 -> 31 -> 35: changes +2, -1, +4: two reversals.
    assert j.loc["2017-11-09", "jump_sign_flips"] == 2
    # 30 -> 32 -> 31: +2, -1: one reversal.
    assert j.loc["2017-11-08", "jump_sign_flips"] == 1


def test_an_unchanged_step_neither_counts_nor_hides_a_reversal():
    # chronological 0 -> 1 -> 1 -> 0 (dry, a shower appears, holds, disappears):
    # changes +1, 0, -1. The forecast went up and came back down: one reversal.
    rain = {"2017-11-06": [0, 0], "2017-11-07": [1, 1],
            "2017-11-08": [1, 1], "2017-11-09": [0, 0]}
    j = _jump(_fc_rows("A", "rainfall_mm", VALID, rain))
    assert j.loc["2017-11-09", "jump_sign_flips"] == 1
    # 0 -> 1 -> 1: +1, 0: nothing reversed.
    assert j.loc["2017-11-08", "jump_sign_flips"] == 0


def test_wind_direction_is_circular():
    # Members 340 and 0 average to 350 as vectors (arithmetic mean would say 170).
    # chronological 350 -> 10 -> 350: each step is 20 degrees across north, not 340.
    wind = {"2017-11-07": [340, 0], "2017-11-08": [0, 20], "2017-11-09": [340, 0]}
    fc = _fc_rows("A", "wind_direction_deg", VALID, wind)
    traj = fe.forecast_trajectories(fc).set_index("init_date")["fc_mean"].sort_index()
    assert traj.tolist() == pytest.approx([350.0, 10.0, 350.0], abs=1e-9)

    j = _jump(fc)
    assert j.loc["2017-11-09", "jump_abs_change"] == pytest.approx(20.0)
    # unwrapped 350, 370, 350: mean 356.667, squares 44.444 + 177.778 + 44.444 = 266.667,
    # /2 = 133.333, sqrt = 11.547005.
    assert j.loc["2017-11-09", "jump_std"] == pytest.approx(math.sqrt(400 / 3))
    assert j.loc["2017-11-09", "jump_sign_flips"] == 1


def test_a_later_cycle_never_changes_an_earlier_row():
    """Causality: a forecast issued on 8 Nov can only know cycles issued by 8 Nov."""
    before = _jump(_fc_rows("A", "temperature_c", VALID, TEMPS))
    later = dict(TEMPS, **{"2017-11-10": [0, 0]})
    after = _jump(_fc_rows("A", "temperature_c", VALID, later))
    cols = ["jump_abs_change", "jump_std", "jump_sign_flips"]
    pd.testing.assert_frame_equal(before[cols], after.loc[before.index, cols])


def test_regions_variables_and_valid_dates_do_not_mix():
    a = _fc_rows("A", "temperature_c", VALID, TEMPS)
    b = _fc_rows("B", "temperature_c", VALID, {"2017-11-08": [100, 100],
                                              "2017-11-09": [0, 0]})
    c = _fc_rows("A", "humidity_pct", VALID, {"2017-11-09": [50, 50]})
    d = _fc_rows("A", "temperature_c", VALID + pd.Timedelta(days=1),
                 {"2017-11-09": [90, 90]})
    out = fe.compute_jumpiness(fe.forecast_trajectories(pd.concat([a, b, c, d])))
    key = ["region_id", "variable", "valid_date", "init_date"]
    out = out.set_index(key)
    init = pd.Timestamp("2017-11-09")
    assert out.loc[("A", "temperature_c", VALID, init), "jump_abs_change"] == pytest.approx(4)
    assert out.loc[("B", "temperature_c", VALID, init), "jump_abs_change"] == pytest.approx(100)
    assert math.isnan(out.loc[("A", "humidity_pct", VALID, init), "jump_abs_change"])
    assert math.isnan(out.loc[("A", "temperature_c", VALID + pd.Timedelta(days=1), init),
                              "jump_abs_change"])


def test_climatology_is_the_mean_absolute_change_per_district_and_variable():
    j = _jump(_fc_rows("A", "temperature_c", VALID, TEMPS)).reset_index()
    j["region_id"], j["variable"] = "A", "temperature_c"
    clim = fe.compute_jump_climatology(j)
    # changes 2, 1, 4 (the first cycle has none and is ignored): mean 7/3.
    assert clim == {("A", "temperature_c"): pytest.approx(7 / 3)}

    rel = fe.attach_jump_climatology(j.copy(), clim).set_index("init_date")
    # 9 Nov: 4 / (7/3) = 12/7 = 1.714286.
    assert rel.loc["2017-11-09", "jump_rel_climatology"] == pytest.approx(12 / 7)
    assert math.isnan(rel.loc["2017-11-06", "jump_rel_climatology"])


def test_a_zero_climatology_is_not_divided_by():
    """A district where the forecast never moved (e.g. dry-season rainfall) has no
    climatological jumpiness to be relative to. That is unknown, not infinite."""
    df = pd.DataFrame({"region_id": ["A", "A"], "variable": ["rainfall_mm"] * 2,
                       "jump_abs_change": [0.0, 0.0]})
    clim = fe.compute_jump_climatology(df)
    out = fe.attach_jump_climatology(df.copy(), clim)
    assert out["jump_rel_climatology"].isna().all()


def test_an_unknown_district_gets_no_relative_jump():
    df = pd.DataFrame({"region_id": ["Z"], "variable": ["temperature_c"],
                       "jump_abs_change": [3.0]})
    out = fe.attach_jump_climatology(df, {("A", "temperature_c"): 1.0})
    assert out["jump_rel_climatology"].isna().all()


# ----------------------------------------------------------- wired into the frames

def _canonical(fc: pd.DataFrame) -> pd.DataFrame:
    """Forecast rows plus an observation for every valid date they cover.
    PLUMBING FIXTURE - invented numbers, never used to produce a metric."""
    ob = (fc[["region_id", "variable", "valid_date"]].drop_duplicates()
          .assign(value_type="observed", value=31.0, init_date=pd.NaT,
                  lead_time_days=np.nan, ensemble_member_id=None))
    return pd.concat([fc, ob], ignore_index=True)


def test_the_training_frame_carries_the_jump_features_on_every_member_row():
    frame = fe.build_training_frame(_canonical(_fc_rows("A", "temperature_c", VALID, TEMPS)))
    for col in fe.JUMP_FEATURES:
        assert col in frame.columns
    last = frame[frame["init_date"] == pd.Timestamp("2017-11-09")]
    assert len(last) == 2                       # both members
    assert last["jump_abs_change"].tolist() == pytest.approx([4.0, 4.0])
    # no climatology supplied -> relative jump unknown, not zero
    assert frame["jump_rel_climatology"].isna().all()


def test_history_outside_the_frame_feeds_the_jump_but_is_not_paired():
    """The chunked trainer and the one-cycle scorer both see only part of the archive at
    once; earlier cycles arrive as `forecast_history`. They inform the jump and add no
    rows of their own."""
    full = _fc_rows("A", "temperature_c", VALID, TEMPS)
    current = full[full["init_date"] == pd.Timestamp("2017-11-09")]
    history = full[full["init_date"] < pd.Timestamp("2017-11-09")]

    frame = fe.build_training_frame(_canonical(current), forecast_history=history)
    assert set(frame["init_date"]) == {pd.Timestamp("2017-11-09")}
    assert frame["jump_abs_change"].iloc[0] == pytest.approx(4.0)
    assert frame["jump_std"].iloc[0] == pytest.approx(math.sqrt(14 / 3))
    assert frame["jump_sign_flips"].iloc[0] == 2


def test_regressor_and_classifier_both_see_the_features():
    from app.features import pivot as pv
    from app.ml import regressors as reg_mod

    frame = fe.build_training_frame(_canonical(_fc_rows("A", "temperature_c", VALID, TEMPS)))
    for col in fe.JUMP_FEATURES:
        assert col in reg_mod.feature_columns(frame)

    pred = pd.Series(1.0, index=frame.index)
    ev = pv.build_event_frame(frame, pred, {"temperature_c": 5.0}, {"temperature_c": 3.0})
    row = ev.set_index("init_date").loc[pd.Timestamp("2017-11-09")]
    assert row["jump_abs_change_temperature_c"] == pytest.approx(4.0)
    feats = pv.classifier_feature_columns(ev)
    assert "jump_abs_change_temperature_c" in feats
    assert "jump_sign_flips_temperature_c" in feats


# ------------------------------------------------ across chunks, splits and serving

def _store_rows(inits, leads=3, members=2, region="A", variable="temperature_c"):
    """Canonical rows for consecutive daily cycles, plus an observation per valid date.
    PLUMBING FIXTURE - invented numbers, never used to produce a metric. The value moves
    with both init and lead so every cycle disagrees with the one before it."""
    rows, valid_dates = [], set()
    for c, init in enumerate(inits):
        init = pd.Timestamp(init)
        for lead in range(1, leads + 1):
            valid = init + pd.Timedelta(days=lead - 1)
            valid_dates.add(valid)
            for m in range(members):
                rows.append({"region_id": region, "variable": variable,
                             "value_type": "forecast", "init_date": init,
                             "valid_date": valid, "lead_time_days": lead,
                             "ensemble_member_id": f"m{m}",
                             "value": 20.0 + ((-1) ** c) * c + lead + m})
    for valid in sorted(valid_dates):
        rows.append({"region_id": region, "variable": variable, "value_type": "observed",
                     "valid_date": valid, "value": 21.0,
                     "verification_status": "final"})
    return pd.DataFrame(rows)


def test_chunked_training_frame_matches_one_pass(fresh_store, monkeypatch):
    """_build_paired_in_chunks sees 12 cycles at a time. Without history from the chunk
    before, the first cycle of every chunk would lose its jump - silently, looking like a
    cycle that simply had no predecessor."""
    from app.storage import parquet_store
    from app.ml import train_pipeline as tp

    inits = pd.date_range("2017-11-01", periods=5, freq="D")
    parquet_store.append_batch("jump-plumbing", _store_rows(inits))

    one_pass = fe.build_training_frame(
        parquet_store.read_dataset(columns=tp._TRAINING_COLUMNS))
    monkeypatch.setattr(tp, "_CHUNK_CYCLES", 1)
    chunked, _ = tp._build_paired_in_chunks()

    key = ["init_date", "valid_date", "ensemble_member_id"]
    cols = ["jump_abs_change", "jump_std", "jump_sign_flips"]
    a = one_pass.sort_values(key)[key + cols].reset_index(drop=True)
    b = chunked.sort_values(key)[key + cols].reset_index(drop=True)
    b["ensemble_member_id"] = b["ensemble_member_id"].astype(str)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, atol=1e-5)
    assert b["jump_std"].notna().any(), "vacuous: no row had three cycles"


def test_history_reads_only_earlier_cycles_that_reach_the_target_dates(fresh_store):
    from app.storage import parquet_store
    from app.features.history import forecast_history

    target = pd.Timestamp("2017-11-20")
    inits = [target - pd.Timedelta(days=d) for d in (12, 9, 1)] + \
            [target, target + pd.Timedelta(days=1)]
    parquet_store.append_batch("jump-plumbing", _store_rows(inits, leads=10))

    hist = forecast_history(target, [c.date() for c in inits])
    got = set(pd.to_datetime(hist["init_date"]))
    # 12 days back cannot reach the target at all; the target itself is the cycle being
    # scored, not history; a later cycle does not exist at issue time.
    assert got == {target - pd.Timedelta(days=9), target - pd.Timedelta(days=1)}
    assert (pd.to_datetime(hist["valid_date"]) >= target).all()
    # Reduced to one ensemble mean per trajectory as each cycle is read, never held as
    # member rows: 2 members -> 1 row. With ten leads, cycle target-9 reaches only the
    # target day (its Day 10) and target-1 reaches target..target+8 (its Days 2-10).
    assert "fc_mean" in hist.columns
    assert not hist.duplicated(["region_id", "variable", "valid_date", "init_date"]).any()
    assert len(hist) == 1 + 9


def test_the_climatology_is_fitted_on_training_rows_only():
    from app.ml import train_pipeline as tp

    def split(values):
        return pd.DataFrame({"region_id": "A", "variable": "temperature_c",
                             "jump_abs_change": values})

    tr, va = split([1.0, 3.0]), split([100.0, np.nan])
    clim = tp._fit_jump_climatology(tr, (va,))
    assert clim == {("A", "temperature_c"): pytest.approx(2.0)}      # (1 + 3) / 2
    assert tr["jump_rel_climatology"].tolist() == pytest.approx([0.5, 1.5])
    assert va["jump_rel_climatology"].iloc[0] == pytest.approx(50.0)  # 100 / 2
    assert math.isnan(va["jump_rel_climatology"].iloc[1])


def test_the_climatology_round_trips_through_the_registry():
    from app.ml import registry

    clim = {("IN-KL-IDUKKI", "rainfall_mm"): 3.25, ("IN-DL", "wind_direction_deg"): 11.0}
    registry.save_jump_climatology("run_jump_roundtrip", clim)
    assert registry.load_jump_climatology("run_jump_roundtrip") == clim
    assert registry.load_jump_climatology("run_that_never_existed") == {}


def test_scoring_hands_history_and_the_runs_climatology_to_the_features(fresh_store,
                                                                         monkeypatch):
    """Training saw jumps built from earlier cycles and scaled by a training-split
    climatology; a scorer that passed neither would serve NaN where the model learned a
    value - skew that no test on the trainer alone can see."""
    from app.storage import parquet_store
    from app.ml import inference
    from app.ml.thresholds import Thresholds

    target = pd.Timestamp("2017-11-20")
    inits = [target - pd.Timedelta(days=2), target - pd.Timedelta(days=1), target]
    parquet_store.append_batch("jump-plumbing", _store_rows(inits, leads=3))

    seen = {}
    real = fe.build_training_frame

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(inference.fe, "build_training_frame", spy)
    clim = {("A", "temperature_c"): 2.0}
    state = inference.ModelState(
        run_id="run_jump_spy", regressors={}, classifier=None, classifier_columns=[],
        thresholds=Thresholds(bust_threshold={}, p90_error={}, risk_band_cuts={}),
        historical_bust_freq={}, jump_climatology=clim)
    inference.score_cycle(state, init_date=target)

    assert seen["jump_climatology"] == clim
    hist = seen["forecast_history"]
    assert set(pd.to_datetime(hist["init_date"])) == set(inits[:2])
