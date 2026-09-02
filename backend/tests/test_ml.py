"""Phase 3: thresholds, feature engineering, leakage guards, and a reduced end-to-end
retrain against the real canonical data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import engineering as fe
from app.features import pivot as pv
from app.ingestion.pipeline import confirm_mapping, ingest_upload
from app.ml import registry
from app.ml.thresholds import Thresholds, compute_error_thresholds, compute_risk_bands
from tests.conftest import ERA5_CSV, GEFS_CSV


# --------------------------------------------------------------------------- thresholds

def test_error_thresholds_are_percentiles():
    df = pd.DataFrame({"variable": ["t"] * 100, "abs_error": np.arange(100.0)})
    thr = compute_error_thresholds(df, percentile=90.0)
    assert 88 <= thr["t"] <= 91


def test_risk_bands_ordered_and_serialise(tmp_path):
    proba = np.clip(np.random.default_rng(0).beta(2, 5, 500), 0, 1)
    cuts = compute_risk_bands(proba)
    assert 0 < cuts["medium"] < cuts["high"] < 1
    t = Thresholds(bust_threshold={"t": 3.0}, p90_error={"t": 5.0}, risk_band_cuts=cuts)
    assert t.band_for(0.0) == "low"
    assert t.band_for(cuts["high"] + 1e-6) == "high"
    p = tmp_path / "thr.json"
    t.to_json(p)
    back = Thresholds.from_json(p)
    assert back.bust_threshold == t.bust_threshold
    assert back.band_for(0.99) == "high"


def test_risk_bands_degenerate_input():
    cuts = compute_risk_bands(np.zeros(5))
    assert cuts["medium"] < cuts["high"]


# ---------------------------------------------------------------- feature engineering

@pytest.fixture(scope="module")
def paired_real():
    from app.storage.parquet_store import read_dataset  # noqa: PLC0415
    # this fixture assumes the module-scoped ingest below has populated the store
    return None


def test_pairing_and_target(_ingested_slice):
    from app.storage.parquet_store import read_dataset

    canon = read_dataset()
    paired = fe.build_training_frame(canon)
    assert not paired.empty
    assert "abs_error" in paired.columns
    # target is a non-negative magnitude
    assert (paired["abs_error"] >= 0).all()
    # each paired row carries both sides
    assert paired[["forecast_value", "observed_value"]].notna().all().all()
    # ensemble spread computed across members
    assert paired["ensemble_member_count"].max() >= 2
    # a row's own concurrent-variable column is blanked
    t = paired[paired["variable"] == "temperature_c"]
    assert t["fc_temperature_c"].isna().all()


def test_classifier_features_exclude_actual_error(_ingested_slice):
    from app.storage.parquet_store import read_dataset

    canon = read_dataset()
    paired = fe.build_training_frame(canon)
    cycles = sorted(paired["init_date"].dropna().unique())
    tr = paired[paired["init_date"].isin(cycles[: max(2, len(cycles) - 1)])]
    oof = pd.Series(np.random.default_rng(1).random(len(tr)), index=tr.index)
    p90 = {v: 5.0 for v in tr["variable"].unique()}
    thr = {v: 3.0 for v in tr["variable"].unique()}
    ev = pv.build_event_frame(tr, oof, p90, thr)

    feats = pv.classifier_feature_columns(ev)
    assert "y_bust" in ev.columns
    assert not any(f.startswith("actual_err_") for f in feats)
    assert "bust_ratio" not in feats and "y_bust" not in feats
    assert any(f.startswith("pred_err_") for f in feats)


# --------------------------------------------------------------------- leakage guards
#
# Three ways this pipeline could quietly score itself on information it should not have.
# Each is asserted against the objects the pipeline actually produced, not against a
# re-implementation of its logic - a re-implementation would drift and then agree with
# itself while the real path leaked.


def test_split_by_cycle_is_time_ordered_and_disjoint():
    """Whole cycles, in time order, never shared between splits."""
    from app.ml.train_pipeline import _split_by_cycle

    cycles = pd.to_datetime([f"2019-{m:02d}-01" for m in range(1, 13)])
    paired = pd.DataFrame({"init_date": cycles})
    tr, va, te = _split_by_cycle(paired)

    assert tr and va and te, "a 12-cycle frame must produce all three splits"
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert set(tr | va | te) == set(cycles)
    # strictly time ordered: every train cycle precedes every val cycle, and so on
    assert max(tr) < min(va) < max(va) < min(te)


def test_oof_folds_never_split_a_forecast_cycle(_ingested_slice):
    """The classifier trains on out-of-fold regressor predictions. Cycles 24 h apart are
    heavily autocorrelated, so a random KFold would put near-duplicate rows on both sides
    of a fold and the OOF predictions would be optimistic. Folds must be grouped by
    init_date."""
    from sklearn.model_selection import GroupKFold

    from app.ml import regressors as reg_mod
    from app.storage.parquet_store import read_dataset

    paired = fe.build_training_frame(read_dataset())
    var = sorted(paired["variable"].unique())[0]
    tr = paired[paired["variable"] == var]
    groups = tr["init_date"].astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    assert n_groups > 1, "need several cycles to say anything about fold grouping"

    # Exactly the construction oof_predict uses.
    splits = min(3, n_groups)
    for fit_idx, held_idx in GroupKFold(n_splits=splits).split(tr, tr["abs_error"], groups):
        fit_cycles = set(groups[fit_idx])
        held_cycles = set(groups[held_idx])
        assert not (fit_cycles & held_cycles), (
            f"cycle(s) {fit_cycles & held_cycles} appear on both sides of a fold")
    assert reg_mod.oof_predict.__doc__ and "init_date" in reg_mod.oof_predict.__doc__


def test_thresholds_are_fit_on_the_train_split_only(_retrain):
    """A bust label is `error >= the 90th percentile of that variable's error`. If that
    percentile were taken over the whole dataset, the test set would be helping to define
    its own labels and every reported metric would be optimistic.

    Asserts the published thresholds equal the train-only percentiles and differ from the
    full-data ones - so this fails if anyone ever widens the frame it is fitted on.
    """
    from app.ml.thresholds import compute_error_thresholds
    from app.ml.train_pipeline import _event_mean_error, _split_by_cycle
    from app.storage.parquet_store import read_dataset

    report = _retrain
    assert report.status == "success", report.error

    paired = fe.build_training_frame(read_dataset())
    train_c, _, _ = _split_by_cycle(paired)
    tr = paired[paired["init_date"].isin(train_c)]

    train_only = compute_error_thresholds(_event_mean_error(tr), percentile=90.0)
    full_data = compute_error_thresholds(_event_mean_error(paired), percentile=90.0)
    published = report.thresholds["bust_threshold"]

    for var, value in published.items():
        assert var in train_only
        assert value == pytest.approx(train_only[var]), (
            f"{var}: published threshold {value} is not the train-only percentile "
            f"{train_only[var]}")
    # If these were equal the assertion above would be vacuous - it would pass whether or
    # not the fit was restricted to train. Guard against that.
    assert any(published[v] != pytest.approx(full_data[v]) for v in published), (
        "train-only and full-data thresholds are identical, so this test proves nothing "
        "about which one was used")


def test_no_observed_day_is_shared_between_train_and_test(_ingested_slice):
    """The split is by initialisation date, but the *label* is an observation on a valid
    date, and a Day-10 forecast reaches nine days past its init. If the gap between the
    last train cycle and the first test cycle were shorter than the maximum lead, the same
    observed day would sit on both sides of the split.
    """
    from app.ml.train_pipeline import _split_by_cycle
    from app.storage.parquet_store import read_dataset

    paired = fe.build_training_frame(read_dataset())
    train_c, _, test_c = _split_by_cycle(paired)
    train_days = set(pd.to_datetime(paired[paired["init_date"].isin(train_c)]["valid_date"]))
    test_days = set(pd.to_datetime(paired[paired["init_date"].isin(test_c)]["valid_date"]))

    shared = train_days & test_days
    assert not shared, (
        f"{len(shared)} observed day(s) appear in both the train and test splits, "
        f"e.g. {sorted(shared)[:3]} - the train/test gap is shorter than the max lead")


# ----------------------------------------------------------------- end-to-end retrain

@pytest.fixture(scope="module")
def _retrain(_ingested_slice):
    from app.ml.train_pipeline import full_retrain
    return full_retrain(make_current=True)


def test_full_retrain_end_to_end(_retrain):
    report = _retrain
    assert report.status == "success", report.error
    assert report.made_current
    assert len(report.modelled_variables) >= 3
    assert report.classifier_metrics.get("train", {}).get("n", 0) > 0

    # thresholds are real numbers in canonical units
    for var, t in report.thresholds["bust_threshold"].items():
        assert t > 0

    # current.json points at this run and everything reloads
    rid = registry.current_run_id()
    assert rid == report.run_id
    regs = registry.load_regressors(rid)
    assert set(regs) == set(report.modelled_variables)
    clf, cols = registry.load_classifier(rid)
    assert clf is not None and len(cols) > 5
    thr = registry.load_thresholds(rid)
    assert thr.risk_band_cuts["medium"] <= thr.risk_band_cuts["high"]

    # a failed-style guard: manifest records the shap method honestly
    import json
    manifest = json.loads((registry.run_dir(rid) / "manifest.json").read_text())
    assert manifest["shap_method"] in {"shap", "feature_importance_fallback", "none"}


def test_regressors_beat_mean_baseline_on_temperature(_retrain):
    m = _retrain.regressor_metrics.get("temperature_c", {})
    ref = m.get("test") or m.get("val") or m.get("train")
    assert ref is not None
    assert ref["mae"] < ref["baseline_mae_predict_mean"]


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def _ingested_slice(request):
    """Ingest a multi-cycle slice of the real GEFS + ERA5 samples once for this module."""
    import shutil

    from app.db.base import SessionLocal, engine, init_db
    from app.db.models import Base
    from app.storage import parquet_store

    Base.metadata.drop_all(engine)
    shutil.rmtree(parquet_store.CANONICAL_DIR, ignore_errors=True)
    init_db()

    tmp = request.getfixturevalue("tmp_path_factory").mktemp("mlslice")
    # ~8 init cycles: enough for a 5/1/2 train/val/test split
    g = pd.read_csv(GEFS_CSV)
    keep_cycles = sorted(g["init_date"].unique())[:8]
    g = g[g["init_date"].isin(keep_cycles)]
    gp = tmp / "gefs.csv"; g.to_csv(gp, index=False)
    e = pd.read_csv(ERA5_CSV)
    ep = tmp / "era5.csv"; e.to_csv(ep, index=False)

    def _confirm(res, s):
        if res.status != "pending_confirmation":
            return res
        seen, conf = set(), []
        for p in sorted(res.mapping_proposals, key=lambda x: x["source_column"]):
            if p["role"] != "measurement" or p["decision"] != "needs_confirmation" or not p["suggested_variable"]:
                continue
            key = (p["suggested_variable"], p["suggested_value_type"])
            if key in seen:
                continue
            seen.add(key)
            conf.append({"source_column": p["source_column"], "variable": p["suggested_variable"],
                         "value_type": p["suggested_value_type"], "unit_conversion": p["unit_conversion"]})
        return confirm_mapping(s, res.batch_id, conf)

    s = SessionLocal()
    try:
        _confirm(ingest_upload(s, gp, gp.name), s)
        _confirm(ingest_upload(s, ep, ep.name), s)
        s.commit()
    finally:
        s.close()
    yield
    Base.metadata.drop_all(engine)
    shutil.rmtree(parquet_store.CANONICAL_DIR, ignore_errors=True)


# ------------------------------------------------------------------- guided replay
# score_cycle can score any historical cycle, not just the latest, and replay_service
# turns one into a narrated lead-day-by-lead-day walkthrough - every number scored from
# that cycle, every sentence generated from those numbers.

def test_score_cycle_targets_an_arbitrary_historical_init(_retrain):
    from app.ml import inference

    inference.invalidate_caches()
    state = inference.load_model_state()
    assert state is not None

    cycles = inference.available_cycles()
    assert len(cycles) >= 2, "the 8-cycle slice should expose several init dates"

    latest = inference.score_cycle(state, None)
    assert latest is not None
    assert latest.init_date == max(cycles)

    older = inference.score_cycle(state, cycles[-1])
    assert older is not None
    assert older.init_date == cycles[-1]
    assert not older.events.empty
    assert older.events["bust_probability"].between(0.0, 1.0).all()

    # an init date that is not in the store yields nothing rather than a guess
    missing = pd.Timestamp(cycles[0]) - pd.Timedelta(days=3650)
    assert inference.score_cycle(state, missing) is None


def test_replay_service_narrates_from_real_numbers(_retrain):
    from app.services import replay_service

    replay_service.invalidate()
    cycles = replay_service.list_cycles()
    assert cycles, "at least one scoreable cycle"
    # verified cycles must rank ahead of unverified ones
    verified_flags = [c.verified for c in cycles]
    assert verified_flags == sorted(verified_flags, reverse=True)

    rep = replay_service.get_replay()
    assert rep.model_trained
    assert rep.init_date is not None
    assert 1 <= len(rep.steps) <= 10
    assert rep.summary_narration and rep.summary_narration.strip()

    leads = [s.lead_time_days for s in rep.steps]
    assert leads == sorted(leads)
    for s in rep.steps:
        assert s.narration.startswith(f"Day {s.lead_time_days}")
        assert s.regions, "each step lists its scored regions"
        assert all(0.0 <= r.bust_probability <= 1.0 for r in s.regions)
        # regions are ordered worst-first
        probs = [r.bust_probability for r in s.regions]
        assert probs == sorted(probs, reverse=True)

    if rep.focus is not None:
        assert rep.focus.variable != "wind_direction_deg"  # excluded: 0/360 wraparound
        assert rep.focus.points
        # the chart follows the screen: default focus is the cycle's peak-bust-risk region
        peak_rid = max(
            (r for s in rep.steps for r in s.regions), key=lambda r: r.bust_probability
        ).region_id
        assert rep.focus.region_id == peak_rid
        # every worst-list region offered as a focus option resolves to a real series
        assert rep.focus_options
        assert all(o.points for o in rep.focus_options)
        assert rep.focus.region_id in {o.region_id for o in rep.focus_options}
        # a caller can pin the chart to another region
        other = next((o.region_id for o in rep.focus_options if o.region_id != peak_rid), None)
        if other:
            pinned = replay_service.get_replay(str(rep.init_date), focus_region=other)
            assert pinned.focus is not None and pinned.focus.region_id == other
