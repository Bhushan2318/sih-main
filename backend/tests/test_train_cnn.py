"""The CNN training harness, and the comparison it produces.

What is pinned here is mostly refusal: the ways this can produce a number that looks like
a result and is not one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="training-only dependency")

from app.ingestion import grid_fields as gf                       # noqa: E402
from app.ingestion.grid_fields import GridBundle                  # noqa: E402
from app.ml import train_cnn                                      # noqa: E402
from app.utils.india_districts import load_registry               # noqa: E402


def _bundle(init: str, n_var=3, leads=(1, 2)):
    lats, lons = gf.domain_coords()
    rng = np.random.default_rng(abs(hash(init)) % 2**32)
    vals = rng.normal(size=(n_var, len(leads), 2, len(lats), len(lons))).astype(np.float32)
    return GridBundle(init, tuple(f"v{i}" for i in range(n_var)), tuple(leads),
                      lats, lons, vals)


def _events(inits, region_ids, leads=(1, 2)):
    rows = []
    rng = np.random.default_rng(0)
    for init in inits:
        for lead in leads:
            for r in region_ids:
                rows.append({"region_id": r, "init_date": init, "lead_time_days": lead,
                             "y_bust": int(rng.random() > 0.5),
                             "bust_ratio": float(rng.random() * 2)})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def region_ids():
    return [d.region_id for d in load_registry()]


def test_no_grids_is_skipped_not_failed(tmp_path, region_ids):
    """Before the fetch runs there are no fields. That is a state to report, not a crash."""
    rep = train_cnn.train(tmp_path, pd.DataFrame(), {}, region_ids)
    assert rep.status == "skipped" and "no grid bundles" in rep.error


def test_too_few_cycles_is_refused(tmp_path, region_ids):
    """The failure this file exists to prevent: 17 cycles is 170 samples against ~115k
    parameters. A score from that measures the sample size, and would be quoted later as
    if it measured the model."""
    inits = [f"2019-01-{d:02d}" for d in range(1, 11)]
    for i in inits:
        gf.save_bundle(tmp_path / f"{i}.npz", _bundle(i))
    ev = _events(inits, region_ids[:5])
    rep = train_cnn.train(tmp_path, ev, {"train": inits, "val": [], "test": []}, region_ids)
    assert rep.status == "refused"
    assert rep.train_cycles == len(inits)
    assert "measures the sample size" in rep.error


def test_refusal_threshold_is_a_real_floor():
    assert train_cnn.MIN_TRAIN_CYCLES >= 100, \
        "a floor low enough to pass on a toy sample is not a floor"


# ------------------------------------------------------------------ alignment

def test_build_arrays_aligns_labels_to_districts(region_ids):
    inits = ["2019-07-17", "2019-07-18"]
    bundles = {i: _bundle(i) for i in inits}
    ev = _events(inits, region_ids[:4])
    X, EX, Y, AUX, CYC = train_cnn.build_arrays(bundles, ev, region_ids)
    assert X.shape[0] == len(inits) * 2          # cycles x lead days
    assert Y.shape == (len(inits) * 2, len(region_ids))
    labelled = ~np.isnan(Y)
    assert labelled.sum() == len(inits) * 2 * 4, "only the districts with events"
    assert np.isnan(Y[:, 100:]).all(), "unlabelled districts must stay NaN, not become 0"


def test_lead_day_reaches_the_model_as_a_feature(region_ids):
    inits = ["2019-07-17"]
    X, EX, Y, AUX, CYC = train_cnn.build_arrays(
        {i: _bundle(i) for i in inits}, _events(inits, region_ids[:2]), region_ids)
    assert EX.shape == (2, len(region_ids), 1)
    assert not np.allclose(EX[0], EX[1]), "lead 1 and lead 2 must differ"


def test_unknown_region_ids_are_dropped_not_crashed(region_ids):
    inits = ["2019-07-17"]
    ev = _events(inits, region_ids[:3])
    ev.loc[len(ev)] = {"region_id": "IN-XX-NOWHERE", "init_date": inits[0],
                       "lead_time_days": 1, "y_bust": 1, "bust_ratio": 1.0}
    out = train_cnn.build_arrays({i: _bundle(i) for i in inits}, ev, region_ids)
    assert out is not None


# ----------------------------------------------------------------- comparison

def test_comparison_names_a_winner_per_metric():
    cnn = train_cnn.CNNReport(status="success",
                              metrics={"test": {"roc_auc": 0.86, "brier": 0.14}})
    out = train_cnn.format_comparison(cnn, {"test": {"roc_auc": 0.84, "brier": 0.16}})
    assert "XGBoost" in out and "CNN" in out
    lines = {l.split()[0]: l for l in out.splitlines() if l.strip().startswith(("ROC", "Brier"))}
    assert lines["ROC-AUC"].strip().endswith("CNN")
    assert lines["Brier"].strip().endswith("CNN"), "lower Brier is better, not higher"


def test_comparison_gets_brier_direction_right():
    """A higher Brier is worse. Reading it like an accuracy would silently invert the
    result and hand the win to the wrong model."""
    cnn = train_cnn.CNNReport(status="success", metrics={"test": {"brier": 0.22}})
    out = train_cnn.format_comparison(cnn, {"test": {"brier": 0.16}})
    assert out.strip().splitlines()[-1].strip().endswith("XGBoost")


def test_comparison_survives_a_missing_metric():
    cnn = train_cnn.CNNReport(status="success", metrics={"test": {"roc_auc": 0.8}})
    assert "-" in train_cnn.format_comparison(cnn, {"test": {}})


def test_fit_one_builds_an_encoder_that_accepts_its_own_input(region_ids):
    """A shape bug that only appears at the first training step.

    BustCNN doubles in_channels internally, because forward concatenates the data with its
    mask. The training loop keeps those two arrays separate and passes the *data* channel
    count, so halving it there builds an encoder for half the channels it will be handed -
    and nothing fails until the first batch reaches the first convolution.
    """
    n_data = 6
    x = np.zeros((2, n_data, 145, 141), dtype=np.float32)
    m = np.ones_like(x)
    ex = np.zeros((2, 666, 1), dtype=np.float32)
    y = np.zeros((2, 666), dtype=np.float32)
    model, _ = train_cnn._fit_one(0, x, m, ex, y, y, x, m, ex, y,
                                  epochs=1, patience=1, lr=1e-3, region_ids=region_ids)
    assert model.encoder[0].in_channels == n_data * 2, \
        "the encoder must accept data and mask concatenated"
