"""The CNN training harness, and the comparison it produces.

What is pinned here is mostly refusal: the ways this can produce a number that looks like
a result and is not one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="training-only dependency")

from app.ingestion import grid_fields as gf                       # noqa: E402
from app.ingestion.grid_fields import GridBundle                  # noqa: E402
from app.ml import train_cnn                                      # noqa: E402
from app.ml.cnn import BustCNN                                    # noqa: E402
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


# --- where the CLI gets its splits ------------------------------------------------------
# Plumbing tests: the frames below are hand-built rows, not weather. They check that the
# cycle lists handed to train() are the ones the tabular run scored, in the exact string
# form the grid bundles are keyed by.

def _eval_frame(split_of: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"region_id": "IN-KL-IDUKKI", "init_date": pd.Timestamp(d), "lead_time_days": 1,
         "y_bust": 0, "split": s}
        for d, s in split_of.items()
    ])


def test_splits_come_from_the_tabular_runs_own_split():
    """The CLI used to pass {} as splits, so every run found 0 training cycles and was
    refused. The splits must be the cycles the tabular model was scored on, or the two
    families are not being compared on identical rows."""
    ev = _eval_frame({"2017-01-01": "train", "2017-01-02": "train",
                      "2017-06-01": "val", "2017-11-29": "test"})
    s = train_cnn.splits_from_eval(ev)
    assert s == {"train": ["2017-01-01", "2017-01-02"], "val": ["2017-06-01"],
                 "test": ["2017-11-29"]}


def test_split_dates_match_the_bundle_key_not_a_timestamp_string():
    """Bundles are keyed by path.stem, '2017-01-01'. str() of a pandas Timestamp is
    '2017-01-01 00:00:00', which matches nothing - and train() would refuse for too few
    cycles rather than say the keys never lined up."""
    s = train_cnn.splits_from_eval(_eval_frame({"2017-03-04": "test"}))
    assert s["test"] == ["2017-03-04"]


def test_a_cycle_in_two_splits_is_refused():
    """The tabular split is by cycle. A cycle in both train and test is leakage, and a
    score computed across it is not a held-out score."""
    ev = pd.concat([_eval_frame({"2017-01-01": "train"}),
                    _eval_frame({"2017-01-01": "test"})], ignore_index=True)
    with pytest.raises(ValueError, match="2017-01-01"):
        train_cnn.splits_from_eval(ev)


# --- train() must stream --------------------------------------------------------------
# Plumbing tests on small random bundles, clearly not weather and never a metric. What they
# pin is structural: the CLI's train() used load_bundles + build_arrays, which decode every
# bundle up front - 196 MB each, ~71 GB for 365 cycles - although the streaming path
# (build_index / fit_streaming / predict_streaming) already existed and was tested.

def _forbid_materialising(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("train() materialised the year instead of streaming it")
    monkeypatch.setattr(train_cnn, "load_bundles", boom)
    monkeypatch.setattr(train_cnn, "build_arrays", boom)


def test_train_never_loads_every_bundle(tmp_path, region_ids, monkeypatch):
    _forbid_materialising(monkeypatch)
    inits = [f"2019-01-{d:02d}" for d in range(1, 11)]
    for i in inits:
        gf.save_bundle(tmp_path / f"{i}.npz", _bundle(i))
    rep = train_cnn.train(tmp_path, _events(inits, region_ids[:5]),
                          {"train": inits, "val": [], "test": []}, region_ids)
    assert rep.status == "refused" and rep.train_cycles == len(inits)


def test_streamed_train_reaches_a_scored_report(tmp_path, region_ids, monkeypatch):
    _forbid_materialising(monkeypatch)
    monkeypatch.setattr(train_cnn, "MIN_TRAIN_CYCLES", 4)
    inits = [f"2019-02-{d:02d}" for d in range(1, 9)]
    for i in inits:
        gf.save_bundle(tmp_path / f"{i}.npz", _bundle(i))
    splits = {"train": inits[:4], "val": inits[4:6], "test": inits[6:]}
    rep = train_cnn.train(tmp_path, _events(inits, region_ids[:6]), splits, region_ids,
                          seeds=1, epochs=1, patience=1)
    assert rep.status == "success", rep.error
    assert (rep.train_cycles, rep.val_cycles, rep.test_cycles) == (4, 2, 2)
    assert rep.n_train_samples == 8, "4 cycles x 2 lead days"
    assert set(rep.metrics["test"]) >= {"roc_auc", "brier"}


# --- device selection --------------------------------------------------------------------
# The CNN was CPU-only. `DistrictPooling` holds its area weights as a sparse buffer, which
# is the one part of the model most likely to be left stranded on the wrong device.

def test_resolve_device_defaults_to_cuda_when_available():
    dev = train_cnn.resolve_device(None)
    assert dev.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_resolve_device_honours_an_explicit_override():
    assert train_cnn.resolve_device("cpu").type == "cpu"


def test_resolve_device_refuses_cuda_by_name_when_absent(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda"):
        train_cnn.resolve_device("cuda")


def test_streaming_predictions_agree_between_cpu_and_cuda(region_ids):
    """The GPU path must not silently change a single number. Same weights, same real
    bundle, two devices - if the sparse pooling buffer were left on the wrong device this
    would either crash (the good outcome) or, if it silently upcast/copied, disagree."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    grid_dir = Path(__file__).resolve().parents[1] / "data" / "samples" / "grids"
    bundle_paths = sorted(grid_dir.glob("*.npz"))[:2]
    if len(bundle_paths) < 2:
        pytest.skip("real grid bundles are not on disk")
    inits = [p.stem for p in bundle_paths]
    ev = _events(inits, region_ids[:8], leads=(1, 2))
    idx = train_cnn.build_index(grid_dir, ev, region_ids)
    idx.fit_normalizer(np.arange(len(idx.samples)))

    torch.manual_seed(0)
    state = {k: v.clone() for k, v in
             BustCNN(in_channels=idx.n_channels, region_ids=region_ids).state_dict().items()}

    cpu_model = BustCNN(in_channels=idx.n_channels, region_ids=region_ids).eval()
    cpu_model.load_state_dict(state)
    cpu_ps, _ = train_cnn.predict_streaming(cpu_model, idx, np.arange(len(idx.samples)),
                                            device=torch.device("cpu"))

    cuda_model = BustCNN(in_channels=idx.n_channels, region_ids=region_ids).eval().to("cuda")
    cuda_model.load_state_dict(state)
    cuda_ps, _ = train_cnn.predict_streaming(cuda_model, idx, np.arange(len(idx.samples)),
                                             device=torch.device("cuda"))

    assert np.allclose(cpu_ps, cuda_ps, atol=1e-4), \
        f"max diff {np.nanmax(np.abs(cpu_ps - cuda_ps)):.2e}"


def test_fit_streaming_trains_on_cuda_without_error(tmp_path, region_ids):
    """A full training step, not just a forward pass - the optimiser, the loss and the
    validation loop all have to agree to run on the same device as the model."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    inits = [f"2019-03-{d:02d}" for d in range(1, 4)]
    for i in inits:
        gf.save_bundle(tmp_path / f"{i}.npz", _bundle(i))
    ev = _events(inits, region_ids[:5])
    idx = train_cnn.build_index(tmp_path, ev, region_ids)
    idx.fit_normalizer(np.arange(len(idx.samples)))
    tr_idx = va_idx = np.arange(len(idx.samples))

    model, best = train_cnn.fit_streaming(0, idx, tr_idx, va_idx, epochs=1, patience=1,
                                          region_ids=region_ids, log=False,
                                          device=torch.device("cuda"))
    assert next(model.parameters()).device.type == "cuda"
    assert np.isfinite(best)


# --- reproducibility (E1) ---------------------------------------------------------------
# docs/team-brief-2026-09-15-updated.md Section 6, PHASE 1, E1: "Assert two runs with the
# same seed give bit-identical weights." No test anywhere did this before now -
# test_streaming_predictions_agree_between_cpu_and_cuda above loads ONE state_dict onto two
# devices and checks inference agrees; it assumes the weights already match. This checks
# whether training itself, run twice, produces them.

def _train_twice(device, region_ids):
    """Two independent fit_streaming(seed=0, ...) runs on the same real bundles.
    Returns both final state_dicts, moved to CPU so the comparison itself never touches
    the device - only training does."""
    grid_dir = Path(__file__).resolve().parents[1] / "data" / "samples" / "grids"
    bundle_paths = sorted(grid_dir.glob("*.npz"))[:2]
    if len(bundle_paths) < 2:
        pytest.skip("real grid bundles are not on disk")
    inits = [p.stem for p in bundle_paths]
    ev = _events(inits, region_ids[:8], leads=(1, 2))

    def run():
        idx = train_cnn.build_index(grid_dir, ev, region_ids)
        idx.fit_normalizer(np.arange(len(idx.samples)))
        all_idx = np.arange(len(idx.samples))
        model, _ = train_cnn.fit_streaming(
            0, idx, all_idx, all_idx, epochs=3, patience=3, region_ids=region_ids,
            log=False, device=device)
        return {k: v.clone().cpu() for k, v in model.state_dict().items()}

    return run(), run()


def _tensors_equal(x: torch.Tensor, y: torch.Tensor) -> bool:
    """torch.equal has no SparseCPU/SparseCUDA kernel - DistrictPooling.weight_matrix is
    a sparse COO buffer in every state_dict here. It is geometry, not a learned weight,
    rebuilt identically from the same parquet file on both runs regardless of seed, so
    densifying it for comparison is exact, not an approximation."""
    if x.is_sparse:
        return torch.equal(x.to_dense(), y.to_dense())
    return torch.equal(x, y)


def _mismatched_tensors(a: dict, b: dict) -> list[str]:
    assert a.keys() == b.keys()
    return [k for k in a if not _tensors_equal(a[k], b[k])]


def test_same_seed_gives_bit_identical_weights_on_cpu(region_ids):
    """E1's actual claim, forced onto CPU so this isolates the training loop's own logic
    (seeding, batch order, the auxiliary loss) from GPU-only sources of nondeterminism,
    which is checked separately below and is expected to behave differently."""
    a, b = _train_twice(torch.device("cpu"), region_ids)
    mismatched = _mismatched_tensors(a, b)
    assert not mismatched, f"non-deterministic on CPU: {mismatched}"


@pytest.mark.xfail(
    strict=True,
    reason="Diagnosed 2026-09-17, see docs/known-issues.md: fit_streaming's default CUDA "
           "path is not bit-reproducible. Root cause isolated to cuBLAS's GEMM algorithm "
           "selection, not cuDNN convolution and not DistrictPooling's torch.sparse.mm "
           "(both suspected, neither was it) - forcing torch.use_deterministic_algorithms"
           "(True) alone raises RuntimeError naming cuBLAS explicitly, and adding "
           "CUBLAS_WORKSPACE_CONFIG=:4096:8 plus torch.backends.cudnn.deterministic=True "
           "makes every tensor bit-identical (verified by hand). None of those three "
           "settings are applied in fit_streaming today. strict=True: if someone adds "
           "them, this starts passing and pytest fails loudly until the marker is removed.")
def test_same_seed_gives_bit_identical_weights_on_cuda(region_ids):
    """Same claim as the CPU test, on CUDA - where it currently fails. See the xfail
    reason above for the diagnosed root cause; this is deliberately not loosened per the
    brief's own instruction."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    a, b = _train_twice(torch.device("cuda"), region_ids)
    mismatched = _mismatched_tensors(a, b)
    assert not mismatched, f"non-deterministic on CUDA: {mismatched}"
