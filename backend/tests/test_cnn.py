"""The convolutional bust model.

torch is not in requirements.txt - it cannot go on a 512 MB serving box - so these skip
where it is absent rather than failing the suite. Install with:
    pip install -r requirements-train.txt
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="training-only dependency")

from app.ingestion import grid_fields as gf                      # noqa: E402
from app.ingestion.grid_fields import GridBundle                 # noqa: E402
from app.ml.cnn import (                                         # noqa: E402
    BustCNN, DistrictPooling, Normalizer, bundle_to_channels, channel_names,
    receptive_field,
)
from app.utils.india_districts import get_aggregator             # noqa: E402


@pytest.fixture(scope="module")
def pool():
    return DistrictPooling()


def _bundle(n_var=4, n_lead=3):
    lats, lons = gf.domain_coords()
    rng = np.random.default_rng(0)
    vals = (rng.normal(size=(n_var, n_lead, 2, len(lats), len(lons))) * 10 + 300)
    return GridBundle("2019-07-17", tuple(f"v{i}" for i in range(n_var)),
                      tuple(range(1, n_lead + 1)), lats, lons, vals.astype(np.float32))


# ----------------------------------------------------------------------- pooling

def test_pooling_matches_the_tabular_aggregator(pool):
    """The load-bearing claim of the architecture: the network pools its feature map
    through the same area weights that build the tabular rows, so both model families
    see identical geography rather than two approximations of it."""
    lats, lons = gf.domain_coords()
    field = np.random.default_rng(0).uniform(-5, 40, (len(lats), len(lons))).astype(np.float32)

    got = pool(torch.from_numpy(field)[None, None])[0, :, 0].numpy()

    agg = get_aggregator()
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    want = agg.aggregate_prepared(agg.prepare(la.ravel(), lo.ravel()), field.ravel())
    want = want.reindex(pool.region_ids).to_numpy()

    assert np.allclose(got, want, atol=1e-4), \
        f"max diff {np.nanmax(np.abs(got - want)):.2e}"


def test_pooling_of_a_constant_field_is_that_constant(pool):
    out = pool(torch.full((1, 1, pool.height, pool.width), 7.5))
    assert torch.allclose(out, torch.tensor(7.5), atol=1e-4)


def test_pooling_covers_every_district(pool):
    assert pool.n_regions == 666
    rows = pool.weight_matrix.coalesce().indices()[0].unique()
    assert len(rows) == 666, "a district with no weights could never be predicted"


def test_pooling_weights_are_not_learned(pool):
    """How much of a district a cell covers is geography, not a parameter. Learning it
    would spend capacity rediscovering something already known exactly."""
    assert not any(p is pool.weight_matrix for p in pool.parameters())
    assert "weight_matrix" in dict(pool.named_buffers())


def test_pooling_rejects_a_wrong_sized_feature_map(pool):
    with pytest.raises(ValueError):
        pool(torch.zeros(1, 1, 10, 10))


# -------------------------------------------------------------------- normaliser

def test_normaliser_standardises_each_channel():
    stack = np.stack([bundle_to_channels(_bundle(), l) for l in (1, 2, 3)])
    x, _ = Normalizer.fit(stack).apply(stack)
    assert abs(np.nanmean(x)) < 0.05 and abs(np.nanstd(x) - 1.0) < 0.05


def test_normaliser_survives_a_constant_channel():
    """Dividing by a zero standard deviation would produce inf and poison every weight
    it touches on the first backward pass."""
    stack = np.zeros((2, 3, 8, 8), dtype=np.float32)
    stack[:, 1] = 5.0
    x, _ = Normalizer.fit(stack).apply(stack)
    assert np.isfinite(x).all()


def test_normaliser_survives_an_absent_channel():
    stack = np.full((2, 2, 8, 8), np.nan, dtype=np.float32)
    stack[:, 0] = 3.0
    n = Normalizer.fit(stack)
    x, mask = n.apply(stack)
    assert np.isfinite(n.mean).all() and np.isfinite(n.std).all()
    assert mask[:, 1].sum() == 0


def test_normaliser_round_trips():
    n = Normalizer.fit(np.random.default_rng(0).normal(size=(4, 3, 8, 8)))
    back = Normalizer.from_dict(n.to_dict())
    assert np.allclose(n.mean, back.mean) and np.allclose(n.std, back.std)


def test_mask_separates_missing_from_average():
    """Soil moisture stops at day 3 and wind at day 5. Filled with 0 after normalisation
    those sit exactly at the channel mean - a plausible reading - so without the mask the
    network cannot tell an average day from an absent one."""
    stack = np.zeros((1, 2, 8, 8), dtype=np.float32)
    stack[0, 1, :4] = np.nan
    _, mask = Normalizer.fit(stack).apply(stack)
    assert mask[0, 1, :4].sum() == 0
    assert mask[0, 1, 4:].min() == 1
    assert mask[0, 0].min() == 1


# ------------------------------------------------------------------------ model

def test_forward_gives_one_probability_per_district():
    b = _bundle()
    stack = np.stack([bundle_to_channels(b, l) for l in b.leads])
    x, mask = Normalizer.fit(stack).apply(stack)
    model = BustCNN(in_channels=stack.shape[1])
    out = model(torch.from_numpy(x), torch.from_numpy(mask),
                torch.zeros(len(b.leads), model.pool.n_regions, 1))
    assert out.shape == (len(b.leads), 666)
    assert torch.isfinite(out).all()


def test_gradients_reach_the_encoder():
    b = _bundle()
    stack = np.stack([bundle_to_channels(b, l) for l in b.leads])
    x, mask = Normalizer.fit(stack).apply(stack)
    model = BustCNN(in_channels=stack.shape[1])
    extra = torch.zeros(len(b.leads), model.pool.n_regions, 1)
    loss = torch.nn.BCEWithLogitsLoss()(
        model(torch.from_numpy(x), torch.from_numpy(mask), extra),
        torch.randint(0, 2, (len(b.leads), 666)).float())
    loss.backward()
    first = model.encoder[0]
    assert first.weight.grad is not None and first.weight.grad.abs().sum() > 0, \
        "the sparse pooling matmul must not cut the graph"


def test_nan_input_does_not_produce_nan_output():
    b = _bundle()
    stack = np.stack([bundle_to_channels(b, l) for l in b.leads])
    stack[:, 0, :10] = np.nan
    x, mask = Normalizer.fit(stack).apply(stack)
    model = BustCNN(in_channels=stack.shape[1])
    out = model(torch.from_numpy(x), torch.from_numpy(mask),
                torch.zeros(len(b.leads), model.pool.n_regions, 1))
    assert torch.isfinite(out).all()


def test_model_is_small_enough_for_the_data():
    """344 cycles is not much evidence. A network with millions of parameters would
    memorise it; this is a deliberate ceiling, not an accident."""
    model = BustCNN(in_channels=16)
    n = sum(p.numel() for p in model.parameters())
    assert n < 200_000, f"{n:,} parameters is too many for the training set"


def test_receptive_field_reaches_synoptic_scale():
    """A bust is caused by a system hundreds of km across. A network that can only see
    its own district cannot see the depression approaching it."""
    km = receptive_field() * gf.GRID_DEG * 111
    assert km > 500, f"receptive field only {km:.0f} km"


def test_channel_names_match_channel_count():
    b = _bundle()
    assert len(channel_names(b)) == bundle_to_channels(b, 1).shape[0]


# ---------------------------------------------------------------- serving path

def test_encoder_exports_to_onnx_and_agrees_with_torch(tmp_path):
    """The serving box cannot hold PyTorch, so the encoder has to survive the trip
    through ONNX exactly."""
    ort = pytest.importorskip("onnxruntime")
    from app.ml.cnn import export_encoder

    model = BustCNN(in_channels=8).eval()
    x = torch.randn(1, model.in_channels, model.pool.height, model.pool.width)
    with torch.no_grad():
        ref = model.encoder(x).numpy()

    path = export_encoder(model, tmp_path / "encoder.onnx")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"x": x.numpy()})[0]
    assert np.allclose(got, ref, atol=1e-4), f"max diff {np.abs(got - ref).max():.2e}"


def test_numpy_pooling_and_head_reproduce_the_torch_model(tmp_path):
    """Pooling and the head are done in numpy at serve time rather than in ONNX, because
    a scatter with duplicated indices exports without error and computes the wrong
    answer. This pins that the hand-written half matches torch."""
    from app.ml.cnn import pool_and_head_numpy

    model = BustCNN(in_channels=8).eval()
    x = torch.randn(2, model.in_channels, model.pool.height, model.pool.width)
    extra = torch.randn(2, model.pool.n_regions, 1)
    with torch.no_grad():
        ref = model.head(
            torch.cat([model.pool(model.encoder(x)), extra], dim=-1)
        ).squeeze(-1).numpy()
        feats = model.encoder(x).numpy()

    head_state = {k: v.numpy() for k, v in model.head.state_dict().items()}
    got = pool_and_head_numpy(feats, extra.numpy(), model.pool, head_state)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-4), f"max diff {np.abs(got - ref).max():.2e}"
