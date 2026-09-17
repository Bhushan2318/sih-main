"""Streaming the fields from disk instead of materialising them.

`build_arrays` stacked every sample into one array. On the ten synthetic samples it was
tested with that is obviously fine; on one real year it is 14.3 GB before a single epoch
runs — X at [3650, 24, 145, 141] float32 is 7.2 GB and the mask is another 7.2 — on a
machine with 16 GB, and it fails the training loop's own requirement of fitting a 16 GB
runner. Five years would be 72 GB.

A cycle's bundle holds all ten of its lead days, so reading one file yields ten samples.
Batching by cycle means one file read per ten samples and a working set of megabytes
rather than gigabytes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="training-only dependency")

from app.ingestion import grid_fields as gf                      # noqa: E402
from app.ingestion.grid_fields import GridBundle                 # noqa: E402
from app.ml import train_cnn                                     # noqa: E402
from app.utils.india_districts import load_registry              # noqa: E402


@pytest.fixture(scope="module")
def region_ids():
    return [d.region_id for d in load_registry()]


def _write(tmp_path, init, n_var=3, leads=(1, 2)):
    lats, lons = gf.domain_coords()
    rng = np.random.default_rng(abs(hash(init)) % 2**32)
    vals = rng.normal(size=(n_var, len(leads), 2, len(lats), len(lons))).astype(np.float32)
    b = GridBundle(init, tuple(f"v{i}" for i in range(n_var)), tuple(leads), lats, lons, vals)
    gf.save_bundle(tmp_path / f"{init}.npz", b)
    return b


def _events(inits, region_ids, leads=(1, 2)):
    rng = np.random.default_rng(0)
    return pd.DataFrame([
        {"region_id": r, "init_date": i, "lead_time_days": l,
         "y_bust": int(rng.random() > 0.5), "bust_ratio": float(rng.random() * 2)}
        for i in inits for l in leads for r in region_ids[:6]])


def test_the_index_holds_paths_not_fields(tmp_path, region_ids):
    """The whole point: the index must be small enough that a decade of it is nothing."""
    inits = [f"2017-01-{d:02d}" for d in range(1, 6)]
    for i in inits:
        _write(tmp_path, i)
    idx = train_cnn.build_index(tmp_path, _events(inits, region_ids), region_ids)
    assert len(idx.samples) == 10, "5 cycles x 2 lead days"
    total = sum(a.nbytes for a in (idx.labels, idx.aux, idx.extra))
    assert total < 5_000_000, f"index carries {total/1e6:.1f} MB; it must not hold fields"


def test_a_batch_loads_only_its_own_cycle(tmp_path, region_ids):
    inits = [f"2017-01-{d:02d}" for d in range(1, 4)]
    for i in inits:
        _write(tmp_path, i)
    idx = train_cnn.build_index(tmp_path, _events(inits, region_ids), region_ids)
    batches = list(idx.batches(np.arange(len(idx.samples)), shuffle=False))
    assert len(batches) == 3, "one batch per cycle"
    x, m, ex, y, aux = batches[0]
    assert x.shape[0] == 2 and x.shape[1] == 6, "2 lead days, 3 variables x mean/spread"
    assert m.shape == x.shape


def test_streamed_values_match_a_direct_load(tmp_path, region_ids):
    """Streaming must not change a single number."""
    init = "2017-07-17"
    _write(tmp_path, init)
    idx = train_cnn.build_index(tmp_path, _events([init], region_ids), region_ids)
    idx.fit_normalizer(np.arange(len(idx.samples)))
    x, m, ex, y, aux = next(iter(idx.batches(np.arange(len(idx.samples)), shuffle=False)))
    # Compare against the bundle as stored, not as built: saving applies the int16
    # quantisation, so the in-memory values differ by up to half a quantum by design.
    b = gf.load_bundle(tmp_path / f"{init}.npz")
    raw = b.values[:, 0].reshape(-1, len(b.lats), len(b.lons))
    want = (raw - idx.norm.mean[:, None, None]) / idx.norm.std[:, None, None]
    assert np.allclose(np.nan_to_num(x[0]), np.nan_to_num(want), atol=1e-4)


def test_normaliser_is_fitted_on_training_samples_only(tmp_path, region_ids):
    """Statistics over the whole archive leak the test period's climate into training."""
    inits = [f"2017-01-{d:02d}" for d in range(1, 7)]
    for i in inits:
        _write(tmp_path, i)
    idx = train_cnn.build_index(tmp_path, _events(inits, region_ids), region_ids)
    train_only = np.arange(4)
    idx.fit_normalizer(train_only)
    a = idx.norm.mean.copy()
    idx.fit_normalizer(np.arange(len(idx.samples)))
    assert not np.allclose(a, idx.norm.mean), "a different sample set must give different stats"


def test_working_set_stays_small(tmp_path, region_ids):
    """One cycle at a time, not one year."""
    inits = [f"2017-01-{d:02d}" for d in range(1, 9)]
    for i in inits:
        _write(tmp_path, i)
    idx = train_cnn.build_index(tmp_path, _events(inits, region_ids), region_ids)
    idx.fit_normalizer(np.arange(len(idx.samples)))
    peak = 0
    for x, m, ex, y, aux in idx.batches(np.arange(len(idx.samples)), shuffle=False):
        peak = max(peak, x.nbytes + m.nbytes)
    assert peak < 50_000_000, f"a batch holds {peak/1e6:.0f} MB"
