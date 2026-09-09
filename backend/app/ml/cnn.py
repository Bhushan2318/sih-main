"""A convolutional bust model over the forecast fields.

The tabular models reduce each forecast to per-district numbers before they see it, which
discards the thing that causes a bust: the synoptic pattern. A depression sitting off the
Odisha coast, a western disturbance crossing Punjab - those are shapes in a field, and a
model given only a column of district means cannot see them. This one reads the field.

Shape of the problem
--------------------
Input is one initialisation's fields at one lead day: ensemble mean and spread for each
variable, on the 145 x 141 grid over lon 65-100, lat 2-38. Output is a bust probability
for each of the 666 districts.

Three decisions worth stating, because each is load-bearing:

**No downsampling.** A U-Net would halve the grid and upsample back, which is standard and
would cost nothing in accuracy - but the district weight table is defined on the 0.25 deg
grid, and keeping the feature map at that exact resolution lets the *same* weights that
aggregate the tabular data pool the network's features. Both model families then see
identical geography rather than two approximations of it. The receptive field comes from
dilation instead: 3x3 kernels at dilations 1, 2, 4, 8 reach 31 cells, about 850 km, which
is the scale of the systems that cause busts.

**A mask channel per input channel.** Missing is not zero. Soil moisture stops at day 3,
wind at day 5, and every land variable is absent over sea. Filling those with 0 after
normalisation puts them at the channel mean, which is a *plausible reading* - the network
would have no way to tell an average day from an absent one. Each channel is therefore
paired with a 1/0 mask, and the network is told where its inputs are real.

**Pooling is area-weighted, not learned.** How much of a district a grid cell covers is a
fact about geography, not a parameter. Learning it would spend capacity rediscovering
something already known exactly, on far less data than the atlas was built from.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from app.ingestion import grid_fields as gf
from app.utils import india_districts as idist

DILATIONS = (1, 2, 4, 8)
CHANNELS = 32


def receptive_field(dilations=DILATIONS, kernel: int = 3) -> int:
    """Cells the output can see, for reporting rather than for use."""
    return 1 + (kernel - 1) * sum(dilations)


class DistrictPooling(nn.Module):
    """Area-weighted mean of a feature map over each district.

    Holds the weight table as a sparse [districts, cells] matrix so pooling is one
    sparse matmul rather than 666 gathers. The weights are a registered buffer, not a
    parameter: they move with the model to a device and into a checkpoint, and they are
    never updated by an optimiser.
    """

    def __init__(self, region_ids: list[str] | None = None):
        super().__init__()
        weights = pd.read_parquet(idist.geo_dir() / idist.WEIGHTS_FILENAME)
        lats, lons = gf.domain_coords()
        self.height, self.width = len(lats), len(lons)

        lat_pos = {int(round(float(v) / gf.GRID_DEG)): i for i, v in enumerate(lats)}
        lon_pos = {int(round(float(v) / gf.GRID_DEG)): i for i, v in enumerate(lons)}

        self.region_ids = region_ids or sorted(weights["region_id"].unique())
        region_pos = {r: i for i, r in enumerate(self.region_ids)}

        rows, cols, vals = [], [], []
        for r, la, lo, w in weights.itertuples(index=False):
            ri = region_pos.get(r)
            y = lat_pos.get(int(round(float(la) / gf.GRID_DEG)))
            x = lon_pos.get(int(round(float(lo) / gf.GRID_DEG)))
            if ri is None or y is None or x is None:
                continue
            rows.append(ri)
            cols.append(y * self.width + x)
            vals.append(float(w))
        if not rows:
            raise ValueError("no district weights landed on the stored domain")

        idx = torch.tensor([rows, cols], dtype=torch.long)
        mat = torch.sparse_coo_tensor(
            idx, torch.tensor(vals, dtype=torch.float32),
            (len(self.region_ids), self.height * self.width),
        ).coalesce()
        self.register_buffer("weight_matrix", mat)

    @property
    def n_regions(self) -> int:
        return len(self.region_ids)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, districts, C]."""
        b, c, h, w = feats.shape
        if (h, w) != (self.height, self.width):
            raise ValueError(f"expected {self.height}x{self.width}, got {h}x{w}")
        flat = feats.reshape(b * c, h * w).t()               # [HW, B*C]
        pooled = torch.sparse.mm(self.weight_matrix, flat)    # [D, B*C]
        return pooled.t().reshape(b, c, self.n_regions).permute(0, 2, 1)


class BustCNN(nn.Module):
    """Fields in, one bust probability per district out."""

    def __init__(self, in_channels: int, n_extra: int = 1,
                 channels: int = CHANNELS, region_ids: list[str] | None = None):
        super().__init__()
        # Doubled: every data channel is paired with a mask saying where it is real.
        self.in_channels = in_channels * 2
        layers: list[nn.Module] = []
        prev = self.in_channels
        for d in DILATIONS:
            layers += [
                nn.Conv2d(prev, channels, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ]
            prev = channels
        self.encoder = nn.Sequential(*layers)
        self.pool = DistrictPooling(region_ids)
        # `n_extra` per-district scalars join after pooling - lead day, and anything else
        # that is a property of the district-day rather than of the field.
        self.head = nn.Sequential(
            nn.Linear(channels + n_extra, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(64, 1),
        )

    def forward(self, fields: torch.Tensor, mask: torch.Tensor,
                extra: torch.Tensor) -> torch.Tensor:
        """fields/mask [B, C, H, W], extra [B, D, n_extra] -> logits [B, D]."""
        x = torch.cat([torch.nan_to_num(fields), mask], dim=1)
        pooled = self.pool(self.encoder(x))                  # [B, D, channels]
        return self.head(torch.cat([pooled, extra], dim=-1)).squeeze(-1)


@dataclass
class Normalizer:
    """Per-channel mean and standard deviation, fit on training bundles only.

    Pressure is ~1000 hPa and rainfall is ~5 mm; without this the first convolution is
    dominated by whichever variable happens to have the largest units.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, stack: np.ndarray) -> "Normalizer":
        """stack: [N, C, H, W], NaN where missing."""
        axes = (0, 2, 3)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(stack, axis=axes)
            std = np.nanstd(stack, axis=axes)
        mean = np.nan_to_num(mean)
        # A channel that is constant, or absent everywhere, must not divide by zero.
        std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def apply(self, stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> (normalised with NaN preserved, mask of where the input was real)."""
        mask = np.isfinite(stack).astype(np.float32)
        out = (stack - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        return out.astype(np.float32), mask

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(np.asarray(d["mean"], dtype=np.float32),
                   np.asarray(d["std"], dtype=np.float32))


def bundle_to_channels(bundle: gf.GridBundle, lead: int) -> np.ndarray:
    """One lead day of a bundle as [C, H, W], C = variables x statistics.

    Channel order is (variable, statistic) and is fixed by the bundle's own ordering, so
    a model trained on one bundle can be applied to another only if the variable list
    matches - which is why it is stored alongside the weights.
    """
    li = bundle.leads.index(lead)
    return bundle.values[:, li].reshape(-1, len(bundle.lats), len(bundle.lons))


def channel_names(bundle: gf.GridBundle) -> list[str]:
    return [f"{v}_{s}" for v in bundle.variables for s in gf.STATS]


# ---------------------------------------------------------------------- serving

ENCODER_ONNX = "bust_cnn_encoder.onnx"


def export_encoder(model: BustCNN, path: Path) -> Path:
    """Export the convolutional encoder to ONNX. The encoder only - deliberately.

    The serving box has 512 MB and cannot hold PyTorch, so the trained network reaches it
    through onnxruntime. Exporting the *whole* model does not work, and fails in two
    different ways worth recording:

    * ``torch.sparse.mm`` has no ONNX operator at all, so exporting the pooling layer as
      written raises outright.
    * Rewriting the pooling as ``index_add`` exports **without error and computes the
      wrong answer**: ONNX does not support duplicated indices in a scatter, and ours are
      duplicated by construction - a district has many cells. Checked against torch, the
      exported model disagreed. That is the dangerous failure: a served model quietly
      producing wrong district values with nothing raising.

    So the split is drawn where ONNX is trustworthy. The encoder is ordinary strided
    convolution and exports exactly (agreement to 1e-7); pooling and the head are a
    weighted sum and a two-layer MLP, which numpy does in a few lines with no dependency
    and no operator coverage to verify.

    Measured, in a process with no torch: numpy plus onnxruntime is +25 MB, the session
    +5 MB, and inference one lead day at a time +21 MB - about +51 MB in total, 490 ms for
    all ten leads. Running all ten as one batch is faster (78 ms) but costs +163 MB, which
    does not fit. Lead days are scored one at a time on purpose.
    """
    import torch.onnx

    model = model.eval()
    dummy = torch.zeros(1, model.in_channels, model.pool.height, model.pool.width)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model.encoder, (dummy,), path,
        input_names=["x"], output_names=["feats"],
        dynamic_axes={"x": {0: "batch"}, "feats": {0: "batch"}},
        opset_version=17,
    )
    return path


def pool_and_head_numpy(feats: np.ndarray, extra: np.ndarray,
                        pooling: DistrictPooling, head_state: dict) -> np.ndarray:
    """The half of the forward pass that does not go through ONNX.

    ``feats`` [B, C, H, W] from the exported encoder; returns logits [B, districts].
    """
    mat = pooling.weight_matrix.coalesce()
    rows, cols = mat.indices().numpy()
    vals = mat.values().numpy()
    b, c, h, w = feats.shape
    flat = feats.reshape(b, c, h * w)

    n_regions = pooling.n_regions
    pooled = np.zeros((b, n_regions, c), dtype=np.float32)
    for i in range(b):
        contrib = flat[i][:, cols] * vals                       # [C, nnz]
        for ch in range(c):
            pooled[i, :, ch] = np.bincount(rows, weights=contrib[ch],
                                           minlength=n_regions)

    x = np.concatenate([pooled, extra], axis=-1)
    w0, b0 = head_state["0.weight"], head_state["0.bias"]
    w1, b1 = head_state["3.weight"], head_state["3.bias"]
    x = np.maximum(x @ w0.T + b0, 0.0)                          # ReLU; dropout is a no-op
    return (x @ w1.T + b1).squeeze(-1)
