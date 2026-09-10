"""Train the convolutional bust model, and score it beside the tabular one.

The point of this file is comparison. The convolutional model is a challenger, not a
replacement, so it is trained on the same cycles as the XGBoost classifier, split the same
way, and scored with the same function - and the numbers it produces slot into the same
baseline ladder. If it wins it should win on rows the incumbent also saw; if it loses,
that is a real result and worth reporting rather than hiding.

Two disciplines are load-bearing here, and both are easy to get silently wrong.

**Split by cycle, never by row.** Two districts on the same day are not independent
observations - they are the same weather seen twice. Splitting rows at random would put
the same synoptic situation in train and test, and the score would look excellent and mean
nothing. The cycle split is taken from the same function the tabular pipeline uses, so the
two models are held out on identical dates.

**Fit the normaliser on training cycles only.** Channel means and standard deviations
computed over the whole archive leak the test period's climate into training.

Sample size governs everything here. One cycle yields ten training samples, one per lead
day, so 17 cycles is 170 samples against 115,000 parameters and any score from it is
noise. The model is deliberately small, heavily regularised, and early-stopped on a
validation set for that reason - and this script refuses to run below a floor rather than
emit a number that would be quoted later.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.ingestion import grid_fields as gf
from app.ml import classifier as clf_mod

# Below this, a run says nothing about the architecture - only about the sample size.
MIN_TRAIN_CYCLES = 120

DEFAULT_EPOCHS = 60
DEFAULT_PATIENCE = 8
DEFAULT_LR = 3e-4
DEFAULT_SEEDS = 5
AUX_WEIGHT = 0.3      # weight on the auxiliary error-magnitude head


@dataclass
class CNNReport:
    status: str
    seeds: int = 0
    train_cycles: int = 0
    val_cycles: int = 0
    test_cycles: int = 0
    n_train_samples: int = 0
    parameters: int = 0
    metrics: dict = field(default_factory=dict)
    seconds: float = 0.0
    error: str | None = None


def load_bundles(grid_dir: Path) -> dict:
    """init_date (ISO string) -> GridBundle, for every bundle on disk."""
    out = {}
    for p in sorted(Path(grid_dir).glob("*.npz")):
        b = gf.load_bundle(p)
        out[str(b.init_date)] = b
    return out


@dataclass
class FieldIndex:
    """Where each sample lives, and its label - never the fields themselves.

    Stacking every sample into one array was 14.3 GB for a single year (X at
    [3650, 24, 145, 141] float32 is 7.2 GB and the mask another 7.2), on a machine with
    16 GB, and five years would be 72 GB. It passed its tests because those used ten
    synthetic samples.

    A cycle's bundle holds all ten of its lead days, so one file read yields ten samples.
    Batching by cycle keeps the working set at tens of megabytes and makes the number of
    years irrelevant to memory.
    """

    samples: list          # (bundle_path, lead_index)
    labels: np.ndarray     # [n, districts]
    aux: np.ndarray        # [n, districts]
    extra: np.ndarray      # [n, districts, 1]
    cycles: np.ndarray     # [n] init date per sample
    n_channels: int
    norm: "Normalizer | None" = None

    def _load(self, i: int) -> np.ndarray:
        path, li = self.samples[i]
        bundle = gf.load_bundle(path)
        return bundle.values[:, li].reshape(-1, len(bundle.lats), len(bundle.lons))

    def fit_normalizer(self, idx: np.ndarray) -> "Normalizer":
        """Channel statistics over the given samples only, accumulated one cycle at a
        time. Fitting over the whole archive would leak the test period's climate into
        training."""
        from app.ml.cnn import Normalizer

        n = np.zeros(self.n_channels)
        total = np.zeros(self.n_channels)
        sq = np.zeros(self.n_channels)
        for i in idx:
            f = self._load(int(i))
            finite = np.isfinite(f)
            n += finite.sum(axis=(1, 2))
            total += np.where(finite, f, 0.0).sum(axis=(1, 2))
            sq += np.where(finite, f.astype(np.float64) ** 2, 0.0).sum(axis=(1, 2))
        mean = np.divide(total, n, out=np.zeros_like(total), where=n > 0)
        var = np.divide(sq, n, out=np.zeros_like(sq), where=n > 0) - mean ** 2
        std = np.sqrt(np.maximum(var, 0.0))
        std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
        self.norm = Normalizer(mean.astype(np.float32), std.astype(np.float32))
        return self.norm

    def batches(self, idx: np.ndarray, shuffle: bool = True, rng=None):
        """One batch per cycle: every lead day of one bundle, from one file read."""
        by_cycle: dict = {}
        for i in idx:
            by_cycle.setdefault(self.samples[int(i)][0], []).append(int(i))
        order = list(by_cycle)
        if shuffle:
            (rng or np.random.default_rng()).shuffle(order)
        for path in order:
            members = by_cycle[path]
            bundle = gf.load_bundle(path)
            raw = np.stack([bundle.values[:, self.samples[i][1]].reshape(
                -1, len(bundle.lats), len(bundle.lons)) for i in members])
            if self.norm is not None:
                x, m = self.norm.apply(raw)
                x = np.nan_to_num(x)
            else:
                x, m = np.nan_to_num(raw), np.isfinite(raw).astype(np.float32)
            yield (x.astype(np.float32), m.astype(np.float32),
                   self.extra[members], self.labels[members], self.aux[members])


def build_index(grid_dir, events, region_ids: list[str]) -> FieldIndex:
    """Index the samples without reading a single field into memory."""
    import pandas as pd

    pos = {r: i for i, r in enumerate(region_ids)}
    ev = events.copy()
    ev["init_date"] = pd.to_datetime(ev["init_date"]).dt.date.astype(str)
    ev["_ri"] = ev["region_id"].astype(str).map(pos)
    ev = ev[ev["_ri"].notna()]
    ev["_ri"] = ev["_ri"].astype(int)

    n_reg = len(region_ids)
    samples, labels, aux, extra, cycles = [], [], [], [], []
    n_channels = 0
    for path in sorted(Path(grid_dir).glob("*.npz")):
        init = path.stem
        rows = ev[ev["init_date"] == init]
        if rows.empty:
            continue
        with np.load(path, allow_pickle=False) as z:
            leads = [int(v) for v in z["leads"]]
            n_channels = int(z["data_hi"].shape[0]) * int(z["data_hi"].shape[2])
        for li, lead in enumerate(leads):
            lr = rows[rows["lead_time_days"] == lead]
            if lr.empty:
                continue
            y = np.full(n_reg, np.nan, dtype=np.float32)
            a = np.full(n_reg, np.nan, dtype=np.float32)
            y[lr["_ri"].to_numpy()] = lr["y_bust"].to_numpy(dtype=np.float32)
            if "bust_ratio" in lr:
                a[lr["_ri"].to_numpy()] = np.log1p(
                    np.clip(lr["bust_ratio"].to_numpy(dtype=np.float64), 0, None))
            samples.append((path, li))
            labels.append(y)
            aux.append(a)
            extra.append(np.full((n_reg, 1), (lead - 5.5) / 4.5, dtype=np.float32))
            cycles.append(init)
    return FieldIndex(samples, np.stack(labels) if labels else np.zeros((0, n_reg), np.float32),
                      np.stack(aux) if aux else np.zeros((0, n_reg), np.float32),
                      np.stack(extra) if extra else np.zeros((0, n_reg, 1), np.float32),
                      np.array(cycles), n_channels)


def build_arrays(bundles: dict, events, region_ids: list[str]):
    """Grid bundles + scored events -> (X, mask, extra, y, aux, cycle) aligned by sample.

    One sample is a (cycle, lead day) pair: the whole field, and one label per district.
    Districts are columns, not rows, because the network predicts all 666 at once - which
    is also why a district with no label has to be masked rather than dropped.
    """
    import pandas as pd

    pos = {r: i for i, r in enumerate(region_ids)}
    ev = events.copy()
    ev["init_date"] = pd.to_datetime(ev["init_date"]).dt.date.astype(str)
    ev["_ri"] = ev["region_id"].astype(str).map(pos)
    ev = ev[ev["_ri"].notna()]
    ev["_ri"] = ev["_ri"].astype(int)

    X, M, EX, Y, AUX, CYC = [], [], [], [], [], []
    n_reg = len(region_ids)
    for init, bundle in sorted(bundles.items()):
        rows = ev[ev["init_date"] == init]
        if rows.empty:
            continue
        for lead in bundle.leads:
            lr = rows[rows["lead_time_days"] == lead]
            if lr.empty:
                continue
            y = np.full(n_reg, np.nan, dtype=np.float32)
            aux = np.full(n_reg, np.nan, dtype=np.float32)
            y[lr["_ri"].to_numpy()] = lr["y_bust"].to_numpy(dtype=np.float32)
            if "bust_ratio" in lr:
                aux[lr["_ri"].to_numpy()] = np.log1p(
                    np.clip(lr["bust_ratio"].to_numpy(dtype=np.float64), 0, None))
            X.append(bundle.values[:, bundle.leads.index(lead)].reshape(
                -1, len(bundle.lats), len(bundle.lons)))
            EX.append(np.full((n_reg, 1), (lead - 5.5) / 4.5, dtype=np.float32))
            Y.append(y)
            AUX.append(aux)
            CYC.append(init)
    if not X:
        return None
    X = np.stack(X)
    return X, np.stack(EX), np.stack(Y), np.stack(AUX), np.array(CYC)


def fit_streaming(seed, index: "FieldIndex", tr_idx, va_idx,
                  epochs=DEFAULT_EPOCHS, patience=DEFAULT_PATIENCE, lr=DEFAULT_LR,
                  region_ids=None, log=True):
    """Train from the index, reading one cycle at a time.

    Same loop as _fit_one, but the fields never all exist at once: a batch is one bundle's
    lead days, loaded, used and dropped. Peak memory is a batch rather than a year, which
    is what lets this run on a 16 GB runner and what makes five years possible at all.
    """
    import time as _t

    import torch
    import torch.nn as nn
    from app.ml.cnn import BustCNN

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = BustCNN(in_channels=index.n_channels, region_ids=region_ids)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    huber = nn.HuberLoss(reduction="none")

    best, best_state, since = np.inf, None, 0
    for ep in range(epochs):
        t0 = _t.time()
        model.train()
        last = 0.0
        for x, m, ex, y, aux in index.batches(tr_idx, shuffle=True, rng=rng):
            opt.zero_grad()
            xb, mb = torch.from_numpy(x), torch.from_numpy(m)
            yb, ab = torch.from_numpy(y), torch.from_numpy(aux)
            logits = model(xb, mb, torch.from_numpy(ex))
            keep = ~torch.isnan(yb)
            if not keep.any():
                continue
            loss = bce(logits[keep], yb[keep]).mean()
            akeep = ~torch.isnan(ab)
            if akeep.any():
                loss = loss + AUX_WEIGHT * huber(
                    torch.sigmoid(logits[akeep]), torch.tanh(ab[akeep])).mean()
            loss.backward()
            opt.step()
            last = float(loss)

        model.eval()
        num = den = 0.0
        with torch.no_grad():
            for x, m, ex, y, aux in index.batches(va_idx, shuffle=False):
                p = torch.sigmoid(model(torch.from_numpy(x), torch.from_numpy(m),
                                        torch.from_numpy(ex))).numpy()
                keep = ~np.isnan(y)
                num += float(((p[keep] - y[keep]) ** 2).sum())
                den += int(keep.sum())
        vb = num / max(den, 1)
        if log:
            print(f"    epoch {ep+1:>2}  train-loss {last:.4f}  val-Brier {vb:.4f}  "
                  f"{_t.time()-t0:.0f}s", flush=True)
        if vb < best - 1e-5:
            best, since = vb, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best


def predict_streaming(model, index: "FieldIndex", idx) -> tuple:
    """Probabilities and labels for the given samples, in index order."""
    import torch

    ps, ys = [], []
    with torch.no_grad():
        for x, m, ex, y, aux in index.batches(idx, shuffle=False):
            ps.append(torch.sigmoid(model(torch.from_numpy(x), torch.from_numpy(m),
                                          torch.from_numpy(ex))).numpy())
            ys.append(y)
    return np.concatenate(ps), np.concatenate(ys)


def _fit_one(seed, Xtr, Mtr, EXtr, Ytr, AUXtr, Xva, Mva, EXva, Yva,
             epochs, patience, lr, region_ids):
    import torch
    import torch.nn as nn
    from app.ml.cnn import BustCNN

    torch.manual_seed(seed)
    np.random.seed(seed)
    # Xtr carries the DATA channels; BustCNN doubles that itself because forward
    # concatenates the mask. Halving here built an encoder for half the channels it would
    # be handed, and nothing failed until the first batch reached the first convolution.
    model = BustCNN(in_channels=Xtr.shape[1], region_ids=region_ids)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    huber = nn.HuberLoss(reduction="none")

    def batches(X, M, EX, Y, AUX, size=8, shuffle=True):
        idx = np.arange(len(X))
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, len(idx), size):
            j = idx[i:i + size]
            yield (torch.from_numpy(X[j]), torch.from_numpy(M[j]),
                   torch.from_numpy(EX[j]), torch.from_numpy(Y[j]),
                   torch.from_numpy(AUX[j]))

    best, best_state, since = np.inf, None, 0
    import time as _t
    for _ep in range(epochs):
        _t0 = _t.time()
        model.train()
        for xb, mb, eb, yb, ab in batches(Xtr, Mtr, EXtr, Ytr, AUXtr):
            opt.zero_grad()
            logits = model(xb, mb, eb)
            keep = ~torch.isnan(yb)
            if not keep.any():
                continue
            loss = bce(logits[keep], yb[keep]).mean()
            akeep = ~torch.isnan(ab)
            if akeep.any():
                # Auxiliary head shares the encoder: predicting how wrong, as well as
                # whether it busts, extracts more signal per sample.
                loss = loss + AUX_WEIGHT * huber(
                    torch.sigmoid(logits[akeep]), torch.tanh(ab[akeep])).mean()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = model(torch.from_numpy(Xva), torch.from_numpy(Mva),
                       torch.from_numpy(EXva))
            vy = torch.from_numpy(Yva)
            keep = ~torch.isnan(vy)
            # Selected on Brier, not accuracy: this ships a probability to someone
            # deciding whether to warn, so calibration is the objective.
            vb = ((torch.sigmoid(vl[keep]) - vy[keep]) ** 2).mean().item()
        print(f"    epoch {_ep+1:>2}  train-loss {float(loss):.4f}  val-Brier {vb:.4f}  "
              f"{_t.time()-_t0:.0f}s", flush=True)
        if vb < best - 1e-5:
            best, since = vb, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best


def _predict(model, X, M, EX, chunk=8) -> np.ndarray:
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            out.append(torch.sigmoid(model(
                torch.from_numpy(X[i:i + chunk]), torch.from_numpy(M[i:i + chunk]),
                torch.from_numpy(EX[i:i + chunk]))).numpy())
    return np.concatenate(out)


def train(grid_dir: Path, events, splits: dict, region_ids: list[str],
          seeds: int = DEFAULT_SEEDS, epochs: int = DEFAULT_EPOCHS,
          patience: int = DEFAULT_PATIENCE, lr: float = DEFAULT_LR) -> CNNReport:
    t0 = time.time()
    try:
        import torch  # noqa: F401
    except ImportError:
        return CNNReport(status="skipped",
                         error="torch is not installed; pip install -r requirements-train.txt")

    from app.ml.cnn import Normalizer

    bundles = load_bundles(grid_dir)
    if not bundles:
        return CNNReport(status="skipped",
                         error=f"no grid bundles in {grid_dir}; the fetch must run first")

    built = build_arrays(bundles, events, region_ids)
    if built is None:
        return CNNReport(status="skipped",
                         error="no bundle matched a scored event; check init_date alignment")
    X, EX, Y, AUX, CYC = built

    tr = np.isin(CYC, [str(c) for c in splits.get("train", [])])
    va = np.isin(CYC, [str(c) for c in splits.get("val", [])])
    te = np.isin(CYC, [str(c) for c in splits.get("test", [])])
    n_tr_cycles = len(set(CYC[tr]))
    if n_tr_cycles < MIN_TRAIN_CYCLES:
        return CNNReport(
            status="refused", train_cycles=n_tr_cycles,
            error=(f"only {n_tr_cycles} training cycles ({tr.sum()} samples) against "
                   f"~115k parameters. Below {MIN_TRAIN_CYCLES} cycles any score measures "
                   f"the sample size, not the architecture, and would be quoted later as "
                   f"if it measured the model. Refusing rather than publishing it."),
            seconds=time.time() - t0)

    # Normaliser fit on training cycles only - whole-archive statistics leak the test
    # period's climate into training.
    norm = Normalizer.fit(X[tr])
    Xn, M = norm.apply(X)
    Xn = np.nan_to_num(Xn)

    models, val_briers = [], []
    for s in range(seeds):
        m, vb = _fit_one(s, Xn[tr], M[tr], EX[tr], Y[tr], AUX[tr],
                         Xn[va], M[va], EX[va], Y[va], epochs, patience, lr, region_ids)
        models.append(m)
        val_briers.append(vb)

    def scored(mask):
        if not mask.any():
            return {}
        # Seed ensemble: averaging probabilities cuts variance and improves calibration,
        # which is nearly free at this model size.
        proba = np.mean([_predict(m, Xn[mask], M[mask], EX[mask]) for m in models], axis=0)
        keep = ~np.isnan(Y[mask])
        return clf_mod._evaluate(Y[mask][keep], proba[keep])

    metrics = {k: scored(m) for k, m in (("train", tr), ("val", va), ("test", te))}
    n_params = sum(p.numel() for p in models[0].parameters())
    return CNNReport(
        status="success", seeds=seeds,
        train_cycles=n_tr_cycles, val_cycles=len(set(CYC[va])), test_cycles=len(set(CYC[te])),
        n_train_samples=int(tr.sum()), parameters=int(n_params),
        metrics=metrics, seconds=time.time() - t0)


def format_comparison(cnn: CNNReport, xgb_metrics: dict) -> str:
    """Both models on the same held-out cycles, side by side."""
    rows = []
    c = (cnn.metrics or {}).get("test", {})
    x = (xgb_metrics or {}).get("test", {})
    for key, label in [("roc_auc", "ROC-AUC"), ("brier", "Brier (lower better)"),
                       ("f1", "F1"), ("pr_auc", "PR-AUC")]:
        cv, xv = c.get(key), x.get(key)
        if cv is None and xv is None:
            continue
        better = ""
        if isinstance(cv, (int, float)) and isinstance(xv, (int, float)):
            better = "CNN" if ((cv > xv) ^ (key == "brier")) else "XGBoost"
        rows.append(f"  {label:<24}{_fmt(xv):>12}{_fmt(cv):>12}   {better}")
    head = (f"  {'metric':<24}{'XGBoost':>12}{'CNN':>12}   winner\n  {'-'*62}")
    return "\n".join([head, *rows])


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "-"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-dir", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data/samples/grids")
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from dataclasses import asdict

    from app.ml import inference, registry
    from app.utils.india_districts import load_registry

    state = inference.load_state() if hasattr(inference, "load_state") else None
    events = getattr(state, "events", None)
    if events is None:
        print("No scored events available - train the tabular pipeline first "
              "(python -m app.ml.train_pipeline).")
        return 1

    rep = train(args.grid_dir, events, {}, [d.region_id for d in load_registry()],
                seeds=args.seeds, epochs=args.epochs)
    print(f"CNN: status={rep.status} {rep.error or ''}")
    if rep.status == "success":
        print(f"  {rep.train_cycles} train cycles, {rep.n_train_samples} samples, "
              f"{rep.parameters:,} parameters, {rep.seeds} seeds, {rep.seconds:.0f}s")
        cur = registry.current_run_id()
        xgb = (registry.load_metrics(cur) or {}).get("classifier", {}) if cur else {}
        print(format_comparison(rep, xgb))
    if args.json:
        print(json.dumps(asdict(rep), indent=2, default=str))
    return 0 if rep.status in ("success", "skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
