"""One-off: train a single CNN seed and dump its predictions to disk.

Not a permanent pipeline script. `app.ml.train_cnn --seeds 5` runs all five seeds
sequentially in one process, sharing one build_index/fit_normalizer pass; this variant
exists only because the sequential run leaves the GPU under 40% utilised (verified with
nvidia-smi on the RTX 4060, 2026-09-11) and there was time pressure to get five seeds
faster by running them as five OS processes instead. Each worker rebuilds its own index
and normalizer independently - cheap relative to a training run, and it avoids any
cross-process state to get wrong. What must stay identical across workers, and does
because build_index's iteration order depends only on `sorted(glob(...))`, is the sample
order backing train/val/test indices - the combine step assumes proba[i] and y[i] mean
the same sample for every seed.

Writes <out-dir>/seed_<seed>.npz with proba_train/y_train/proba_val/y_val/proba_test/
y_test/parameters, so a separate combine step can reproduce exactly what
`app.ml.train_cnn.train()`'s `scored()` computes for a multi-seed run, without
re-running any training.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--grid-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "data/samples/grids")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=3,
                    help="cap CPU threads per worker so 5 concurrent workers do not "
                         "oversubscribe the machine's cores")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)

    import pandas as pd

    from app.config import settings
    from app.db.base import resolve_path
    from app.ml import train_cnn
    from app.utils.india_districts import load_registry

    t0 = time.time()
    region_ids = [d.region_id for d in load_registry()]
    eval_dir = resolve_path(settings.data_dir) / "analysis" / "eval_events"
    events = pd.read_parquet(eval_dir / f"{args.run_id}.parquet")
    splits = train_cnn.splits_from_eval(events)

    index = train_cnn.build_index(args.grid_dir, events, region_ids)
    cyc = np.asarray(index.cycles)
    tr_idx = np.flatnonzero(np.isin(cyc, [str(c) for c in splits["train"]]))
    va_idx = np.flatnonzero(np.isin(cyc, [str(c) for c in splits["val"]]))
    te_idx = np.flatnonzero(np.isin(cyc, [str(c) for c in splits["test"]]))
    index.fit_normalizer(tr_idx)
    print(f"seed {args.seed}: index+normalizer ready, {time.time()-t0:.0f}s", flush=True)

    device = train_cnn.resolve_device(args.device)
    model, best_val_brier = train_cnn.fit_streaming(
        args.seed, index, tr_idx, va_idx, epochs=args.epochs, patience=args.patience,
        region_ids=region_ids, log=True, device=device)
    n_params = sum(p.numel() for p in model.parameters())

    out = {"parameters": n_params, "best_val_brier": best_val_brier}
    for name, idx in (("train", tr_idx), ("val", va_idx), ("test", te_idx)):
        proba, y = train_cnn.predict_streaming(model, index, idx, device=device)
        out[f"proba_{name}"] = proba
        out[f"y_{name}"] = y

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / f"seed_{args.seed}.npz", **out)
    print(f"seed {args.seed}: done, {time.time()-t0:.0f}s total, "
          f"{n_params:,} params, best-val-brier={best_val_brier:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
