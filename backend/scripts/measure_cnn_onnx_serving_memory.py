"""Reproduce the CNN's ONNX-serving memory/timing claim (E4, docs/team-brief-2026-09-15-
updated.md Section 6, PHASE 1).

`app/ml/cnn.py`'s `export_encoder` docstring and CLAUDE.md's measured facts both state:
"+51 MB, 490 ms for all 10 lead days via onnxruntime, scoring one lead at a time". Per the
brief: "Reproduce it. If you get a different number, report the difference - do not
quietly adopt ours." This script does that, on whatever machine runs it, and prints the
real number instead of the one written down.

Two phases, in two separate OS processes, because the claim being measured is specifically
about a process that never imports torch - the whole reason `export_encoder` splits the
model at the ONNX boundary in the first place (the serving box has 512 MB and cannot hold
PyTorch). Running both phases in one interpreter would let torch's own import cost bleed
into the "no torch" number.

  Phase 1 (this process, has torch): build a BustCNN at real dimensions - 24 channels (12
  variables x 2 statistics, matching the live fetch) and 666 real districts - export its
  encoder to ONNX, and dump the head weights and district pooling matrix as plain numpy
  .npz files. Nothing measured here.

  Phase 2 (a fresh `python -m scripts.measure_cnn_onnx_serving_memory --child`,
  subprocess.run'd from phase 1): imports ONLY numpy and onnxruntime - never torch, never
  app.ml.cnn (which imports torch at module scope) - loads the phase-1 artifacts, and
  measures peak RSS after each stage plus wall time for all 10 lead days, one at a time,
  matching export_encoder's own "scored one lead at a time on purpose" design.

The pooling math (weighted sum over district cells, mirroring `DistrictPooling.forward`
and the existing `pool_and_head_numpy` in app/ml/cnn.py) is reimplemented here rather than
imported, because importing app.ml.cnn at all would pull in torch and defeat the point of
phase 2.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

N_CHANNELS = 24     # 12 variables x (mean, spread) - matches the live fetch
N_LEADS = 10


def peak_rss_mb() -> float:
    """Same measure as scripts/ingest_districts_chunked.py's peak_rss_mb - kept as a
    separate copy, not an import, because phase 2 must not import anything from this
    package's other modules that might drag in a heavier dependency by accident."""
    if sys.platform == "win32":
        import psutil
        return psutil.Process().memory_info().peak_wset / 1e6
    import resource
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if sys.platform == "darwin" else raw / 1024


# --------------------------------------------------------------------------- phase 1

def export_artifacts(out_dir: Path) -> None:
    """Build a real-shaped model and write everything phase 2 needs, as plain numpy -
    no torch object, no pickle of anything torch-specific, so phase 2 truly never needs
    to import torch to read them back."""
    import numpy as np
    import torch

    from app.ml.cnn import BustCNN, export_encoder
    from app.utils.india_districts import load_registry

    region_ids = [d.region_id for d in load_registry()]
    model = BustCNN(in_channels=N_CHANNELS, region_ids=region_ids).eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    export_encoder(model, out_dir / "encoder.onnx")

    head_state = {k: v.numpy() for k, v in model.head.state_dict().items()}
    np.savez(out_dir / "head.npz", **head_state)

    mat = model.pool.weight_matrix.coalesce()
    rows, cols = mat.indices().numpy()
    vals = mat.values().numpy()
    np.savez(
        out_dir / "pooling.npz",
        rows=rows, cols=cols, vals=vals,
        n_regions=np.int64(model.pool.n_regions),
        height=np.int64(model.pool.height), width=np.int64(model.pool.width),
        in_channels_doubled=np.int64(model.in_channels),
    )
    print(f"exported: encoder.onnx, head.npz, pooling.npz -> {out_dir}")
    print(f"  {model.pool.n_regions} districts, {model.in_channels} in-channels "
          f"(data+mask), {sum(p.numel() for p in model.parameters()):,} parameters")


# --------------------------------------------------------------------------- phase 2

def run_measurement(artifact_dir: Path) -> dict:
    """The actual measurement. Import order is the point: numpy and onnxruntime only,
    nothing from this repo's own package that could reach torch transitively."""
    base_rss = peak_rss_mb()

    import numpy as np
    import onnxruntime as ort
    after_import_rss = peak_rss_mb()

    head = np.load(artifact_dir / "head.npz")
    pooling = np.load(artifact_dir / "pooling.npz")
    rows, cols, vals = pooling["rows"], pooling["cols"], pooling["vals"]
    n_regions = int(pooling["n_regions"])
    height, width = int(pooling["height"]), int(pooling["width"])
    in_ch = int(pooling["in_channels_doubled"])
    w0, b0, w1, b1 = head["0.weight"], head["0.bias"], head["3.weight"], head["3.bias"]

    sess = ort.InferenceSession(str(artifact_dir / "encoder.onnx"),
                                providers=["CPUExecutionProvider"])
    after_session_rss = peak_rss_mb()

    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, in_ch, height, width)).astype(np.float32)
    extra = rng.normal(size=(1, n_regions, 1)).astype(np.float32)

    t0 = time.perf_counter()
    for _lead in range(N_LEADS):
        feats = sess.run(None, {"x": x})[0]                    # [1, C, H, W]
        c = feats.shape[1]
        flat = feats.reshape(1, c, height * width)
        contrib = flat[0][:, cols] * vals                      # [C, nnz]
        pooled = np.zeros((n_regions, c), dtype=np.float32)
        for ch in range(c):
            pooled[:, ch] = np.bincount(rows, weights=contrib[ch], minlength=n_regions)
        z = np.concatenate([pooled[None], extra], axis=-1)
        h = np.maximum(z @ w0.T + b0, 0.0)
        _logits = (h @ w1.T + b1).squeeze(-1)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    final_rss = peak_rss_mb()
    return {
        "base_rss_mb": base_rss,
        "after_numpy_onnxruntime_import_mb": after_import_rss,
        "after_session_create_mb": after_session_rss,
        "final_rss_mb": final_rss,
        "delta_total_mb": final_rss - base_rss,
        "delta_import_mb": after_import_rss - base_rss,
        "delta_session_mb": after_session_rss - after_import_rss,
        "delta_inference_mb": final_rss - after_session_rss,
        "leads_scored": N_LEADS,
        "elapsed_ms_for_all_leads": elapsed_ms,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child", action="store_true",
                    help=argparse.SUPPRESS)  # internal: phase 2's own invocation
    ap.add_argument("--artifacts", type=Path, default=None,
                    help="directory for exported artifacts (default: a temp dir)")
    args = ap.parse_args()

    if args.child:
        result = run_measurement(args.artifacts)
        print(json.dumps(result))
        return 0

    import tempfile

    artifact_dir = args.artifacts or Path(tempfile.mkdtemp(prefix="cnn_onnx_measure_"))
    export_artifacts(artifact_dir)

    print("\nmeasuring in a fresh subprocess (no torch import) ...")
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.measure_cnn_onnx_serving_memory",
         "--child", "--artifacts", str(artifact_dir)],
        cwd=BACKEND_DIR, capture_output=True, text=True,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return proc.returncode

    result = json.loads(proc.stdout.strip().splitlines()[-1])
    print("\n### CNN ONNX serving memory/timing (measured, not assumed)\n")
    print(f"| stage | peak RSS (MB) |")
    print(f"|---|---|")
    print(f"| baseline (interpreter up) | {result['base_rss_mb']:.1f} |")
    print(f"| + numpy/onnxruntime imported | {result['after_numpy_onnxruntime_import_mb']:.1f} "
          f"({result['delta_import_mb']:+.1f}) |")
    print(f"| + ONNX session created | {result['after_session_create_mb']:.1f} "
          f"({result['delta_session_mb']:+.1f}) |")
    print(f"| after {result['leads_scored']} lead days scored | {result['final_rss_mb']:.1f} "
          f"({result['delta_inference_mb']:+.1f}) |")
    print(f"\n**total delta: {result['delta_total_mb']:+.1f} MB, "
          f"{result['elapsed_ms_for_all_leads']:.0f} ms for {result['leads_scored']} "
          f"lead days.**")
    print(f"\nDocumented claim (app/ml/cnn.py export_encoder docstring, "
          f"CLAUDE.md measured facts): +51 MB, 490 ms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
