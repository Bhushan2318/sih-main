"""Publish one finished run as the model the site serves.

    python -m scripts.publish_serving_model --run-id run_20260922T043925Z [--deploy]

Nothing here decides on its own that a model is good enough. It runs the SAME promotion
gate the training pipeline uses (`train_pipeline._promotion_decision`) against whatever
the site currently serves, and refuses when the gate refuses.

What it checks, in order, before a single byte is uploaded:

1. The run directory carries everything a serving box needs (install_serving_model's
   REQUIRED_FILES, including shap_summary.parquet).
2. The promotion gate, against the model the live artifact currently names.
3. The API itself: the real published data bundle, this model dropped in, and the four
   endpoints the dashboard calls - including a region's `top_factors`, which is the SHAP
   explanation and a shipped feature. A model that loads but explains nothing is refused
   here rather than discovered on the site.

Then it uploads the bundle and its manifest to the `serving-model` release, and with
--deploy dispatches the refresh workflow, which repacks and fires the Render hook.

To go back to CI-trained models: `gh release delete serving-model`. The workflow falls
back on its own, because the pinned model is a release that either exists or does not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from scripts.install_serving_model import missing_files, sha256_of  # noqa: E402

RELEASE_TAG = "serving-model"
BUNDLE_NAME = "sanket-model.tar.gz"
MANIFEST_NAME = "model.json"
DATA_TAG = "data-latest"
DATA_ASSET = "sanket-data.tar.gz"


def _gh(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], check=check, capture_output=True, text=True)


def pack(run_dir: Path, out: Path) -> Path:
    """The run directory alone, at the path a serving checkout expects it."""
    run_id = run_dir.name
    with tarfile.open(out, "w:gz") as tar:
        tar.add(run_dir, arcname=f"data/models/{run_id}")
    return out


def fetch_live_bundle(dest: Path) -> "tuple[Path, str | None]":
    """The published data artifact, and the run id it currently serves."""
    _gh("release", "download", DATA_TAG, "-p", DATA_ASSET, "-D", str(dest), "--clobber")
    with tarfile.open(dest / DATA_ASSET, "r:gz") as tar:
        tar.extractall(dest)
    current = dest / "data" / "models" / "current.json"
    return dest, (json.loads(current.read_text()).get("run_id") if current.is_file() else None)


def gate_decision(new_metrics: dict, live_root: Path, live_run_id: "str | None") -> tuple:
    """The training pipeline's own promotion gate, asked about this model against the
    model the site serves. The gate reads the registry, so the registry is pointed at the
    downloaded live bundle for the length of the question - the gate itself is untouched."""
    from app.ml import registry
    from app.ml.train_pipeline import _promotion_decision

    saved_dir, saved_json = registry.MODEL_DIR, registry.CURRENT_JSON
    try:
        registry.MODEL_DIR = live_root / "data" / "models"
        registry.CURRENT_JSON = registry.MODEL_DIR / "current.json"
        return _promotion_decision(new_metrics)
    finally:
        registry.MODEL_DIR, registry.CURRENT_JSON = saved_dir, saved_json


def serving_check(live_root: Path, run_dir: Path) -> dict:
    """Boot the API against the real published data with this model installed, and call
    what the dashboard calls. Returns the observed facts; raises on anything not served."""
    import shutil

    run_id = run_dir.name
    target = live_root / "data" / "models" / run_id
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(run_dir, target)
    (live_root / "data" / "models" / "current.json").write_text(
        json.dumps({"run_id": run_id, "set_at": "serving-check"}))

    env = {
        "DATA_DIR": str(live_root / "data"),
        "MODEL_DIR": str(live_root / "data" / "models"),
        "CANONICAL_DIR": str(live_root / "data" / "canonical"),
        "DB_PATH": str(live_root / "metadata.db"),
        "LIVE_INGEST_ENABLED": "false",
        "WARM_CACHES_ON_STARTUP": "false",
        # The serving box has no GPU, so neither may this check. A pooled model is trained
        # with device="cuda" and XGBoost keeps that in the saved JSON, so checking it on a
        # machine that has a GPU would exercise a path the 512 MB box never takes and hide
        # whatever it does instead. It also keeps the check off a GPU a training run may be
        # using: contention there cost a variable a restart on 2026-09-22, when a
        # concurrent finalize left too little VRAM for the trainer to allocate.
        "CUDA_VISIBLE_DEVICES": "",
    }
    # A subprocess, because this process has already imported app.config with different
    # paths, and settings are read once at import.
    checker = SCRIPT_DIR / "_serving_check_worker.py"
    proc = subprocess.run([sys.executable, str(checker), "--run-id", run_id],
                          env={**os.environ, **env}, capture_output=True, text=True,
                          cwd=str(BACKEND_DIR))
    if proc.returncode != 0:
        raise RuntimeError(f"serving check failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dry-run", action="store_true", help="check and pack, upload nothing")
    ap.add_argument("--deploy", action="store_true",
                    help="after uploading, dispatch the refresh workflow so the site picks it up")
    args = ap.parse_args()

    from app.ml import registry

    run_dir = registry.run_dir(args.run_id)
    if not run_dir.is_dir():
        print(f"no such run: {run_dir}", file=sys.stderr)
        return 1
    gaps = missing_files(run_dir)
    if gaps:
        print(f"{args.run_id} is missing {gaps} - not publishable. For a pooled run, "
              f"`python -m scripts.train_pooled --finalize {args.run_id}` writes the "
              f"SHAP summary.", file=sys.stderr)
        return 1

    metrics = registry.load_metrics(args.run_id) or {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        live_root, live_run_id = fetch_live_bundle(td / "live")
        promote, why = gate_decision(metrics, live_root, live_run_id)
        print(f"currently served: {live_run_id}")
        print(f"gate: {why}")
        if not promote:
            print("refusing to publish: the promotion gate said no", file=sys.stderr)
            return 2

        observed = serving_check(live_root, run_dir)
        print(f"serving check: {json.dumps(observed)}")

        bundle = pack(run_dir, td / BUNDLE_NAME)
        manifest = {
            "run_id": args.run_id,
            "sha256": sha256_of(bundle),
            "bytes": bundle.stat().st_size,
            "gate": why,
            "held_out": ((metrics.get("classifier") or {}).get("test")
                         or (metrics.get("classifier") or {}).get("val") or {}),
            "replaces": live_run_id,
        }
        (td / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest, indent=2))
        if args.dry_run:
            print("dry run: nothing uploaded")
            return 0

        _gh("release", "view", RELEASE_TAG, check=False)
        if _gh("release", "view", RELEASE_TAG, check=False).returncode != 0:
            _gh("release", "create", RELEASE_TAG, "--title", "Pinned serving model",
                "--notes", "The model the site serves. Delete this release to return to "
                           "CI-trained models.")
        _gh("release", "upload", RELEASE_TAG, str(bundle), str(td / MANIFEST_NAME), "--clobber")
        _gh("release", "edit", RELEASE_TAG, "--notes",
            f"Serving model `{args.run_id}`\n\n```json\n{json.dumps(manifest, indent=2)}\n```")
        print(f"published {args.run_id} to the {RELEASE_TAG} release")

        if args.deploy:
            _gh("workflow", "run", "refresh-data.yml")
            print("dispatched refresh-data.yml; it will repack and trigger the deploy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
