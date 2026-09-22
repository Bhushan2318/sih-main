"""Install a published serving model into this checkout, and make it current.

    python -m scripts.install_serving_model --bundle /tmp/sanket-model.tar.gz \
        --manifest /tmp/model.json

Used by the refresh workflow: when a model is pinned (the `serving-model` release), CI
serves that model instead of training its own, so a model trained on seventeen years of
reforecast - which needs a GPU and far more memory than a runner has - can be what the
site serves while CI keeps refreshing the forecast data every six hours.

Refuses rather than installs a bundle whose bytes do not match the sha256 its manifest
declares, or whose run id does not match. A half-right model is worse than the previous
one: the site would look healthy and serve something nobody checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

# What a run directory must carry to be served. shap_summary.parquet is not optional:
# the region panel's "what drove this prediction" reads it, and a model without it
# serves an empty explanation rather than failing loudly.
REQUIRED_FILES = ("classifier.json", "feature_columns.json", "thresholds.json",
                  "historical_bust_freq.json", "metrics.json", "manifest.json",
                  "shap_summary.parquet")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def missing_files(run_dir: Path) -> list:
    """Required files this run directory lacks, plus a note if it has no regressor."""
    gaps = [n for n in REQUIRED_FILES if not (run_dir / n).is_file()]
    if not list(run_dir.glob("*_regressor.json")):
        gaps.append("*_regressor.json")
    return gaps


def install(bundle: Path, manifest_path: Path, root: Path) -> str:
    manifest = json.loads(manifest_path.read_text())
    run_id = manifest["run_id"]
    want = manifest["sha256"]
    got = sha256_of(bundle)
    if got != want:
        raise ValueError(f"bundle sha256 {got} does not match the {want} its manifest "
                         f"declares - refusing to install")

    with tarfile.open(bundle, "r:gz") as tar:
        names = tar.getnames()
        prefix = f"data/models/{run_id}/"
        # The run directory's own entry is the one name without the trailing slash.
        strays = [n for n in names if not n.startswith(prefix) and n != prefix.rstrip("/")]
        if strays:
            raise ValueError(f"bundle writes outside {prefix}: {strays[:5]} - refusing")
        tar.extractall(root)

    run_dir = root / "data" / "models" / run_id
    gaps = missing_files(run_dir)
    if gaps:
        raise ValueError(f"{run_id} is missing {gaps} - refusing to make it current")

    # set_current writes current.json atomically, the same way training does.
    from app.ml import registry
    registry.set_current(run_id)
    return run_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=BACKEND_DIR,
                    help="backend directory the bundle's data/models path is relative to")
    args = ap.parse_args()
    try:
        run_id = install(args.bundle, args.manifest, args.root.resolve())
    except (ValueError, KeyError, OSError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
