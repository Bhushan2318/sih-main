"""Pack exactly what a serving box needs into one tarball.

Only the *current* model run goes in. The registry keeps every historical run on disk,
which is right for a workstation and wrong for a rolling release asset: the refresh
workflow restores the previous asset, trains, and repacks, so shipping the whole model
directory would grow the artifact by ~8 MB every six hours and never shrink.

    python -m scripts.package_for_deploy /tmp/sanket-data.tar.gz

Exits non-zero when there is no current run, so CI never publishes an artifact that would
deploy a site with no model.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

# What the box needs to answer a request, and nothing else: the canonical store it scores
# against, the geo index for region resolution, the metadata db, and one model.
EXTRA_PATHS = ("data/canonical", "data/geo", "metadata.db")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path, help="tarball to write")
    ap.add_argument("--root", type=Path, default=Path("."), help="backend directory")
    args = ap.parse_args()

    root: Path = args.root.resolve()
    current = root / "data" / "models" / "current.json"
    if not current.is_file():
        print("no data/models/current.json - nothing trained to publish", file=sys.stderr)
        return 1

    run_id = json.loads(current.read_text()).get("run_id")
    run_dir = root / "data" / "models" / str(run_id)
    if not run_id or not run_dir.is_dir():
        print(f"current.json names {run_id!r}, which is not on disk", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.out, "w:gz") as tar:
        # arcname keeps paths relative to the backend dir so the image can untar at /app
        tar.add(current, arcname="data/models/current.json")
        tar.add(run_dir, arcname=f"data/models/{run_id}")
        for rel in EXTRA_PATHS:
            path = root / rel
            if path.exists():
                tar.add(path, arcname=rel)
            else:
                print(f"note: {rel} missing, skipped", file=sys.stderr)

    size_mb = args.out.stat().st_size / 1_000_000
    print(f"packed {args.out} ({size_mb:.1f} MB) with model run {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
