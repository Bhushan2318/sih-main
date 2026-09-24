"""Pack exactly what a serving box needs into one tarball.

Only the *current* model run goes in. The registry keeps every historical run on disk,
which is right for a workstation and wrong for a rolling release asset: the refresh
workflow restores the previous asset, trains, and repacks, so shipping the whole model
directory would grow the artifact by ~8 MB every six hours and never shrink.

    python -m scripts.package_for_deploy /tmp/sanket-data.tar.gz

Exits non-zero when there is no current run or a required serving path is missing, so CI
never publishes an artifact that would deploy an incomplete site.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import tarfile
from pathlib import Path

# What the box needs to answer a request, and nothing else: the canonical store it scores
# against, the geo index for region resolution, the MJO feature cache, the metadata db,
# and one model.  The release must be self-contained; do not rely on files left in the
# Docker build context.
EXTRA_PATHS = (
    "data/canonical",
    "data/geo",
    "data/mjo_omi_index.parquet",
    "data/summary.json",
    "metadata.db",
)
DIRECTORY_PATHS = {"data/canonical", "data/geo"}
SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _safe_tar_filter(member: tarfile.TarInfo) -> tarfile.TarInfo:
    """Refuse links and special files while packaging the release tree."""
    if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
        raise ValueError(f"unsafe release-tree member: {member.name!r}")
    return member


def _checkpoint_metadata_db(root: Path) -> None:
    """Fold SQLite WAL pages into metadata.db before packaging the main file.

    A release that copies only metadata.db can otherwise silently lose the most recent
    upload/profile rows still living in metadata.db-wal.  A busy checkpoint is a hard
    packaging failure rather than permission to publish a partial database.
    """
    db_path = root / "metadata.db"
    if not db_path.is_file():
        return
    with sqlite3.connect(db_path, timeout=30) as conn:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if row is not None and int(row[0]) != 0:
        raise RuntimeError(f"SQLite WAL checkpoint was busy (status={row[0]})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path, help="tarball to write")
    ap.add_argument("--root", type=Path, default=Path("."), help="backend directory")
    ap.add_argument("--no-compact", action="store_true",
                    help="skip removing superseded rows (they are dropped by default)")
    args = ap.parse_args()

    root: Path = args.root.resolve()
    data_root = root / "data"
    models_root = data_root / "models"
    if data_root.is_symlink() or models_root.is_symlink():
        print("refusing to package through a symlinked data/models tree", file=sys.stderr)
        return 1
    current = models_root / "current.json"
    if current.is_symlink() or not current.is_file():
        print("no data/models/current.json - nothing trained to publish", file=sys.stderr)
        return 1

    run_id = json.loads(current.read_text()).get("run_id")
    run_name = str(run_id) if run_id else ""
    if not SAFE_RUN_ID.fullmatch(run_name):
        print(f"current.json contains an unsafe run id: {run_id!r}", file=sys.stderr)
        return 1
    run_dir = models_root / run_name
    if run_dir.is_symlink() or not run_dir.is_dir() or not any(
        child.is_file() for child in run_dir.iterdir()
    ):
        print(
            f"current.json names {run_id!r}, which is not a populated model directory",
            file=sys.stderr,
        )
        return 1

    missing: list[str] = []
    for rel in EXTRA_PATHS:
        # summary.json is generated below; the other serving inputs must already exist.
        if rel == "data/summary.json":
            continue
        path = root / rel
        if path.is_symlink():
            missing.append(f"{rel} (symlink)")
        elif rel in DIRECTORY_PATHS and not path.is_dir():
            missing.append(rel)
        elif rel not in DIRECTORY_PATHS and not path.is_file():
            missing.append(rel)
    if missing:
        print(
            "refusing to publish an incomplete artifact; missing required path(s): "
            + ", ".join(missing),
            file=sys.stderr,
        )
        return 1

    from app.storage import parquet_store

    # Drop superseded rows before anything else measures or packs the store.
    #
    # Observations are re-pulled every verification window and each re-ingest writes a new
    # partition, so the store grew ~25k dead rows a day. `_dedupe` collapsed them at read
    # time, so nothing was ever wrong - it was just paid for repeatedly: in every read, in
    # an artifact downloaded on every container start, and in the memory of a box with
    # 512 MB and no room to spare.
    #
    # Doing it here, on every publish, is also what stops them coming back: the runner's
    # store is rebuilt from the previous artifact each run, so compacting the artifact
    # compacts the input to the next run too.
    if not args.no_compact:
        st = parquet_store.compact_store()
        print(f"compacted: {st['rows_before']:,} -> {st['rows_after']:,} rows "
              f"({st['removed']:,} superseded rows dropped)")

    # Precompute the store summary here rather than on the serving box: it needs a full
    # deduplication of every row (~300 MB), which is fine on a CI runner and is not fine
    # on a 512 MB instance that also has to score a cycle.
    summary = parquet_store.write_summary_cache()
    print(f"summary cached: {summary['total_rows']:,} rows, {summary['regions']} regions")
    summary_path = root / "data" / "summary.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        print("refusing to publish: data/summary.json was not generated as a regular file", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _checkpoint_metadata_db(root)
    with tarfile.open(args.out, "w:gz") as tar:
        # arcname keeps paths relative to the backend dir so the image can untar at /app
        tar.add(current, arcname="data/models/current.json", filter=_safe_tar_filter)
        tar.add(run_dir, arcname=f"data/models/{run_name}", filter=_safe_tar_filter)
        for rel in EXTRA_PATHS:
            path = root / rel
            tar.add(path, arcname=rel, filter=_safe_tar_filter)

    size_mb = args.out.stat().st_size / 1_000_000
    print(f"packed {args.out} ({size_mb:.1f} MB) with model run {run_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
