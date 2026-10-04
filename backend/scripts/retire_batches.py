"""Retire superseded batches: move them out of the store, write MOVED.md, mark them 'retired'.

For the retrain, after the estimator-v2 observation years are ingested:

    python -m scripts.retire_batches --pattern "era5_cds_district_observations_india_*.parquet" \\
        --out data/retired_era5_v1 --reason "superseded by estimator v2"          # dry run
    ... the same with --apply

The store's dedupe keeps the newest row per key, so a newer batch shadows an older one only
where it has a value; retiring removes the older rows outright. A 'retired' batch whose files
come back is refused by `parquet_store.assert_no_excluded_batches` before any training frame
is built. Only batches with status 'ingested' are selected; the pattern matches the file name
the batch was ingested from (`upload_batch.original_filename`).
"""
from __future__ import annotations

import argparse
import fnmatch
import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.db.base import get_session
from app.db.models import UploadBatch
from app.storage import parquet_store


def select_batches(patterns: list[str]) -> list[UploadBatch]:
    with get_session() as session:
        rows = session.query(UploadBatch).filter(UploadBatch.status == "ingested").all()
        session.expunge_all()
    return sorted((b for b in rows
                   if any(fnmatch.fnmatch(b.original_filename, p) for p in patterns)),
                  key=lambda b: (b.original_filename, b.id))


def retire(patterns: list[str], out_dir: Path, apply: bool, reason: str) -> list[str]:
    """Batch ids retired (or that would be, without `apply`)."""
    chosen = select_batches(patterns)
    present = parquet_store.present_batch_ids()
    for b in chosen:
        where = "in the store" if b.id in present else "not in the store"
        print(f"{b.id}  {b.original_filename}  ({where})")
    print(f"{len(chosen)} batch(es) match {patterns}")
    if not apply or not chosen:
        if chosen:
            print("dry run: nothing moved (add --apply)")
        return [b.id for b in chosen]

    out_dir = Path(out_dir)
    clashes = [out_dir / f"batch_id={b.id}" for b in chosen
               if (out_dir / f"batch_id={b.id}").exists()]
    if clashes:
        raise FileExistsError(f"already in {out_dir}: {', '.join(p.name for p in clashes)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    lines = [f"# Retired {stamp}", "", f"Reason: {reason}", "",
             "To restore: move each `batch_id=...` folder back into the canonical store and",
             "set its `upload_batch.status` back to 'ingested'. Restoring the folder alone",
             "is refused by the trainer while the status says 'retired'.", ""]
    for b in chosen:
        src = parquet_store.CANONICAL_DIR / f"batch_id={b.id}"
        if src.exists():
            shutil.move(str(src), str(out_dir / src.name))
        lines.append(f"- `{b.id}` {b.original_filename}")
    with open(out_dir / "MOVED.md", "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    with get_session() as session:
        for b in chosen:
            session.get(UploadBatch, b.id).status = "retired"
    print(f"retired {len(chosen)} batch(es) -> {out_dir}")
    return [b.id for b in chosen]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", action="append", required=True,
                    help="file-name pattern of the batches to retire (repeatable)")
    ap.add_argument("--out", type=Path, required=True, help="where the folders go")
    ap.add_argument("--reason", required=True)
    ap.add_argument("--apply", action="store_true", help="without it, only list")
    args = ap.parse_args()
    retire(args.pattern, args.out, args.apply, args.reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
