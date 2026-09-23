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
EXTRA_PATHS = ("data/canonical", "data/geo", "data/summary.json", "metadata.db",
               "data/analysis/scored_cycles")


def _replay_window() -> int:
    """How many cycles Replay offers, read from Replay rather than restated here.

    Two constants that must agree and are written down twice eventually disagree, and this
    pair fails silently: Replay would offer a cycle nothing had precomputed, and the box
    would score it - which at 666 districts is the 1,406 MB path this whole mechanism
    exists to avoid, reached by a click instead of a page load.
    """
    from app.services.replay_service import _MAX_CYCLES

    return _MAX_CYCLES


PRECOMPUTE_CYCLES = _replay_window()


def _available_cycles():
    from app.ml import inference

    return inference.available_cycles()


def _score_and_write(state, init) -> int:
    """Score one cycle and write it; returns bytes written. Raises if it cannot score."""
    from app.ml import inference, precomputed

    scored = inference.score_cycle(state, init_date=init)
    if scored is None:
        raise RuntimeError("no scoreable rows")
    out = precomputed.write_scored_cycle(scored)
    return sum(f.stat().st_size for f in out.glob("*"))


def precompute_cycles(state) -> tuple:
    """Score the cycles the box must not score itself: the national map's latest, and
    every cycle Replay can offer.

    Sequential and with the score cache cleared between cycles. `inference._score_cache`
    holds up to 24, so without clearing, the whole window accumulates in the packaging
    process - ten frames of 6,660 events. The runner has 16 GB and would survive it; the
    point is that this loop is the one that runs against a larger window later.

    A cycle that cannot be scored is skipped, not fatal. It simply has no artifact and is
    scored live, which is the behaviour that existed before any of this.
    """
    from app.ml import inference

    written, total = 0, 0
    for init in _available_cycles()[:PRECOMPUTE_CYCLES]:
        inference.invalidate_caches()
        try:
            total += _score_and_write(state, init)
        except Exception as exc:  # noqa: BLE001 - one bad cycle must not lose the rest
            print(f"note: {pd_date(init)} not precomputed ({exc})", file=sys.stderr)
            continue
        written += 1
    inference.invalidate_caches()
    return written, total


def pd_date(init) -> str:
    return str(getattr(init, "date", lambda: init)())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path, help="tarball to write")
    ap.add_argument("--root", type=Path, default=Path("."), help="backend directory")
    ap.add_argument("--no-compact", action="store_true",
                    help="skip removing superseded rows (they are dropped by default)")
    ap.add_argument("--no-precompute", action="store_true",
                    help="skip scoring the latest cycle into the package; the serving box "
                         "will then score it itself, which does not fit at 666 districts")
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

    # Score the cycle here too, for the same reason and with more of it at stake. At 666
    # districts scoring one cycle peaks at 1,406 MB; the box is killed at 512 with ~388
    # already in use. The answer is 3.96 MB, so it travels in this tarball and the box
    # never scores. See app/ml/precomputed.py.
    if not args.no_precompute:
        from app.ml import inference
        state = inference.load_model_state(run_id)
        if state is None:
            print(f"note: {run_id} would not load, no cycle precomputed", file=sys.stderr)
        else:
            n, nbytes = precompute_cycles(state)
            print(f"precomputed {n} cycle(s) - the national map's latest and Replay's "
                  f"window of {PRECOMPUTE_CYCLES} - {nbytes / 1_048_576:.2f} MB")

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
