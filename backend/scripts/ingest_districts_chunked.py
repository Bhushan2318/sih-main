"""Ingest a district-grain reforecast year in cycle-sized chunks.

Why this exists
---------------
`scripts/ingest_backfill.py` hands a whole file to `parse_upload`, which loads it into
one DataFrame. That is fine for the 1.6 MB single-year samples it was written for. It is
not fine for a year at daily density: gefs_reforecast_india_2017.parquet is 675 MB and
12,154,500 wide rows, which melt to roughly 76 million long rows. Measured on a 3-cycle
subset (2026-09-10): 209,790 long rows and 30 s per cycle. One process cannot hold the
year.

So the source is split by cycle, each chunk is written to a temp parquet and ingested on
its own, and the chunk is deleted before the next one starts. Peak memory is set by the
chunk size rather than by the year.

That bounds the growth but does not make it small: one cycle at 666 districts is 33,300
wide rows melting to 209,790 long, and it peaks at 5.8 GB. Measured 2026-09-11 over 225
consecutive cycles of 2017 - 4,947 MB on the first, 5,761 MB by cycle 24, flat thereafter.
Give this script a 16 GB machine and do not run a dev server beside it; see
docs/known-issues.md.

Idempotent: a cycle already present in the target store is skipped, not re-ingested, so a
death at 80% resumes rather than restarting.

    python -m scripts.ingest_districts_chunked --year 2017 --months 11
    python -m scripts.ingest_districts_chunked --year 2017            # whole year
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

SAMPLES = BACKEND_DIR / "data" / "samples"


def peak_rss_mb() -> float:
    """macOS reports ru_maxrss in BYTES; Linux in kilobytes. Getting this wrong produces
    a number 1000x out, which is how a 2.08 GB peak got printed as '2079.97 GB' once.

    Windows has no `resource` module and no ru_maxrss equivalent - psutil's `peak_wset`
    (peak working set, bytes) is the closest match: the high-water mark, not current
    usage, which is what every other branch here reports and what the per-cycle ceiling
    in docs/known-issues.md was measured against.
    """
    # Both imports live here, not at module level. `resource` is POSIX-only, so a
    # top-level import made the script unimportable on Windows. The platform-guarded
    # top-level version that replaced it still broke the core install on Windows -
    # psutil is a training extra (requirements-train.txt) - and put `resource` on the
    # module on Mac and Linux, which is what tests/test_ingest_districts_chunked.py
    # checks against. Importing on first call keeps the module importable everywhere.
    if sys.platform == "win32":
        import psutil
        return psutil.Process().memory_info().peak_wset / 1e6
    import resource
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if sys.platform == "darwin" else raw / 1024


def resolve_source(year: int, source: str | None = None) -> Path:
    """Where to read forecast rows from for this year's ingest.

    Defaults to today's behaviour (`SAMPLES/gefs_reforecast_india_{year}.parquet`). An
    override exists because a district-scale year can collide with a filename the test
    suite's fixtures already own - 2019 is the first case: `tests/conftest.py` hardcodes
    `gefs_reforecast_india_2019.parquet` as the small 36-city legacy sample every test
    fixture depends on, so the real district-scale 2019 archive year has to live under a
    different name. A bare filename resolves under `SAMPLES`; a path with directories
    (relative or absolute) is used as given.
    """
    if source:
        p = Path(source)
        return p if p.is_absolute() or p.parent != Path(".") else SAMPLES / p
    return SAMPLES / f"gefs_reforecast_india_{year}.parquet"


def cycles_in_store() -> set:
    from app.storage import parquet_store
    df = parquet_store.read_dataset(value_types=["forecast"], columns=["init_date"],
                                    dedupe=False)
    if df.empty:
        return set()
    return set(pd.to_datetime(df["init_date"]).dt.normalize().unique())


def districts_for_cycle(cycle) -> int:
    from app.storage import parquet_store
    df = parquet_store.read_dataset(value_types=["forecast"], columns=["region_id"],
                                    init_dates=[pd.Timestamp(cycle).date()], dedupe=False)
    return df.region_id.nunique() if not df.empty else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--months", default=None, help="11, or 1-3, or 1,2,3. Default: all")
    ap.add_argument("--chunk-cycles", type=int, default=1,
                    help="cycles per ingest call. 1 is safest; raise only after measuring")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--source", default=None,
                    help="override the source parquet (default: "
                         "gefs_reforecast_india_<year>.parquet under data/samples). "
                         "A bare filename resolves there too; a path with directories "
                         "is used as given. Needed when a year's filename collides with "
                         "an existing sample, e.g. 2019's legacy 36-city test fixture.")
    args = ap.parse_args()

    months = None
    if args.months:
        months = set()
        for c in args.months.split(","):
            c = c.strip()
            if "-" in c:
                a, b = c.split("-", 1)
                months.update(range(int(a), int(b) + 1))
            elif c:
                months.add(int(c))

    src = resolve_source(args.year, args.source)
    if not src.exists():
        print(f"MISSING {src}", file=sys.stderr)
        return 1

    from app.storage import parquet_store
    print(f"source          : {src.name}  {src.stat().st_size/1e6:,.0f} MB")
    print(f"target store    : {parquet_store.CANONICAL_DIR}")

    inits = pd.read_parquet(src, columns=["init_date"])["init_date"]
    all_cycles = sorted(pd.to_datetime(inits).dt.normalize().unique())
    del inits
    if months:
        all_cycles = [c for c in all_cycles if pd.Timestamp(c).month in months]

    have = cycles_in_store()
    # A cycle already carrying the full district set is done. One carrying only the old
    # 36 is NOT - it must be re-ingested to pick up the other 630.
    todo = []
    for c in all_cycles:
        if c in have and districts_for_cycle(c) > 600:
            continue
        todo.append(c)
    print(f"cycles selected : {len(all_cycles)}   already complete: "
          f"{len(all_cycles)-len(todo)}   to ingest: {len(todo)}")
    if args.dry_run or not todo:
        print("nothing ingested" if not todo else "--dry-run: nothing ingested")
        return 0

    from app.db.base import init_db, SessionLocal
    from app.ingestion.pipeline import ingest_upload
    from app.live.orchestrator import FORECAST_MAPPINGS
    init_db()

    tmp_dir = BACKEND_DIR / "data" / "_ingest_chunks"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    t0, total = time.time(), 0

    for i in range(0, len(todo), args.chunk_cycles):
        batch = todo[i:i + args.chunk_cycles]
        # Push the cycle filter down into Parquet. Reading the whole 708 MB file per
        # chunk and filtering in pandas costs ~2 GB of peak memory and re-reads the year
        # once per cycle - 21 GB of I/O for a month. init_date is stored as datetime.date
        # objects, so the filter values must be dates, not Timestamps.
        want = [pd.Timestamp(c).date() for c in batch]
        df = pd.read_parquet(src, filters=[("init_date", "in", want)])
        chunk = tmp_dir / f"gefs_{args.year}_chunk_{i:04d}.parquet"
        df.to_parquet(chunk, index=False)
        n_districts, n_wide = df.region_id.nunique(), len(df)
        del df

        s = SessionLocal()
        try:
            res = ingest_upload(s, chunk, chunk.name,
                                confirmed_mappings=FORECAST_MAPPINGS,
                                verification_status=None)
            s.commit()
        finally:
            s.close()
        chunk.unlink(missing_ok=True)

        if res.status != "ingested":
            print(f"  REFUSED at {batch[0]}: status={res.status} - stopping",
                  file=sys.stderr)
            return 1
        total += res.row_count_ingested
        done = i + len(batch)
        el = time.time() - t0
        print(f"  {str(batch[0])[:10]}  {n_districts} districts  {n_wide:,} wide -> "
              f"{res.row_count_ingested:,} long   [{done}/{len(todo)}]  "
              f"{el:,.0f}s  peak {peak_rss_mb():,.0f} MB", flush=True)

    print(f"\ningested {total:,} long rows in {time.time()-t0:,.0f}s   "
          f"peak RSS {peak_rss_mb():,.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
