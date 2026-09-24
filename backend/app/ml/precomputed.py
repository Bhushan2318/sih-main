"""A cycle scored in CI, read back on the serving box.

CLAUDE.md's first diagram splits this system because "the serving box is killed, not
throttled" at 512 MB: CI trains, Render serves. Scoring every district of a cycle is
training-shaped work that had quietly crept onto the serving side. It cost little while
the live feed sampled 36 city points; at all 666 districts it does not fit.

Measured on a real cycle, 666 districts x 10 lead days, 6,660 events:

  scoring it on the box            1,406 MB peak RSS
  reading the precomputed answer     105 MB total process RSS, 38 MB growth
  the artifact                      3.96 MB zstd (2.56 events + 1.40 per-variable)

The model tarball that already ships is 11.9 MB, so this is not a meaningful addition to
the deploy. It is the same move as reading forecast cycles from Parquet footers rather
than scanning a column of every row - +253 MB became +2 MB - applied one level up.

WHY THE KEY IS (run_id, init_date) AND NOT THE STORE FINGERPRINT
`parquet_store.store_fingerprint` is built from directory mtimes. Those cannot agree
between the CI runner that writes this artifact and the box that reads it, so keying on it
would never hit. A scored cycle is identified by the model that scored it and the cycle it
scored; the run_id half is what stops a newly promoted model from serving the previous
one's answers. Consistency with the store is by construction rather than by checksum: the
same CI run ingests the cycle, scores it and packages both, so they ship together or not
at all.

WHAT IS NOT PRECOMPUTED
Only the cycles CI writes, in practice the latest. Replay asks for arbitrary historical
cycles on request, and those still score live - `score_cycle` falls through when there is
no artifact. On a 666-district store that path is as expensive as the one this avoids, so
Replay remains a real memory risk on the serving box and is written up in
docs/known-issues.md rather than silently fixed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.config import settings
from app.db.base import resolve_path

DIR_NAME = "scored_cycles"
_EVENTS = "events.parquet"
_PER_VARIABLE = "per_variable.parquet"
_META = "meta.json"


def default_dir() -> Path:
    """Where the packaged artifacts live. A function, not a constant, so a test can point
    it somewhere else without reaching into module state."""
    return Path(resolve_path(settings.data_dir)) / "analysis" / DIR_NAME


def cycle_dir(base: Path, run_id: str, init_date) -> Path:
    return Path(base) / f"{run_id}__{pd.Timestamp(init_date).strftime('%Y%m%d')}"


def write_scored_cycle(scored, base: Path | None = None) -> Path:
    """Write one scored cycle. `meta.json` is written LAST, on purpose.

    A run killed part-way through then leaves a directory the reader rejects, rather than
    one that reads as a whole cycle with some districts missing. Missing never becomes
    partial (CLAUDE.md rule 3) - a map quietly short of districts says nothing about
    itself, which is worse than no map.
    """
    d = cycle_dir(base or default_dir(), scored.run_id, scored.init_date)
    d.mkdir(parents=True, exist_ok=True)
    (d / _META).unlink(missing_ok=True)

    scored.events.to_parquet(d / _EVENTS, index=False, compression="zstd")
    scored.per_variable.to_parquet(d / _PER_VARIABLE, index=False, compression="zstd")
    (d / _META).write_text(json.dumps({
        "run_id": scored.run_id,
        "init_date": pd.Timestamp(scored.init_date).strftime("%Y-%m-%d"),
        "n_rows_scored": int(scored.n_rows_scored),
        "n_events": int(len(scored.events)),
    }))
    return d


def read_scored_cycle(run_id: str, init_date, base: Path | None = None):
    """The scored cycle for this (model, cycle), or None if there is not a complete one."""
    from app.ml.inference import ScoredCycle

    d = cycle_dir(base or default_dir(), run_id, init_date)
    meta_path = d / _META
    if not (meta_path.exists() and (d / _EVENTS).exists() and (d / _PER_VARIABLE).exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
        events = pd.read_parquet(d / _EVENTS)
        per_variable = pd.read_parquet(d / _PER_VARIABLE)
    except Exception:  # noqa: BLE001 - an unreadable artifact is a miss, never a crash
        return None
    if meta.get("run_id") != run_id or len(events) != meta.get("n_events"):
        return None
    return ScoredCycle(
        run_id=run_id,
        init_date=pd.Timestamp(init_date).normalize(),
        events=events,
        per_variable=per_variable,
        n_rows_scored=int(meta.get("n_rows_scored", len(events))),
    )
