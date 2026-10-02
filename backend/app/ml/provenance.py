"""What produced a model: the code, the libraries, the cached data and the fit settings.

Two uses. A finished variable's checkpoint must never be reused by a retrain that would
have trained it differently (pooled_training._variable_checkpoint_path keys on these
fingerprints). And a published run must say what made it, so a number on the site can be
traced back to a commit and a data state rather than reconstructed from memory.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]

# The code a regressor's or classifier's result depends on: features, models, contracts.
# Hashed by content, so an uncommitted edit counts and a docs-only commit does not.
TRAINING_SOURCES = [
    BACKEND_DIR / "app" / "features",
    BACKEND_DIR / "app" / "ml",
    BACKEND_DIR / "app" / "contracts.py",
]


def source_fingerprint() -> str:
    """sha256 over the training code's .py files, in a stable order."""
    h = hashlib.sha256()
    files = []
    for root in TRAINING_SOURCES:
        root = Path(root)
        files.extend(sorted(root.rglob("*.py")) if root.is_dir() else [root])
    for f in files:
        if f.exists():
            h.update(str(f.relative_to(f.parents[len(f.parents) - 1])).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def cache_fingerprint(cached_paths: dict) -> dict:
    """year -> sha256 of a cached year's Parquet footer and size.

    The footer carries the schema, the row count and every row group's statistics, so a
    rebuilt year almost always changes it, and reading it costs kilobytes, not the
    gigabytes hashing the file would.
    """
    import pyarrow.parquet as pq

    out = {}
    for year, path in cached_paths.items():
        path = Path(path)
        md = pq.ParquetFile(path).metadata
        h = hashlib.sha256()
        h.update(str(path.stat().st_size).encode())
        h.update(str(md.num_rows).encode())
        h.update(md.schema.to_arrow_schema().to_string().encode())
        for i in range(md.num_row_groups):
            rg = md.row_group(i)
            h.update(str(rg.num_rows).encode())
            for j in range(rg.num_columns):
                st = rg.column(j).statistics
                if st is not None and st.has_min_max:
                    h.update(repr((st.min, st.max, st.null_count)).encode())
        out[str(year)] = h.hexdigest()
    return out


def library_versions() -> dict:
    import numpy
    import pandas
    import pyarrow
    import sklearn
    import xgboost

    return {"xgboost": xgboost.__version__, "numpy": numpy.__version__,
            "pandas": pandas.__version__, "pyarrow": pyarrow.__version__,
            "sklearn": sklearn.__version__}


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=BACKEND_DIR, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - no git (a copied tree) is recorded, not fatal
        return "unknown"


def run_provenance(**fit_settings) -> dict:
    """The manifest's record of what produced a run. JSON-serialisable."""
    status = _git("status", "--porcelain", "--", ".")
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": None if status == "unknown" else bool(status),
        "source_sha256": source_fingerprint(),
        "libraries": library_versions(),
        "python": platform.python_version(),
        **fit_settings,
    }


def context_digest(context: dict) -> bytes:
    """A canonical encoding of a checkpoint context: key order never changes the key."""
    return json.dumps(context, sort_keys=True, default=str).encode()
