"""Test fixtures.

The metadata DB and the canonical parquet store are redirected to a throwaway temp
directory *before* any ``app`` module is imported, by setting the same env vars the
production config reads. Every test then gets a clean store via the ``fresh_store``
fixture.

Sample data: tests assert against the real files. ``SAMPLE_DIR`` resolves to
``$FORECASTGUARD_SAMPLE_DIR`` -> ``~/Desktop/data`` -> ``backend/data/samples``. The two
generated samples (GEFS reforecast + ERA5) always live in ``backend/data/samples`` and
have their own path constants.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="forecastguard-test-"))
os.environ.setdefault("DATA_DIR", str(_TMP))
os.environ["DB_PATH"] = str(_TMP / "metadata.db")
os.environ["CANONICAL_DIR"] = str(_TMP / "canonical")
os.environ["RAW_UPLOAD_DIR"] = str(_TMP / "raw_uploads")
# MODEL_DIR must be redirected too: the API tests wipe the model directory between
# cases, and without this they would delete the developer's real trained runs under
# backend/data/models/.
os.environ["MODEL_DIR"] = str(_TMP / "models")

BACKEND_DIR = Path(__file__).resolve().parents[1]
GENERATED_SAMPLES = BACKEND_DIR / "data" / "samples"
GEFS_CSV = GENERATED_SAMPLES / "gefs_reforecast_india_2019.csv"
ERA5_CSV = GENERATED_SAMPLES / "era5_observations_india_2019.csv"


def _resolve_sample_dir() -> Path:
    for cand in (
        os.environ.get("FORECASTGUARD_SAMPLE_DIR"),
        str(Path.home() / "Desktop" / "data"),
        str(GENERATED_SAMPLES),
    ):
        if cand and Path(cand).is_dir():
            return Path(cand)
    return GENERATED_SAMPLES


SAMPLE_DIR = _resolve_sample_dir()


def iter_sample_files():
    exts = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".parquet"}
    for root in {SAMPLE_DIR, GENERATED_SAMPLES}:
        for p in sorted(Path(root).rglob("*")):
            if p.is_file() and p.suffix.lower() in exts and "_gefs_parts" not in p.parts:
                yield p


def find_sample(*name_fragments: str) -> Path | None:
    for p in iter_sample_files():
        if all(frag.lower() in p.name.lower() for frag in name_fragments):
            return p
    return None


@pytest.fixture
def fresh_store():
    """A clean metadata DB + empty canonical store for one test."""
    from app.db.base import engine, init_db
    from app.db.models import Base
    from app.storage import parquet_store

    Base.metadata.drop_all(engine)
    shutil.rmtree(parquet_store.CANONICAL_DIR, ignore_errors=True)
    shutil.rmtree(_TMP / "raw_uploads", ignore_errors=True)
    init_db()
    yield
    Base.metadata.drop_all(engine)
    shutil.rmtree(parquet_store.CANONICAL_DIR, ignore_errors=True)


@pytest.fixture
def session(fresh_store):
    from app.db.base import SessionLocal

    s = SessionLocal()
    try:
        yield s
        s.commit()
    finally:
        s.close()


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP, ignore_errors=True)
