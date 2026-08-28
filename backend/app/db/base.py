"""SQLAlchemy engine, session factory, and declarative base for the metadata DB.

The metadata DB (SQLite) holds *lineage*, not measurements: which files were uploaded,
how their columns were mapped, and which header fingerprints have a confirmed mapping.
The measurements themselves live in the parquet store (see storage/parquet_store.py).

Path handling: ``settings`` carries paths relative to ``backend/`` so they can be
overridden from ``.env`` without code changes. Code, though, may be imported with the
process CWD at the repo root (scripts) or at ``backend/`` (the API). ``resolve_path``
anchors any relative settings path to ``backend/`` so both work.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

BACKEND_DIR = Path(__file__).resolve().parents[2]  # .../backend


def resolve_path(path: Path | str) -> Path:
    """Absolute path for a (possibly relative) settings path, anchored at ``backend/``."""
    p = Path(path)
    return p if p.is_absolute() else (BACKEND_DIR / p)


DB_PATH = resolve_path(settings.db_path)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# check_same_thread=False: FastAPI runs handlers in a threadpool; the BackgroundTask
# retrain (Phase 4) also touches the DB from a worker thread. SQLite is fine with this
# as long as writes are serialised, which they are here (one ingest at a time).
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Create any missing tables. Safe to call on every startup."""
    from app.db import models  # noqa: F401  (register mappers before create_all)

    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Iterator[Session]:
    """Session context manager for non-request callers (pipeline, CLI, tests)."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
