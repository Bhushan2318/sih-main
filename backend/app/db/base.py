from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

BACKEND_DIR = Path(__file__).resolve().parents[2]


def resolve_path(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (BACKEND_DIR / p)


DB_PATH = resolve_path(settings.db_path)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Real crash 2026-09-18: two local ingest processes (forecast ingest, observation
# ingest) writing to this same file at once hit "database is locked" and one of them
# died outright - SQLite's default is to fail immediately, not wait, when a writer
# finds the file locked. `timeout` makes Python's sqlite3 driver retry for up to 30 s
# before raising, and WAL mode (set once per connection, persists in the db file)
# lets readers and a writer proceed together instead of blocking each other, which is
# most of what was colliding here - only two actual writers ever contend.
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False, "timeout": 30},
    future=True,
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from app.db import models  # noqa: F401  (register mappers before create_all)

    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
