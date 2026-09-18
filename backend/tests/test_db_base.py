"""Real crash 2026-09-18: two local processes ingesting into the same SQLite
metadata.db at once ("database is locked") killed one of them outright, mid-run,
losing real ingestion progress. app/db/base.py now sets WAL journal mode and a 30 s
busy timeout so a writer waits for the other to finish instead of failing immediately.
"""
from __future__ import annotations

import sqlite3
import threading

from app.db.base import DB_PATH, engine


def test_sqlite_pragmas_are_set_for_concurrent_writers():
    with engine.connect() as conn:
        journal_mode = conn.exec_driver_sql("PRAGMA journal_mode").scalar()
        busy_timeout = conn.exec_driver_sql("PRAGMA busy_timeout").scalar()
    assert journal_mode == "wal"
    assert busy_timeout == 30000


def test_concurrent_writers_do_not_raise_database_is_locked():
    """Two threads, each holding a real write transaction open on the same file the
    engine points at, used to reproduce 'database is locked' immediately without a
    busy timeout. With WAL + busy_timeout, the second writer waits instead of raising."""
    errors: list[Exception] = []

    def _write(table_suffix: str) -> None:
        try:
            conn = sqlite3.connect(str(DB_PATH), timeout=30)
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute(f"CREATE TABLE IF NOT EXISTS _lock_probe_{table_suffix} (n INTEGER)")
            conn.execute("BEGIN IMMEDIATE")
            for i in range(50):
                conn.execute(
                    f"INSERT INTO _lock_probe_{table_suffix} VALUES (?)", (i,))
            conn.commit()
            conn.close()
        except Exception as exc:  # noqa: BLE001 - captured for the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=_write, args=(str(i),)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"concurrent writers raised: {errors}"
