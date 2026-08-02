# app/services/job_store.py
"""
SQLite-backed job store.

An in-memory dict loses every job on restart and breaks the moment you run
more than one worker process. SQLite gets you persistence and cross-process
visibility with zero extra infrastructure to run — a good fit for a
single-instance deployment. Swap for Redis/Postgres if you ever run this
with multiple app instances behind a load balancer.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path

from app.core.config import settings


class JobStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


DB_PATH = Path(settings.JOBS_DB_PATH)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    status TEXT NOT NULL,
    chunk_count INTEGER,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(_SCHEMA)


def create_job(job_id: str, file_path: str) -> None:
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO jobs "
            "(job_id, file_path, status, chunk_count, error, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (job_id, file_path, JobStatus.PENDING.value, None, None, now, now),
        )


def update_job(job_id: str, **kwargs) -> bool:
    """Returns True if a row was actually updated, False if job_id no
    longer exists (SQLite UPDATE on zero matching rows doesn't raise —
    it's a silent no-op — so callers that need to know their job record
    is still there must check this return value explicitly)."""
    if not kwargs:
        return False

    allowed = {"status", "chunk_count", "error"}
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return False

    fields["updated_at"] = time.time()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [job_id]

    with _connect() as conn:
        cursor = conn.execute(f"UPDATE jobs SET {set_clause} WHERE job_id = ?", values)
        return cursor.rowcount > 0


def get_job(job_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None


init_db()
