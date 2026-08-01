# app/services/chat_history.py
"""
SQLite-backed chat history, keyed by session_id.

Shares the same lightweight SQLite database as job tracking — no new
infrastructure to run, and it persists across restarts like everything
else in this single-instance deployment. Swap for Redis/Postgres under
the same conditions you'd swap the job store.

Messages older than `settings.CHAT_HISTORY_TTL_DAYS` are purged so
sessions don't accumulate forever. There's no scheduler/cron in this
deployment, so cleanup runs opportunistically from `append_message`
instead — throttled via a stored timestamp so it's a cheap no-op on
most calls rather than a DELETE scan on every chat turn.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager

from app.core.config import settings
from app.services.job_store import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages (session_id);

CREATE TABLE IF NOT EXISTS chat_history_cleanup_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_cleanup_at REAL NOT NULL
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
        conn.executescript(_SCHEMA)


def append_message(session_id: str, role: str, content: str) -> None:
    _maybe_cleanup_expired()

    with _connect() as conn:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time()),
        )


def get_history(session_id: str, max_turns: int) -> list[dict]:
    """Returns up to the last `max_turns` (user, assistant) exchanges, oldest first,
    in the {"role": ..., "content": ...} shape the OpenAI-compatible chat API expects."""
    limit = max_turns * 2  # each turn is one user + one assistant message

    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_messages WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()

    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def cleanup_expired_messages() -> int:
    """Deletes messages older than CHAT_HISTORY_TTL_DAYS. Returns the number
    of rows deleted. Safe to call anytime — a no-op if TTL is disabled
    (CHAT_HISTORY_TTL_DAYS <= 0)."""
    if settings.CHAT_HISTORY_TTL_DAYS <= 0:
        return 0

    cutoff = time.time() - (settings.CHAT_HISTORY_TTL_DAYS * 86400)

    with _connect() as conn:
        cursor = conn.execute("DELETE FROM chat_messages WHERE created_at < ?", (cutoff,))
        return cursor.rowcount


def _maybe_cleanup_expired() -> None:
    """Runs `cleanup_expired_messages` at most once per
    CHAT_HISTORY_CLEANUP_INTERVAL_SECONDS, tracked via a single-row table
    so this works correctly even across process restarts."""
    if settings.CHAT_HISTORY_TTL_DAYS <= 0:
        return

    now = time.time()

    with _connect() as conn:
        row = conn.execute(
            "SELECT last_cleanup_at FROM chat_history_cleanup_state WHERE id = 1"
        ).fetchone()
        last_cleanup_at = row["last_cleanup_at"] if row else 0.0

        if now - last_cleanup_at < settings.CHAT_HISTORY_CLEANUP_INTERVAL_SECONDS:
            return

        conn.execute(
            "INSERT INTO chat_history_cleanup_state (id, last_cleanup_at) VALUES (1, ?) "
            "ON CONFLICT(id) DO UPDATE SET last_cleanup_at = excluded.last_cleanup_at",
            (now,),
        )

    cleanup_expired_messages()


init_db()
