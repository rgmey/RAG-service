# app/rag/keyword_index.py
"""
Keyword (BM25) index, backed by SQLite's FTS5 extension.

Vector search retrieves by semantic similarity, which misses exact-term
matches that don't cluster nearby in embedding space — product codes,
names, acronyms, numbers. Keyword search catches those. Combining both
(see `retrieval.hybrid_search`) covers more ground than either alone.

Uses the same SQLite file as job tracking/chat history — one piece of
infrastructure, not three.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from contextlib import contextmanager

from app.services.job_store import DB_PATH

logger = logging.getLogger(__name__)

_SCHEMA = "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(id UNINDEXED, text);"

# Set to False if FTS5 isn't compiled into the local sqlite3 build (rare,
# but happens on some minimal/older system Python builds). Keyword search
# and hybrid search both degrade gracefully to vector-only in that case.
FTS5_AVAILABLE = True

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")  # keeps codes like "E-4021" intact


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
    global FTS5_AVAILABLE
    try:
        with _connect() as conn:
            conn.execute(_SCHEMA)
    except sqlite3.OperationalError:
        logger.warning(
            "SQLite FTS5 extension unavailable — keyword/hybrid search will "
            "fall back to vector-only search.",
            exc_info=True,
        )
        FTS5_AVAILABLE = False


def index_chunks(vectors: list[dict]) -> None:
    """vectors: list of {"id": str, "text": str, ...} — same shape ingestion
    already builds for the vector store, so this is a direct pass-through."""
    if not FTS5_AVAILABLE or not vectors:
        return

    with _connect() as conn:
        conn.executemany(
            "INSERT INTO chunks_fts (id, text) VALUES (?, ?)",
            [(v["id"], v["text"]) for v in vectors],
        )


def _build_match_query(query: str) -> str | None:
    """
    Builds an FTS5 MATCH expression that OR's individual terms together,
    rather than requiring the literal input string as one exact phrase —
    a phrase match essentially never fires for natural-language questions
    against prose chunks. BM25 ranking still rewards chunks matching more
    (or rarer) terms higher, which is the point of using BM25 at all.
    Each term is double-quoted so FTS5 keywords in the text (AND/OR/NOT)
    can't be mistaken for query syntax.
    """
    tokens = _TOKEN_RE.findall(query)
    if not tokens:
        return None
    return " OR ".join(f'"{token}"' for token in tokens)


def keyword_search(query: str, k: int = 10) -> list[tuple[str, str]]:
    """Returns up to k (id, text) pairs ranked by BM25, best match first."""
    if not FTS5_AVAILABLE or not query.strip():
        return []

    match_query = _build_match_query(query)
    if not match_query:
        return []

    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT id, text FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY bm25(chunks_fts) LIMIT ?",
                (match_query, k),
            ).fetchall()
        return [(row["id"], row["text"]) for row in rows]
    except sqlite3.OperationalError:
        # e.g. a query that FTS5's tokenizer can't parse at all — treat as
        # no keyword matches rather than failing the whole search.
        logger.warning("Keyword search query failed, returning no matches", exc_info=True)
        return []


init_db()
