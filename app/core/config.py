# app/core/config.py
import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-4o-mini")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Single source of truth for the vector store so ingestion and
    # retrieval always talk to the same place.
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "documents")

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
    ALLOWED_UPLOAD_EXTENSIONS = {".pdf"}

    CORS_ORIGINS = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]

    # Where job tracking + chat history are persisted (SQLite).
    JOBS_DB_PATH = os.getenv("JOBS_DB_PATH", "data/jobs.db")

    # --- Re-ranking ---
    # Vector search retrieves by embedding similarity, which is a decent
    # but imperfect proxy for relevance. When enabled, we over-fetch
    # candidates and have the LLM re-score them for actual relevance to
    # the question before keeping the top `k`.
    RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
    RERANK_MODEL = os.getenv("RERANK_MODEL", DEFAULT_MODEL)
    RERANK_FETCH_MULTIPLIER = int(os.getenv("RERANK_FETCH_MULTIPLIER", "4"))
    RERANK_MAX_FETCH = int(os.getenv("RERANK_MAX_FETCH", "20"))

    # --- Hybrid search ---
    # Combines vector search with SQLite FTS5 keyword (BM25) search via
    # Reciprocal Rank Fusion, so exact-term matches (names, codes, numbers)
    # aren't missed just because they don't cluster nearby in embedding
    # space. Falls back to vector-only automatically if FTS5 isn't
    # available or returns nothing.
    HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
    RRF_K = int(os.getenv("RRF_K", "60"))

    # --- Chat history ---
    # Number of prior (user, assistant) turns kept per session and fed
    # back to the model as conversation context.
    CHAT_HISTORY_MAX_TURNS = int(os.getenv("CHAT_HISTORY_MAX_TURNS", "6"))

    # Rewrites follow-up questions ("what about last year?") into
    # standalone questions using chat history before embedding, so
    # retrieval isn't thrown off by missing context.
    CONDENSE_QUESTION_ENABLED = os.getenv("CONDENSE_QUESTION_ENABLED", "true").lower() == "true"


settings = Settings()

if not settings.OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY missing")
