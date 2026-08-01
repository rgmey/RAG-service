# app/rag/pipeline.py
import logging

from app.core.config import settings
from app.rag.chunking import chunk_text
from app.rag.embeddings import get_embeddings
from app.rag.index import store_chunks
from app.rag.ingestion import extract_text
from app.rag.keyword_index import index_chunks
from app.services.job_store import update_job

logger = logging.getLogger(__name__)


def process_document(file_path: str, job_id: str) -> None:
    try:
        update_job(job_id, status="processing")

        text = extract_text(file_path)
        chunks = chunk_text(
            text,
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP,
        )

        if not chunks:
            update_job(job_id, status="failed", error="No extractable text in document")
            return

        embeddings = get_embeddings(chunks)  # one batched API call instead of N calls

        vectors = [
            {"id": f"{job_id}_{i}", "text": chunk, "embedding": embedding}
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]

        store_chunks(vectors)
        index_chunks(vectors)  # keyword (BM25) index for hybrid search

        update_job(job_id, status="done", chunk_count=len(vectors))

    except Exception as e:
        logger.exception("Failed to process document %s (job %s)", file_path, job_id)
        update_job(job_id, status="failed", error=str(e))
