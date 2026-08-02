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

_MISSING_JOB_HINT = (
    "The job record vanished mid-processing — this happens if the underlying "
    "storage was reset while this task was running (e.g. a platform "
    "redeploy/restart on a free-tier host, or DEV_RESET_DATA_ON_START firing "
    "mid-upload locally). The client polling /status/%s will see 'job not "
    "found' with no other explanation, so logging it clearly here."
)


def process_document(file_path: str, job_id: str) -> None:
    try:
        if not update_job(job_id, status="processing"):
            logger.warning(_MISSING_JOB_HINT, job_id)
            return

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

        if not update_job(job_id, status="done", chunk_count=len(vectors)):
            # Processing succeeded (embeddings were paid for, chunks were
            # stored) but there's no job record left to mark done — worth
            # knowing about explicitly rather than as a silent no-op.
            logger.warning(_MISSING_JOB_HINT, job_id)

    except Exception as e:
        logger.exception("Failed to process document %s (job %s)", file_path, job_id)
        update_job(job_id, status="failed", error=str(e))
