# app/api/routes.py
import uuid

from fastapi import APIRouter, BackgroundTasks, File, UploadFile

from app.core.config import settings
from app.rag.embeddings import get_embedding
from app.rag.pipeline import process_document
from app.rag.reranking import rerank
from app.rag.retrieval import hybrid_search, search
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_history import append_message, get_history
from app.services.job_store import create_job, get_job
from app.services.llm_service import ask_llm, condense_question
from app.services.storage import save_file

router = APIRouter()


# -------------------
# Upload endpoint
# -------------------
@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    # save_file raises HTTPException on bad type/size, which FastAPI
    # turns into a proper 400 response automatically.
    file_path = await save_file(file)

    job_id = str(uuid.uuid4())
    create_job(job_id, file_path)

    background_tasks.add_task(process_document, file_path, job_id)

    return {"job_id": job_id, "status": "queued"}


# -------------------
# Status endpoint
# -------------------
@router.get("/status/{job_id}")
def get_status(job_id: str):
    job = get_job(job_id)

    if not job:
        return {"error": "job not found"}

    return job


# -------------------
# Chat / query endpoint
# -------------------
@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # No session_id means "start a new conversation" — generate one and
    # hand it back so the client can continue the thread on later calls.
    session_id = request.session_id or str(uuid.uuid4())
    history = get_history(session_id, max_turns=settings.CHAT_HISTORY_MAX_TURNS)

    # Rewrite follow-up questions ("what about last year?") into a
    # standalone query using history, before embedding for retrieval —
    # otherwise vector search only sees the bare pronoun-laden question.
    search_question = (
        condense_question(request.question, history)
        if settings.CONDENSE_QUESTION_ENABLED and history
        else request.question
    )

    query_embedding = get_embedding(search_question)

    fetch_k = (
        min(request.k * settings.RERANK_FETCH_MULTIPLIER, settings.RERANK_MAX_FETCH)
        if settings.RERANK_ENABLED
        else request.k
    )

    if settings.HYBRID_SEARCH_ENABLED:
        # Fuses vector + keyword (BM25) search results via Reciprocal
        # Rank Fusion, so exact-term matches aren't missed just because
        # they don't cluster nearby in embedding space.
        candidates = hybrid_search(search_question, query_embedding, k=fetch_k)
    else:
        candidates = [doc for doc, _distance in search(query_embedding, k=fetch_k)]

    if settings.RERANK_ENABLED:
        # Over-fetched above; let the LLM re-score for actual relevance
        # to the question before keeping the top k.
        context_chunks = rerank(search_question, candidates, top_k=request.k)
    else:
        context_chunks = candidates[: request.k]

    # The LLM answers the question as the user actually phrased it — only
    # retrieval used the condensed version.
    answer = ask_llm(request.question, context=context_chunks, history=history)

    append_message(session_id, "user", request.question)
    append_message(session_id, "assistant", answer)

    return ChatResponse(answer=answer, sources=context_chunks, session_id=session_id)
