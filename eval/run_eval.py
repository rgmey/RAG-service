# eval/run_eval.py
"""
Retrieval quality eval: compares vector-only, keyword-only, hybrid (RRF),
and hybrid+rerank on a small synthetic question set, reporting Hit@k and
MRR for each.

This makes real embedding (and, for the last strategy, real LLM
re-ranking) calls against OPENROUTER_API_KEY — it's a dev tool you run
manually, not part of the (fully-mocked, offline) pytest suite.

Usage:
    export OPENROUTER_API_KEY=sk-or-v1-...
    python -m eval.run_eval [--k 3]
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

# Isolate this run's data from your real deployment — set before importing
# anything from `app`, since those modules read these env vars at import time.
_TMP_DIR = tempfile.mkdtemp(prefix="rag_eval_")
os.environ.setdefault("CHROMA_PERSIST_DIR", str(Path(_TMP_DIR) / "chroma"))
os.environ.setdefault("JOBS_DB_PATH", str(Path(_TMP_DIR) / "eval_jobs.db"))

if not os.getenv("OPENROUTER_API_KEY"):
    print(
        "OPENROUTER_API_KEY is not set. Export it before running the eval:\n"
        "  export OPENROUTER_API_KEY=sk-or-v1-...",
        file=sys.stderr,
    )
    sys.exit(1)

from app.core.config import settings  # noqa: E402
from app.rag.embeddings import get_embeddings  # noqa: E402
from app.rag.index import store_chunks  # noqa: E402
from app.rag.keyword_index import index_chunks, keyword_search  # noqa: E402
from app.rag.reranking import rerank  # noqa: E402
from app.rag.retrieval import _reciprocal_rank_fusion, _vector_candidates  # noqa: E402
from eval.dataset import CORPUS, QUESTIONS  # noqa: E402


def build_index() -> None:
    texts = [doc["text"] for doc in CORPUS]
    embeddings = get_embeddings(texts)
    vectors = [
        {"id": doc["id"], "text": doc["text"], "embedding": emb}
        for doc, emb in zip(CORPUS, embeddings, strict=True)
    ]
    store_chunks(vectors)
    index_chunks(vectors)


def vector_only(question: str, embedding: list[float], k: int) -> list[str]:
    return [doc_id for doc_id, _text in _vector_candidates(embedding, k)]


def keyword_only(question: str, embedding: list[float], k: int) -> list[str]:
    return [doc_id for doc_id, _text in keyword_search(question, k)]


def hybrid(question: str, embedding: list[float], k: int) -> list[str]:
    vector_hits = _vector_candidates(embedding, k)
    keyword_hits = keyword_search(question, k)
    vector_ids = [doc_id for doc_id, _text in vector_hits]
    keyword_ids = [doc_id for doc_id, _text in keyword_hits]
    fused = _reciprocal_rank_fusion([vector_ids, keyword_ids], rrf_k=settings.RRF_K)
    return [doc_id for doc_id, _score in fused[:k]]


def hybrid_rerank(question: str, embedding: list[float], k: int) -> list[str]:
    vector_hits = _vector_candidates(embedding, k)
    keyword_hits = keyword_search(question, k)
    id_to_text = dict([*vector_hits, *keyword_hits])

    vector_ids = [doc_id for doc_id, _text in vector_hits]
    keyword_ids = [doc_id for doc_id, _text in keyword_hits]
    fused_ids = [
        doc_id
        for doc_id, _score in _reciprocal_rank_fusion(
            [vector_ids, keyword_ids], rrf_k=settings.RRF_K
        )
    ]
    fused_texts = [id_to_text[i] for i in fused_ids if i in id_to_text]

    reranked_texts = rerank(question, fused_texts, top_k=k)
    text_to_id = {text: doc_id for doc_id, text in id_to_text.items()}
    return [text_to_id[t] for t in reranked_texts if t in text_to_id]


STRATEGIES = {
    "vector-only": vector_only,
    "keyword-only": keyword_only,
    "hybrid (RRF)": hybrid,
    "hybrid + rerank": hybrid_rerank,
}


def evaluate(strategy_fn, question_embeddings: list[list[float]], k: int) -> dict:
    hits = 0
    reciprocal_ranks = []

    for item, embedding in zip(QUESTIONS, question_embeddings, strict=True):
        retrieved_ids = strategy_fn(item["question"], embedding, k)

        if item["gold_id"] in retrieved_ids:
            hits += 1
            rank = retrieved_ids.index(item["gold_id"]) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

    n = len(QUESTIONS)
    return {"hit_rate": hits / n, "mrr": sum(reciprocal_ranks) / n}


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval quality eval for the RAG pipeline")
    parser.add_argument("--k", type=int, default=3, help="Chunks retrieved per question")
    args = parser.parse_args()

    print(f"Indexing {len(CORPUS)} synthetic chunks...")
    build_index()

    print("Embedding evaluation questions...")
    question_embeddings = get_embeddings([q["question"] for q in QUESTIONS])

    print(
        f"Running {len(QUESTIONS)} questions against {len(STRATEGIES)} strategies "
        f"(k={args.k})...\n"
    )

    results = {name: evaluate(fn, question_embeddings, args.k) for name, fn in STRATEGIES.items()}

    name_width = max(len(name) for name in results) + 2
    print(f"{'Strategy'.ljust(name_width)}{'Hit@k'.rjust(8)}{'MRR'.rjust(8)}")
    print("-" * (name_width + 16))
    for name, metrics in results.items():
        print(f"{name.ljust(name_width)}{metrics['hit_rate'] * 100:7.1f}%{metrics['mrr']:8.3f}")

    print(f"\n(scratch data for this run lives in {_TMP_DIR} — safe to delete)")


if __name__ == "__main__":
    main()
