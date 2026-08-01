# app/rag/retrieval.py
from app.core.config import settings
from app.rag.index import collection
from app.rag.keyword_index import keyword_search


def search(query_embedding, k: int = 3):
    """Returns top-k (document_text, distance) pairs from the shared collection."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    documents = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results.get("distances") else [None] * len(documents)

    return list(zip(documents, distances, strict=True))


def _vector_candidates(query_embedding, k: int) -> list[tuple[str, str]]:
    """Returns up to k (id, text) pairs from vector search, best match first."""
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    ids = results["ids"][0] if results.get("ids") else []
    documents = results["documents"][0] if results.get("documents") else []
    return list(zip(ids, documents, strict=True))


def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]], rrf_k: int = 60
) -> list[tuple[str, float]]:
    """
    Combines multiple ranked ID lists into one fused ranking.

    Standard Reciprocal Rank Fusion: each id's score is the sum of
    1 / (rrf_k + rank) across every list it appears in (rank is 1-based).
    Using rank position rather than each method's raw score is what makes
    this work at all — vector distances and BM25 scores aren't on
    comparable scales, but "how many places from the top" is.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def hybrid_search(query_text: str, query_embedding, k: int = 10) -> list[str]:
    """
    Fuses vector search and keyword (BM25) search results via Reciprocal
    Rank Fusion, returning up to k chunk texts, best first.

    Falls back to vector-only results if the keyword index has nothing
    (e.g. FTS5 unavailable, or no keyword matches at all) — hybrid search
    should only ever add coverage, never subtract it.
    """
    vector_hits = _vector_candidates(query_embedding, k)
    keyword_hits = keyword_search(query_text, k)

    if not keyword_hits:
        return [text for _id, text in vector_hits[:k]]

    id_to_text = {doc_id: text for doc_id, text in [*vector_hits, *keyword_hits]}
    vector_ids = [doc_id for doc_id, _text in vector_hits]
    keyword_ids = [doc_id for doc_id, _text in keyword_hits]

    fused = _reciprocal_rank_fusion([vector_ids, keyword_ids], rrf_k=settings.RRF_K)

    return [id_to_text[doc_id] for doc_id, _score in fused[:k] if doc_id in id_to_text]
