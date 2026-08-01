import uuid

from app.rag.keyword_index import index_chunks, keyword_search


def test_keyword_search_finds_exact_term_match():
    unique_term = f"zynthex{uuid.uuid4().hex[:8]}"
    chunk_id = f"test_{uuid.uuid4()}"

    index_chunks([{"id": chunk_id, "text": f"The {unique_term} protocol was ratified in 1998."}])

    results = keyword_search(unique_term, k=5)

    assert any(doc_id == chunk_id for doc_id, _text in results)


def test_keyword_search_returns_empty_for_no_match():
    nonsense = f"qqqzzz{uuid.uuid4().hex}"
    assert keyword_search(nonsense, k=5) == []


def test_keyword_search_handles_empty_query():
    assert keyword_search("", k=5) == []


def test_index_chunks_handles_empty_list():
    # Should not raise
    index_chunks([])
