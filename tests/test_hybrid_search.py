from app.rag.retrieval import _reciprocal_rank_fusion, hybrid_search


def test_rrf_ranks_items_in_both_lists_above_items_in_one():
    vector_list = ["a", "b", "c"]
    keyword_list = ["b", "d", "a"]

    fused = _reciprocal_rank_fusion([vector_list, keyword_list])
    fused_ids = [doc_id for doc_id, _score in fused]

    # "a" and "b" appear in both lists, so they should outrank "c"/"d",
    # which each appear in only one.
    assert set(fused_ids[:2]) == {"a", "b"}


def test_rrf_handles_empty_lists():
    assert _reciprocal_rank_fusion([[], []]) == []


def test_rrf_handles_single_list():
    fused = _reciprocal_rank_fusion([["x", "y", "z"]])
    assert [doc_id for doc_id, _score in fused] == ["x", "y", "z"]


def test_hybrid_search_falls_back_to_vector_only_when_no_keyword_hits(mocker):
    mocker.patch(
        "app.rag.retrieval._vector_candidates",
        return_value=[("id1", "vector result one"), ("id2", "vector result two")],
    )
    mocker.patch("app.rag.retrieval.keyword_search", return_value=[])

    result = hybrid_search("some question", [0.1, 0.2, 0.3], k=2)

    assert result == ["vector result one", "vector result two"]


def test_hybrid_search_fuses_vector_and_keyword_results(mocker):
    mocker.patch(
        "app.rag.retrieval._vector_candidates",
        return_value=[("id1", "vector-favored chunk"), ("id2", "shared chunk")],
    )
    mocker.patch(
        "app.rag.retrieval.keyword_search",
        return_value=[("id2", "shared chunk"), ("id3", "keyword-favored chunk")],
    )

    result = hybrid_search("some question", [0.1, 0.2, 0.3], k=3)

    # "shared chunk" (id2) appears in both lists and should rank first.
    assert result[0] == "shared chunk"
    assert set(result) == {"vector-favored chunk", "shared chunk", "keyword-favored chunk"}
