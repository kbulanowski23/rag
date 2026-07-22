"""RRF fusion, filter translation, and citation extraction.

No OpenSearch and no LLM: these are pure functions, and they are where relevance
bugs actually live.
"""

from __future__ import annotations

from rag_core.config import OpenSearchSettings, RetrievalSettings
from rag_core.documents import SearchHit
from rag_core.rag.pipeline import extract_cited
from rag_core.rag.prompt import select_within_budget
from rag_core.search.retriever import HybridRetriever, build_filter


def hit(chunk_id: str, text: str = "x" * 50) -> SearchHit:
    return SearchHit(chunk_id=chunk_id, doc_id="d", text=text, score=0.0)


def retriever() -> HybridRetriever:
    return HybridRetriever(None, None, RetrievalSettings(rrf_k=60), OpenSearchSettings())


def test_rrf_ranks_documents_found_by_both_retrievers_first():
    # The core justification for hybrid search: agreement between a lexical and
    # a semantic retriever is a stronger signal than either one's top result.
    bm25 = [hit("a"), hit("b"), hit("c")]
    knn = [hit("c"), hit("d"), hit("a")]
    fused = retriever()._rrf({"bm25": bm25, "knn": knn})
    ids = [h.chunk_id for h in fused]
    assert set(ids) == {"a", "b", "c", "d"}
    assert ids[0] in ("a", "c"), "a doc found by both should outrank one found by one"
    assert ids.index("a") < ids.index("b")
    assert ids.index("c") < ids.index("d")


def test_rrf_records_provenance():
    fused = retriever()._rrf({"bm25": [hit("a")], "knn": [hit("b"), hit("a")]})
    by_id = {h.chunk_id: h for h in fused}
    assert by_id["a"].retrievers == {"bm25": 1, "knn": 2}
    assert by_id["b"].retrievers == {"knn": 1}


def test_rrf_handles_one_empty_leg():
    fused = retriever()._rrf({"bm25": [], "knn": [hit("a"), hit("b")]})
    assert [h.chunk_id for h in fused] == ["a", "b"]


def test_rrf_scores_descend():
    fused = retriever()._rrf({"bm25": [hit("a"), hit("b"), hit("c")]})
    scores = [h.score for h in fused]
    assert scores == sorted(scores, reverse=True)


def test_build_filter_shapes():
    clauses = build_filter({
        "doc_id": "abc",
        "extraction_source": ["tika", "ocr"],
        "indexed_at": {"gte": "2024-01-01"},
        "ignored": None,
        "also_ignored": [],
    })
    assert {"term": {"doc_id": "abc"}} in clauses
    assert {"terms": {"extraction_source": ["tika", "ocr"]}} in clauses
    assert {"range": {"indexed_at": {"gte": "2024-01-01"}}} in clauses
    assert len(clauses) == 3


def test_build_filter_empty():
    assert build_filter(None) == []
    assert build_filter({}) == []


def test_context_budget_is_respected():
    hits = [hit(str(i), "y" * 1000) for i in range(20)]
    selected = select_within_budget(hits, budget_tokens=1000)  # ~4000 chars
    assert 0 < len(selected) < 20


def test_budget_always_keeps_at_least_one_hit():
    # A single oversized chunk must still be sent; returning nothing would make
    # the model answer with no context at all.
    selected = select_within_budget([hit("big", "z" * 100000)], budget_tokens=100)
    assert len(selected) == 1


def test_extract_cited_finds_referenced_sources():
    used = [hit("a"), hit("b"), hit("c")]
    assert extract_cited("Per policy [1], the period is seven years [3].", used) == [1, 3]


def test_extract_cited_ignores_out_of_range_and_duplicates():
    used = [hit("a"), hit("b")]
    assert extract_cited("See [1], [2], [1] and the bogus [9].", used) == [1, 2]


def test_extract_cited_with_no_citations():
    assert extract_cited("I could not find this in the documents.", [hit("a")]) == []
