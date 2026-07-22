"""Hybrid retrieval: BM25 + kNN, fused with Reciprocal Rank Fusion.

Why fusion happens here and not in OpenSearch: the native hybrid search pipeline
(the `normalization-processor`) requires a search pipeline registered on the
cluster and behaves differently across OpenSearch versions. Doing RRF in ~30
lines of Python removes a dependency on a cluster we do not control, costs one
extra round trip, and is trivially inspectable when relevance looks wrong.

RRF also has a practical advantage over score normalisation: BM25 scores and
cosine similarities are not comparable and their ranges shift per query, so any
weighted sum of them needs per-query calibration. Rank position needs none.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from opensearchpy import OpenSearch

from rag_core.config import OpenSearchSettings, RetrievalSettings
from rag_core.documents import SearchHit
from rag_core.embeddings.base import EmbeddingProvider

log = logging.getLogger(__name__)

_SOURCE_FIELDS = [
    "chunk_id", "doc_id", "text", "ordinal", "page_start", "page_end",
    "filename", "title", "source_uri", "extraction_source", "metadata",
]


def build_filter(filters: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Translate a flat filter dict into OpenSearch filter clauses.

    Scalars become `term`, lists become `terms`, and a {"gte":..,"lte":..} dict
    becomes a `range`. Filters apply to both retrievers so that BM25 and kNN see
    the same candidate space -- otherwise fusion silently favours whichever leg
    was less restricted.
    """
    if not filters:
        return []
    clauses: list[dict[str, Any]] = []
    for field, value in filters.items():
        if value is None or value == []:
            continue
        if isinstance(value, (list, tuple, set)):
            clauses.append({"terms": {field: list(value)}})
        elif isinstance(value, dict):
            clauses.append({"range": {field: value}})
        else:
            clauses.append({"term": {field: value}})
    return clauses


class HybridRetriever:
    def __init__(
        self,
        client: OpenSearch,
        embedder: EmbeddingProvider,
        retrieval: RetrievalSettings,
        opensearch: OpenSearchSettings,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.r = retrieval
        self.os = opensearch

    # -- individual retrievers ----------------------------------------------

    def bm25(self, query: str, k: int, filters: dict | None = None) -> list[SearchHit]:
        body = {
            "size": k,
            "_source": _SOURCE_FIELDS,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                # title is weighted up: a chunk from a document
                                # whose title matches is usually more on-topic.
                                "fields": ["text^1.0", "text.raw^0.5", "title^2.0", "filename.text^1.5"],
                                "type": "best_fields",
                            }
                        }
                    ],
                    "filter": build_filter(filters),
                }
            },
        }
        res = self.client.search(index=self.os.index, body=body)
        return [SearchHit.from_os(h) for h in res["hits"]["hits"]]

    def knn(self, query: str, k: int, filters: dict | None = None) -> list[SearchHit]:
        vector = self.embedder.embed_query(query)
        knn_clause: dict[str, Any] = {"vector": vector, "k": k}
        if clauses := build_filter(filters):
            # Pre-filtering inside the knn clause (Lucene engine) keeps recall
            # correct; a post-filter would prune the k results after the fact
            # and can return far fewer than k.
            knn_clause["filter"] = {"bool": {"filter": clauses}}
        body = {
            "size": k,
            "_source": _SOURCE_FIELDS,
            "query": {"knn": {"embedding": knn_clause}},
        }
        res = self.client.search(index=self.os.index, body=body)
        return [SearchHit.from_os(h) for h in res["hits"]["hits"]]

    # -- fusion --------------------------------------------------------------

    def _rrf(self, ranked_lists: dict[str, list[SearchHit]]) -> list[SearchHit]:
        k = self.r.rrf_k
        scores: dict[str, float] = {}
        merged: dict[str, SearchHit] = {}
        for name, hits in ranked_lists.items():
            for rank, hit in enumerate(hits, start=1):
                scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
                if hit.chunk_id not in merged:
                    merged[hit.chunk_id] = hit
                merged[hit.chunk_id].retrievers[name] = rank
        out = []
        for chunk_id, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            hit = merged[chunk_id]
            hit.score = score
            out.append(hit)
        return out

    def retrieve(
        self, query: str, k: int | None = None, filters: dict | None = None
    ) -> list[SearchHit]:
        final_k = k or self.r.final_k
        mode = self.r.fusion

        if mode == "bm25_only":
            hits = self.bm25(query, self.r.top_k_bm25, filters)
        elif mode == "vector_only":
            hits = self.knn(query, self.r.top_k_vector, filters)
        else:
            # Both legs hit the same cluster; run them concurrently. opensearch-py
            # with RequestsHttpConnection is blocking, hence threads not asyncio.
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_bm25 = pool.submit(self.bm25, query, self.r.top_k_bm25, filters)
                f_knn = pool.submit(self.knn, query, self.r.top_k_vector, filters)
                bm25_hits, knn_hits = f_bm25.result(), f_knn.result()
            hits = self._rrf({"bm25": bm25_hits, "knn": knn_hits})

        if self.r.min_score > 0:
            hits = [h for h in hits if h.score >= self.r.min_score]

        if self.r.rerank_enabled:
            from rag_core.search.rerank import get_reranker

            reranker = get_reranker(self.r)
            if reranker is not None:
                hits = reranker.rerank(query, hits[: self.r.rerank_candidates], final_k)

        return hits[:final_k]
