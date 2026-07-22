from __future__ import annotations

import logging
from typing import Any, Iterable

from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk

from rag_core.config import OpenSearchSettings, get_settings
from rag_core.documents import Chunk
from rag_core.search.index_mapping import build_mapping

log = logging.getLogger(__name__)


def build_client(s: OpenSearchSettings) -> OpenSearch:
    kwargs: dict[str, Any] = {
        "hosts": s.host_list,
        "timeout": s.timeout_s,
        "max_retries": 3,
        "retry_on_timeout": True,
        "connection_class": RequestsHttpConnection,
        "verify_certs": s.verify_certs,
    }
    if s.username:
        kwargs["http_auth"] = (s.username, s.password)
    if s.ca_certs:
        kwargs["ca_certs"] = s.ca_certs
    if not s.verify_certs:
        # Local development against the demo security config only. Never set
        # RAG_OPENSEARCH__VERIFY_CERTS=false in a deployed environment.
        kwargs["ssl_show_warn"] = False
    return OpenSearch(**kwargs)


_client: OpenSearch | None = None


def get_client() -> OpenSearch:
    global _client
    if _client is None:
        _client = build_client(get_settings().opensearch)
    return _client


class IndexAdmin:
    def __init__(self, client: OpenSearch, settings: OpenSearchSettings) -> None:
        self.client = client
        self.s = settings

    def exists(self, index: str | None = None) -> bool:
        return bool(self.client.indices.exists(index=index or self.s.index))

    def create(self, dim: int, index: str | None = None, recreate: bool = False) -> str:
        name = index or self.s.index
        if self.exists(name):
            if not recreate:
                log.info("index %s already exists", name)
                return name
            log.warning("deleting index %s", name)
            self.client.indices.delete(index=name)
        self.client.indices.create(index=name, body=build_mapping(self.s, dim))
        log.info("created index %s (dim=%d, engine=%s)", name, dim, self.s.knn_engine)
        return name

    def delete_by_doc_id(self, doc_id: str, index: str | None = None) -> int:
        """Used before re-ingesting a document, so a shorter new version does
        not leave orphaned chunks from the previous one behind."""
        res = self.client.delete_by_query(
            index=index or self.s.index,
            body={"query": {"term": {"doc_id": doc_id}}},
            refresh=True,
            conflicts="proceed",
        )
        return int(res.get("deleted", 0))

    def stats(self, index: str | None = None) -> dict[str, Any]:
        name = index or self.s.index
        if not self.exists(name):
            return {"index": name, "exists": False}
        count = self.client.count(index=name)["count"]
        docs = self.client.search(
            index=name,
            body={"size": 0, "aggs": {"docs": {"cardinality": {"field": "doc_id"}}}},
        )
        return {
            "index": name,
            "exists": True,
            "chunks": count,
            "documents": docs["aggregations"]["docs"]["value"],
        }


class ChunkIndexer:
    def __init__(self, client: OpenSearch, settings: OpenSearchSettings) -> None:
        self.client = client
        self.s = settings

    def _actions(self, chunks: Iterable[Chunk], index: str) -> Iterable[dict[str, Any]]:
        for c in chunks:
            yield {
                "_op_type": "index",   # deterministic id => re-ingest overwrites
                "_index": index,
                "_id": c.chunk_id,
                "_source": c.to_os_doc(),
            }

    def index_chunks(
        self, chunks: list[Chunk], index: str | None = None, refresh: bool = False
    ) -> tuple[int, list[dict]]:
        if not chunks:
            return 0, []
        name = index or self.s.index
        ok, errors = bulk(
            self.client,
            self._actions(chunks, name),
            chunk_size=self.s.bulk_size,
            raise_on_error=False,
            refresh="true" if refresh else "false",
        )
        if errors:
            log.error("bulk index: %d/%d failed; first: %s", len(errors), len(chunks), errors[0])
        return ok, list(errors)
