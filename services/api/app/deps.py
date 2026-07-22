"""Wiring. Everything expensive is built once at startup and reused.

Loading an ONNX session or opening an OpenSearch connection pool per request
would dominate latency, so they live on app.state and are handed out from here.
"""

from __future__ import annotations

import logging

from fastapi import Request

from rag_core.config import Settings, get_settings
from rag_core.embeddings import get_embedder
from rag_core.extraction import IngestPipeline, OCRClient, TikaClient
from rag_core.llm import get_llm
from rag_core.rag import RagPipeline
from rag_core.search import ChunkIndexer, HybridRetriever, IndexAdmin, get_client

log = logging.getLogger(__name__)


class Container:
    """Explicit composition root. No framework magic, no global service locator
    beyond this one object."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = get_client()
        self.embedder = get_embedder()
        self.llm = get_llm()

        self.admin = IndexAdmin(self.client, settings.opensearch)
        self.indexer = ChunkIndexer(self.client, settings.opensearch)
        self.retriever = HybridRetriever(
            self.client, self.embedder, settings.retrieval, settings.opensearch
        )
        self.rag = RagPipeline(self.retriever, self.llm, settings)

        self.tika = TikaClient(settings.tika)
        self.ocr = OCRClient(settings.ocr) if settings.ocr.enabled else None
        self.ingest = IngestPipeline(
            settings, self.tika, self.ocr, self.embedder, self.indexer, self.admin
        )

    async def aclose(self) -> None:
        self.tika.close()
        if self.ocr:
            self.ocr.close()
        await self.llm.aclose()


def build_container() -> Container:
    return Container(get_settings())


def get_container(request: Request) -> Container:
    return request.app.state.container
