"""Embeddings from an OpenAI-compatible /v1/embeddings endpoint.

Use this only if the work environment mandates a central embedding service. The
local ONNX provider is preferred: it removes a network hop from every ingest
batch and every query, and it cannot drift out from under the index.

Warning worth stating once: changing the embedding model invalidates every
vector already indexed. Re-index from scratch after any such switch.
"""

from __future__ import annotations

import httpx

from rag_core.config import EmbeddingSettings
from rag_core.embeddings.base import EmbeddingError, EmbeddingProvider


class RemoteOpenAIEmbedder(EmbeddingProvider):
    def __init__(self, settings: EmbeddingSettings) -> None:
        if not settings.base_url or not settings.model:
            raise EmbeddingError(
                "RAG_EMBEDDING__BASE_URL and RAG_EMBEDDING__MODEL are required "
                "for the openai_compatible embedding provider"
            )
        self.settings = settings
        self.dim = settings.dim
        headers = {"Content-Type": "application/json"}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        self.client = httpx.Client(
            base_url=settings.base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(settings.timeout_s, connect=10.0),
        )

    def _post(self, inputs: list[str]) -> list[list[float]]:
        try:
            r = self.client.post("/embeddings", json={"model": self.settings.model, "input": inputs})
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as e:
            raise EmbeddingError(f"HTTP {e.response.status_code}: {e.response.text[:400]}") from e
        except httpx.HTTPError as e:
            raise EmbeddingError(str(e)) from e

        try:
            # The API does not promise ordering; sort by index before returning.
            items = sorted(data["data"], key=lambda d: d["index"])
        except (KeyError, TypeError) as e:
            raise EmbeddingError(f"unexpected response: {str(data)[:400]}") from e
        return [item["embedding"] for item in items]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prefix = self.settings.passage_prefix
        prepared = [f"{prefix} {t}".strip() if prefix else t for t in texts]
        out: list[list[float]] = []
        bs = max(1, self.settings.batch_size)
        for i in range(0, len(prepared), bs):
            out.extend(self._post(prepared[i : i + bs]))
        return out

    def embed_query(self, text: str) -> list[float]:
        prefix = self.settings.query_prefix
        return self._post([f"{prefix} {text}".strip() if prefix else text])[0]

    def close(self) -> None:
        self.client.close()
