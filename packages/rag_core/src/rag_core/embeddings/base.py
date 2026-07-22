from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingError(RuntimeError):
    pass


class EmbeddingProvider(ABC):
    """Embeddings are asymmetric: a query and a passage may need different
    prefixes even with the same weights (bge, e5 and friends). Callers must say
    which side they are on rather than passing raw text to one method."""

    dim: int

    @abstractmethod
    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...

    def close(self) -> None:
        return None
