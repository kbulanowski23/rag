from __future__ import annotations

from rag_core.config import EmbeddingSettings, get_settings
from rag_core.embeddings.base import EmbeddingError, EmbeddingProvider

# Providers are imported lazily inside build_embedder rather than here. Importing
# them eagerly would pull onnxruntime and numpy into every process that touches
# `rag_core.embeddings.base` -- including ones that only need the type, such as
# the retriever's constructor signature. Keep this module import-light.
_PROVIDER_NAMES = ("local_onnx", "openai_compatible")

_instance: EmbeddingProvider | None = None


def build_embedder(settings: EmbeddingSettings) -> EmbeddingProvider:
    if settings.provider == "local_onnx":
        from rag_core.embeddings.local_onnx import LocalOnnxEmbedder

        return LocalOnnxEmbedder(settings)
    if settings.provider == "openai_compatible":
        from rag_core.embeddings.remote_openai import RemoteOpenAIEmbedder

        return RemoteOpenAIEmbedder(settings)
    raise EmbeddingError(
        f"unknown embedding provider {settings.provider!r}; "
        f"available: {', '.join(_PROVIDER_NAMES)}"
    )


def get_embedder() -> EmbeddingProvider:
    """Loading an ONNX session costs a second or two; do it once per process."""
    global _instance
    if _instance is None:
        _instance = build_embedder(get_settings().embedding)
    return _instance


def close_embedder() -> None:
    global _instance
    if _instance is not None:
        _instance.close()
        _instance = None


__all__ = [
    "EmbeddingError", "EmbeddingProvider",
    "build_embedder", "get_embedder", "close_embedder",
]
