"""LLM provider registry.

Adding a provider is: write the adapter, add one line to _PROVIDERS. Callers use
get_llm() and never import a concrete adapter.
"""

from __future__ import annotations

from rag_core.config import LLMSettings, get_settings
from rag_core.llm.anthropic import AnthropicProvider
from rag_core.llm.azure_openai import AzureOpenAIProvider
from rag_core.llm.base import Completion, LLMError, LLMProvider, Message
from rag_core.llm.echo import EchoProvider
from rag_core.llm.openai_compatible import OpenAICompatibleProvider

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "openai_compatible": OpenAICompatibleProvider,
    "azure_openai": AzureOpenAIProvider,
    "anthropic": AnthropicProvider,
    "echo": EchoProvider,
}

_instance: LLMProvider | None = None


def build_llm(settings: LLMSettings) -> LLMProvider:
    try:
        cls = _PROVIDERS[settings.provider]
    except KeyError:
        raise LLMError(
            f"unknown LLM provider {settings.provider!r}; "
            f"available: {', '.join(sorted(_PROVIDERS))}"
        ) from None
    return cls(settings)


def get_llm() -> LLMProvider:
    """Process-wide provider built from the environment."""
    global _instance
    if _instance is None:
        _instance = build_llm(get_settings().llm)
    return _instance


async def close_llm() -> None:
    global _instance
    if _instance is not None:
        await _instance.aclose()
        _instance = None


__all__ = [
    "Completion", "LLMError", "LLMProvider", "Message",
    "build_llm", "get_llm", "close_llm",
]
