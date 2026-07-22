"""The LLM seam.

Everything above this module talks only to `LLMProvider`. Swapping Ollama for
Azure OpenAI for a self-hosted vLLM gateway is a config change; no calling code
changes. Adapters are thin on purpose -- they translate a request and normalise a
response, nothing more. No retries-with-backoff cleverness, no prompt rewriting.
"""

from __future__ import annotations

import ssl
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

import httpx

from rag_core.config import LLMSettings

Role = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class Message:
    role: Role
    content: str


@dataclass(slots=True)
class Completion:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    raw: dict = field(default_factory=dict)


class LLMError(RuntimeError):
    """Any failure reaching or parsing the model endpoint."""


class LLMProvider(ABC):
    """Implement these two methods and the rest of the platform works."""

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

    @abstractmethod
    async def complete(self, messages: list[Message], **overrides) -> Completion: ...

    @abstractmethod
    def stream(self, messages: list[Message], **overrides) -> AsyncIterator[str]: ...

    async def health(self) -> bool:
        try:
            await self.complete([Message("user", "ping")], max_tokens=1)
            return True
        except Exception:
            return False

    # -- shared HTTP plumbing ------------------------------------------------

    def _verify(self) -> ssl.SSLContext | bool | str:
        # A corporate CA bundle is the normal case behind TLS interception.
        if not self.settings.verify_ssl:
            return False
        return self.settings.ca_bundle or True

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.timeout_s, connect=10.0),
                verify=self._verify(),
                headers=self.settings.extra_headers or None,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _param(self, name: str, overrides: dict):
        return overrides.get(name, getattr(self.settings, name))
