"""Azure OpenAI.

Differs from vanilla OpenAI in exactly three ways, all handled here:
  * the deployment name lives in the path, not the body (`model` holds it)
  * auth is the `api-key` header, not `Authorization: Bearer`
  * an `api-version` query parameter is mandatory

Newer reasoning-family deployments reject `temperature` and rename the token
limit to `max_completion_tokens`; both are handled below so that GPT-5.x
deployments work without a code change.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from rag_core.llm.openai_compatible import OpenAICompatibleProvider
from rag_core.llm.base import Message

# Deployments whose backing model rejects `temperature` / `max_tokens`.
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


class AzureOpenAIProvider(OpenAICompatibleProvider):
    name = "azure_openai"

    def _url(self) -> str:
        return (
            f"{self.settings.base_url}/openai/deployments/{self.settings.model}"
            f"/chat/completions?api-version={self.settings.api_version}"
        )

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.settings.api_key:
            h["api-key"] = self.settings.api_key
        return h

    def _is_reasoning(self, model: str) -> bool:
        m = model.lower()
        return any(p in m for p in _REASONING_PREFIXES)

    def _body(self, messages: list[Message], overrides: dict, stream: bool) -> dict:
        body = super()._body(messages, overrides, stream)
        # The deployment is in the URL; sending it again is harmless on some
        # api-versions and rejected on others, so drop it.
        model = body.pop("model")
        if self._is_reasoning(model):
            body["max_completion_tokens"] = body.pop("max_tokens")
            body.pop("temperature", None)
            body.pop("top_p", None)
        return body
