"""OpenAI-compatible chat completions.

Covers Ollama, vLLM, TGI's OpenAI router, llama.cpp server, LiteLLM, and most
internal gateways. This is the adapter to prefer at work if there is any choice:
it is the one exercised during local development.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from rag_core.llm.base import Completion, LLMError, LLMProvider, Message


class OpenAICompatibleProvider(LLMProvider):
    name = "openai_compatible"

    def _url(self) -> str:
        return f"{self.settings.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.settings.api_key:
            h["Authorization"] = f"Bearer {self.settings.api_key}"
        return h

    def _body(self, messages: list[Message], overrides: dict, stream: bool) -> dict:
        body = {
            "model": self._param("model", overrides),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._param("temperature", overrides),
            "max_tokens": self._param("max_tokens", overrides),
            "top_p": self._param("top_p", overrides),
            "stream": stream,
        }
        # Vendor-specific parameters, merged last so they can also override the
        # standard fields. The OpenAI wire format is a lower bound, not a
        # ceiling: gateways expose real capability through extra keys, and the
        # most valuable one here is turning a reasoning model's thinking off.
        # Measured on qwen3-8b: 10.2s/257 tokens -> 0.9s/22 tokens.
        if self.settings.extra_body:
            body.update(self.settings.extra_body)
        return body

    async def complete(self, messages: list[Message], **overrides) -> Completion:
        try:
            r = await self.client.post(
                self._url(), headers=self._headers(),
                json=self._body(messages, overrides, stream=False),
            )
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as e:
            raise LLMError(f"{self.name}: HTTP {e.response.status_code}: {e.response.text[:500]}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"{self.name}: {e}") from e

        try:
            choice = data["choices"][0]
            usage = data.get("usage") or {}
            return Completion(
                text=choice["message"]["content"] or "",
                model=data.get("model", self.settings.model),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                finish_reason=choice.get("finish_reason", ""),
                raw=data,
            )
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"{self.name}: unexpected response shape: {str(data)[:500]}") from e

    async def stream(self, messages: list[Message], **overrides) -> AsyncIterator[str]:
        body = self._body(messages, overrides, stream=True)
        try:
            async with self.client.stream(
                "POST", self._url(), headers=self._headers(), json=body
            ) as r:
                if r.status_code >= 400:
                    detail = (await r.aread()).decode("utf-8", "replace")[:500]
                    raise LLMError(f"{self.name}: HTTP {r.status_code}: {detail}")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = chunk["choices"][0].get("delta", {})
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue  # keepalives and vendor-specific frames
                    if text := delta.get("content"):
                        yield text
        except httpx.HTTPError as e:
            raise LLMError(f"{self.name}: {e}") from e
