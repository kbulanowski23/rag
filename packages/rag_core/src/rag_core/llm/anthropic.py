"""Anthropic Messages API, spoken over plain httpx.

Included so the platform is not locked to one vendor, and because it is a useful
quality baseline while developing at home. It has no place in the air-gapped
deployment and pulls no SDK dependency.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from rag_core.llm.base import Completion, LLMError, LLMProvider, Message

DEFAULT_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def _url(self) -> str:
        base = self.settings.base_url or DEFAULT_BASE_URL
        # Tolerate a base_url that already carries the /v1 suffix.
        base = base[:-3].rstrip("/") if base.endswith("/v1") else base
        return f"{base}/v1/messages"

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.settings.api_key,
            "anthropic-version": API_VERSION,
        }

    def _body(self, messages: list[Message], overrides: dict, stream: bool) -> dict:
        # Anthropic takes the system prompt as a top-level field, not a message.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        ]
        body = {
            "model": self._param("model", overrides),
            "messages": turns,
            "max_tokens": self._param("max_tokens", overrides),
            "temperature": self._param("temperature", overrides),
            "stream": stream,
        }
        if system:
            body["system"] = system
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

        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            model=data.get("model", self.settings.model),
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            finish_reason=data.get("stop_reason", ""),
            raw=data,
        )

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
                    try:
                        event = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "content_block_delta":
                        if text := event.get("delta", {}).get("text"):
                            yield text
        except httpx.HTTPError as e:
            raise LLMError(f"{self.name}: {e}") from e
