"""A provider that calls nothing.

Lets the whole ingest + retrieval stack be exercised (and CI run) with no model
endpoint available at all. It echoes back the assembled context so you can see
exactly what the retriever fed the model.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from rag_core.llm.base import Completion, LLMProvider, Message


class EchoProvider(LLMProvider):
    name = "echo"

    def _render(self, messages: list[Message]) -> str:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        return f"[echo provider — no LLM configured]\n\n{last_user}"

    async def complete(self, messages: list[Message], **overrides) -> Completion:
        return Completion(text=self._render(messages), model="echo", finish_reason="stop")

    async def stream(self, messages: list[Message], **overrides) -> AsyncIterator[str]:
        for word in self._render(messages).split(" "):
            yield word + " "

    async def health(self) -> bool:
        return True
