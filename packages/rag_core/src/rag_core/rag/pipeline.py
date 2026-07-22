"""The query-side orchestration. This is the whole of what a framework would
have hidden from you: retrieve, budget, prompt, generate, attach citations."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from rag_core.config import Settings
from rag_core.documents import SearchHit
from rag_core.llm.base import LLMProvider, Message
from rag_core.rag.prompt import build_messages, load_system_prompt
from rag_core.search.retriever import HybridRetriever

log = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")


@dataclass(slots=True)
class RagAnswer:
    answer: str
    sources: list[SearchHit] = field(default_factory=list)
    cited_indices: list[int] = field(default_factory=list)
    timings_ms: dict[str, int] = field(default_factory=dict)
    model: str = ""


def extract_cited(text: str, used: list[SearchHit]) -> list[int]:
    """Which excerpt numbers did the model actually reference?

    The UI dims uncited sources rather than hiding them: an uncited-but-retrieved
    document is useful signal to the user, and it exposes retrieval quality
    instead of concealing it.
    """
    seen: list[int] = []
    for m in _CITATION_RE.finditer(text):
        n = int(m.group(1))
        if 1 <= n <= len(used) and n not in seen:
            seen.append(n)
    return seen


class RagPipeline:
    def __init__(
        self, retriever: HybridRetriever, llm: LLMProvider, settings: Settings
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings
        self.system_prompt = load_system_prompt(settings.api.system_prompt_path)

    def _retrieve(
        self, question: str, k: int | None, filters: dict | None
    ) -> tuple[list[SearchHit], int]:
        t0 = time.perf_counter()
        hits = self.retriever.retrieve(question, k=k, filters=filters)
        ms = int((time.perf_counter() - t0) * 1000)
        log.info("retrieved %d hits in %dms for %r", len(hits), ms, question[:80])
        return hits, ms

    async def answer(
        self,
        question: str,
        k: int | None = None,
        filters: dict | None = None,
        history: list[Message] | None = None,
    ) -> RagAnswer:
        hits, retrieve_ms = self._retrieve(question, k, filters)
        messages, used = build_messages(
            question, hits, self.settings.retrieval, self.system_prompt, history
        )

        t0 = time.perf_counter()
        completion = await self.llm.complete(messages)
        gen_ms = int((time.perf_counter() - t0) * 1000)

        return RagAnswer(
            answer=completion.text,
            sources=used,
            cited_indices=extract_cited(completion.text, used),
            timings_ms={"retrieval": retrieve_ms, "generation": gen_ms},
            model=completion.model,
        )

    async def stream(
        self,
        question: str,
        k: int | None = None,
        filters: dict | None = None,
        history: list[Message] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yields event dicts for the transport layer to serialise as SSE.

        Sources are emitted *before* the first token so the UI can render the
        citation panel while the answer is still streaming -- the retrieval is
        already done at that point, and showing it immediately makes the wait
        feel like progress rather than a hang.
        """
        hits, retrieve_ms = self._retrieve(question, k, filters)
        messages, used = build_messages(
            question, hits, self.settings.retrieval, self.system_prompt, history
        )

        yield {
            "type": "sources",
            "sources": [
                {
                    "index": i,
                    "chunk_id": h.chunk_id,
                    "doc_id": h.doc_id,
                    "title": h.title or h.filename,
                    "filename": h.filename,
                    "source_uri": h.source_uri,
                    "page_start": h.page_start,
                    "page_end": h.page_end,
                    "score": round(h.score, 5),
                    "retrievers": h.retrievers,
                    "extraction_source": h.extraction_source,
                    "text": h.text,
                }
                for i, h in enumerate(used, start=1)
            ],
            "retrieval_ms": retrieve_ms,
        }

        t0 = time.perf_counter()
        buffer: list[str] = []
        try:
            async for token in self.llm.stream(messages):
                buffer.append(token)
                yield {"type": "token", "text": token}
        except Exception as e:
            log.exception("generation failed")
            yield {"type": "error", "message": str(e)}
            return

        full = "".join(buffer)
        yield {
            "type": "done",
            "cited": extract_cited(full, used),
            "timings_ms": {
                "retrieval": retrieve_ms,
                "generation": int((time.perf_counter() - t0) * 1000),
            },
        }
