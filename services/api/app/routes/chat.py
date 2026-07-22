from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.deps import Container, get_container
from app.schemas import ChatRequest, ChatResponse, SourceOut
from rag_core.llm.base import LLMError, Message

log = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


def _to_messages(history) -> list[Message]:
    return [Message(role=t.role, content=t.content) for t in history]


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, c: Container = Depends(get_container)) -> ChatResponse:
    """Non-streaming answer. Use /chat/stream for the UI."""
    try:
        result = await c.rag.answer(
            req.question, k=req.k, filters=req.filters, history=_to_messages(req.history)
        )
    except LLMError as e:
        # 502: the failure is upstream, not in the request. Distinguishing this
        # from a 500 is what tells an operator to look at the model endpoint.
        raise HTTPException(status_code=502, detail=f"LLM error: {e}") from e

    return ChatResponse(
        answer=result.answer,
        sources=[
            SourceOut(
                index=i, chunk_id=h.chunk_id, doc_id=h.doc_id,
                title=h.title or h.filename, filename=h.filename,
                source_uri=h.source_uri, page_start=h.page_start, page_end=h.page_end,
                score=round(h.score, 5), extraction_source=h.extraction_source,
                retrievers=h.retrievers, text=h.text,
            )
            for i, h in enumerate(result.sources, start=1)
        ],
        cited=result.cited_indices,
        model=result.model,
        timings_ms=result.timings_ms,
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, c: Container = Depends(get_container)):
    """Server-Sent Events.

    Event order is: one `sources` event, then many `token` events, then one
    `done`. The client renders citations from the first event immediately so the
    user sees what was found before the answer finishes generating.
    """

    async def gen():
        try:
            async for event in c.rag.stream(
                req.question, k=req.k, filters=req.filters, history=_to_messages(req.history)
            ):
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
        except Exception as e:
            log.exception("stream failed")
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            # OpenShift's default router buffers responses, which turns a token
            # stream into one lump at the end. This disables that.
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
