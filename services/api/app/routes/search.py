from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from app.deps import Container, get_container
from app.schemas import SearchRequest, SearchResponse, SourceOut

log = logging.getLogger(__name__)
router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, c: Container = Depends(get_container)) -> SearchResponse:
    """Retrieval only, no generation.

    This endpoint is how you debug relevance. It returns the per-retriever ranks
    on every hit, so you can see whether BM25 or kNN found a document and where
    fusion placed it.
    """
    retriever = c.retriever
    if req.fusion and req.fusion != retriever.r.fusion:
        # Per-request override without mutating shared state.
        retriever = type(retriever)(
            retriever.client, retriever.embedder,
            retriever.r.model_copy(update={"fusion": req.fusion}),
            retriever.os,
        )

    t0 = time.perf_counter()
    try:
        hits = retriever.retrieve(req.query, k=req.k, filters=req.filters)
    except Exception as e:
        log.exception("search failed")
        raise HTTPException(status_code=502, detail=f"search backend error: {e}") from e
    took = int((time.perf_counter() - t0) * 1000)

    return SearchResponse(
        query=req.query,
        took_ms=took,
        hits=[
            SourceOut(
                index=i, chunk_id=h.chunk_id, doc_id=h.doc_id,
                title=h.title or h.filename, filename=h.filename,
                source_uri=h.source_uri, page_start=h.page_start, page_end=h.page_end,
                score=round(h.score, 5), extraction_source=h.extraction_source,
                retrievers=h.retrievers, text=h.text,
            )
            for i, h in enumerate(hits, start=1)
        ],
    )
