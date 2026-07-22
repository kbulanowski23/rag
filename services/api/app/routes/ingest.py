from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.deps import Container, get_container
from app.schemas import IngestResponse

log = logging.getLogger(__name__)
router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    source_uri: str = Form(""),
    metadata: str = Form(""),
    refresh: bool = Form(True),
    c: Container = Depends(get_container),
) -> IngestResponse:
    """Synchronous single-file ingest.

    Fine for the UI's upload button and for testing. Bulk loading of a document
    corpus should go through services/worker instead, which does not tie up an
    API pod for the duration of an OCR run.
    """
    data = await file.read()
    max_bytes = c.settings.api.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"file is {len(data) // 1048576} MB; limit is {c.settings.api.max_upload_mb} MB",
        )
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    extra: dict = {}
    if metadata:
        try:
            extra = json.loads(metadata)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"metadata is not valid JSON: {e}") from e

    # Extraction, embedding and indexing are all blocking; keep them off the
    # event loop or a single large PDF stalls every other request in the pod.
    result = await run_in_threadpool(
        c.ingest.ingest_bytes,
        data,
        file.filename or "upload",
        source_uri or (file.filename or "upload"),
        file.content_type or "",
        extra,
        refresh,
    )

    return IngestResponse(
        doc_id=result.doc_id, filename=result.filename, pages=result.pages,
        chunks_indexed=result.chunks_indexed, ocr_pages=result.ocr_pages,
        failed=result.failed, errors=result.errors, timings_ms=result.timings_ms,
    )


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, c: Container = Depends(get_container)) -> dict:
    deleted = await run_in_threadpool(c.admin.delete_by_doc_id, doc_id)
    return {"doc_id": doc_id, "chunks_deleted": deleted}


@router.get("/index/stats")
async def index_stats(c: Container = Depends(get_container)) -> dict:
    return await run_in_threadpool(c.admin.stats)
