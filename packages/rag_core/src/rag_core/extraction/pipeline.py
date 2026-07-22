"""Ingest orchestration: bytes in, indexed chunks out.

The interesting decision is *when to OCR*. Running OCR on everything is slow and
degrades quality on documents that already have a good text layer. Running it on
nothing loses every scanned document. So the rule is per-page, not per-file:
Tika extracts first, and any page whose text yield falls below a threshold is
re-extracted with OCR. Mixed documents -- a born-digital contract with scanned
signature pages appended -- are extremely common and this handles them.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field

from rag_core.chunking import Chunker, load_tokenizer
from rag_core.config import Settings
from rag_core.documents import Document, Page, stable_id, utcnow_iso
from rag_core.embeddings.base import EmbeddingProvider
from rag_core.extraction.ocr import OCRClient, OCRError
from rag_core.extraction.tika import TikaClient, TikaError
from rag_core.search.client import ChunkIndexer, IndexAdmin

log = logging.getLogger(__name__)

# Tika metadata keys vary by format; check several for each logical field.
_TITLE_KEYS = ("dc:title", "title", "pdf:docinfo:title")
_AUTHOR_KEYS = ("dc:creator", "meta:author", "Author", "pdf:docinfo:creator")
_DATE_KEYS = ("dcterms:created", "meta:creation-date", "Creation-Date", "pdf:docinfo:created")


def _first(meta: dict, keys: tuple[str, ...]) -> str:
    for k in keys:
        v = meta.get(k)
        if isinstance(v, list):
            v = v[0] if v else None
        if v:
            return str(v)
    return ""


@dataclass(slots=True)
class IngestResult:
    doc_id: str
    filename: str
    pages: int
    chunks_indexed: int
    ocr_pages: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    timings_ms: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.chunks_indexed > 0 and self.failed == 0


class IngestPipeline:
    def __init__(
        self,
        settings: Settings,
        tika: TikaClient,
        ocr: OCRClient | None,
        embedder: EmbeddingProvider,
        indexer: ChunkIndexer,
        admin: IndexAdmin,
    ) -> None:
        self.s = settings
        self.tika = tika
        self.ocr = ocr
        self.embedder = embedder
        self.indexer = indexer
        self.admin = admin
        self.chunker = Chunker(
            settings.chunking,
            load_tokenizer(
                settings.embedding.model_path
                if settings.embedding.provider == "local_onnx"
                else None
            ),
        )

    # -- extraction ----------------------------------------------------------

    def _needs_ocr(self, page: Page) -> bool:
        return len(page.text.strip()) < self.s.ocr.min_chars_per_page

    def _apply_ocr(
        self, data: bytes, filename: str, pages: list[Page], content_type: str
    ) -> tuple[list[Page], int]:
        if self.ocr is None or not self.s.ocr.enabled:
            return pages, 0

        # No pages at all is the strongest signal there is: Tika found no text
        # layer whatsoever. Deriving targets only from pages Tika returned would
        # skip OCR on exactly the documents that need it most. None means "every
        # page" to the OCR service.
        targets = [p.number for p in pages if self._needs_ocr(p)] if pages else None
        if targets == []:
            return pages, 0

        is_pdf = content_type == "application/pdf" or filename.lower().endswith(".pdf")
        is_image = content_type.startswith("image/") or filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
        )

        try:
            if is_pdf:
                results = self.ocr.ocr_pdf(data, pages=targets)
                by_number = {p.number: p for p in pages}
                for number, (text, conf) in results.items():
                    page = by_number.get(number)
                    if page is None:
                        # A scanned PDF with no text layer: OCR is creating the
                        # page, not improving one Tika already found.
                        page = Page(number, "", "tika")
                        by_number[number] = page
                        pages.append(page)
                    if len(text.strip()) > len(page.text.strip()):
                        page.text, page.source, page.ocr_confidence = text, "ocr", conf
                pages.sort(key=lambda p: p.number)
                return pages, len(results)
            if is_image:
                text, conf = self.ocr.ocr_image(data)
                if text.strip():
                    if pages:
                        pages[0].text, pages[0].source, pages[0].ocr_confidence = text, "ocr", conf
                    else:
                        pages = [Page(1, text, "ocr", conf)]
                    return pages, 1
        except OCRError as e:
            # A degraded document beats a failed ingest; the text layer, however
            # thin, is still indexed and the gap is visible in the logs.
            log.warning("OCR failed for %s, keeping Tika output: %s", filename, e)
        return pages, 0

    def extract(
        self, data: bytes, filename: str, source_uri: str = "", content_type: str = "",
        metadata: dict | None = None,
    ) -> tuple[Document, int]:
        meta = self.tika.extract_metadata(data, filename, content_type)
        detected = content_type or str(meta.get("Content-Type", "") or "")
        if isinstance(detected, list):
            detected = detected[0] if detected else ""

        pages = self.tika.extract_pages(data, filename, content_type)
        pages, ocr_count = self._apply_ocr(data, filename, pages, detected)

        checksum = hashlib.sha256(data).hexdigest()
        doc = Document(
            # Identity is the source path, not the content hash: re-ingesting an
            # edited file must replace the old version, not sit beside it.
            doc_id=stable_id(source_uri or filename),
            source_uri=source_uri or filename,
            filename=filename,
            content_type=detected,
            title=_first(meta, _TITLE_KEYS) or filename,
            author=_first(meta, _AUTHOR_KEYS),
            created_at=_first(meta, _DATE_KEYS),
            ingested_at=utcnow_iso(),
            size_bytes=len(data),
            checksum=checksum,
            pages=pages,
            metadata=metadata or {},
        )
        return doc, ocr_count

    # -- full pipeline -------------------------------------------------------

    def ingest_bytes(
        self, data: bytes, filename: str, source_uri: str = "", content_type: str = "",
        metadata: dict | None = None, refresh: bool = False,
    ) -> IngestResult:
        timings: dict[str, int] = {}
        errors: list[str] = []

        t0 = time.perf_counter()
        try:
            doc, ocr_count = self.extract(data, filename, source_uri, content_type, metadata)
        except TikaError as e:
            return IngestResult(
                doc_id="", filename=filename, pages=0, chunks_indexed=0,
                failed=1, errors=[str(e)],
            )
        timings["extract"] = int((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        chunks = self.chunker.chunk(doc)
        timings["chunk"] = int((time.perf_counter() - t0) * 1000)
        if not chunks:
            return IngestResult(
                doc_id=doc.doc_id, filename=filename, pages=len(doc.pages),
                chunks_indexed=0, ocr_pages=ocr_count, failed=1,
                errors=["no extractable text"], timings_ms=timings,
            )

        t0 = time.perf_counter()
        vectors = self.embedder.embed_passages([c.text for c in chunks])
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector
        timings["embed"] = int((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        # Clear the previous version first: an edited document with fewer chunks
        # would otherwise leave stale trailing chunks in the index forever.
        removed = self.admin.delete_by_doc_id(doc.doc_id)
        if removed:
            log.info("replaced %d existing chunks for %s", removed, doc.doc_id)
        indexed, bulk_errors = self.indexer.index_chunks(chunks, refresh=refresh)
        timings["index"] = int((time.perf_counter() - t0) * 1000)
        errors.extend(str(e)[:300] for e in bulk_errors[:5])

        return IngestResult(
            doc_id=doc.doc_id,
            filename=filename,
            pages=len(doc.pages),
            chunks_indexed=indexed,
            ocr_pages=ocr_count,
            failed=len(bulk_errors),
            errors=errors,
            timings_ms=timings,
        )
