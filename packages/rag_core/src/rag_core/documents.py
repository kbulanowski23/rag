"""The shapes that move between ingest, index and retrieval.

One rule: a Chunk must always carry enough provenance to render a citation the
user can act on -- which document, which page, and where in the page. A retrieval
hit the user cannot trace back is worse than no hit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_id(*parts: Any) -> str:
    """Deterministic id. Re-ingesting the same document overwrites its chunks
    rather than duplicating them, which makes ingest idempotent and re-runnable."""
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()[:32]


@dataclass(slots=True)
class Page:
    number: int
    text: str
    # "tika" when the text layer was usable, "ocr" when EasyOCR produced it.
    source: str = "tika"
    ocr_confidence: float | None = None


@dataclass(slots=True)
class Document:
    doc_id: str
    source_uri: str          # where the file came from; shown in citations
    filename: str
    content_type: str = ""
    title: str = ""
    author: str = ""
    created_at: str = ""
    ingested_at: str = field(default_factory=utcnow_iso)
    size_bytes: int = 0
    checksum: str = ""
    pages: list[Page] = field(default_factory=list)
    # Free-form, indexed as a flat object: security labels, business unit, tags.
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    ordinal: int                     # position within the document
    page_start: int
    page_end: int
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0
    extraction_source: str = "tika"
    # Denormalised so a search hit renders without a second fetch.
    filename: str = ""
    title: str = ""
    source_uri: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_os_doc(self) -> dict[str, Any]:
        body = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "ordinal": self.ordinal,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_count": self.token_count,
            "extraction_source": self.extraction_source,
            "filename": self.filename,
            "title": self.title,
            "source_uri": self.source_uri,
            "metadata": self.metadata,
            "indexed_at": utcnow_iso(),
        }
        if self.embedding is not None:
            body["embedding"] = self.embedding
        return body


@dataclass(slots=True)
class SearchHit:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    filename: str = ""
    title: str = ""
    source_uri: str = ""
    page_start: int = 0
    page_end: int = 0
    ordinal: int = 0
    extraction_source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # Provenance for debugging relevance: which retriever(s) surfaced this and
    # at what rank. Rendered in the UI's inspector panel.
    retrievers: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_os(cls, raw: dict[str, Any]) -> "SearchHit":
        src = raw.get("_source", {})
        return cls(
            chunk_id=src.get("chunk_id", raw.get("_id", "")),
            doc_id=src.get("doc_id", ""),
            text=src.get("text", ""),
            score=float(raw.get("_score") or 0.0),
            filename=src.get("filename", ""),
            title=src.get("title", ""),
            source_uri=src.get("source_uri", ""),
            page_start=src.get("page_start", 0),
            page_end=src.get("page_end", 0),
            ordinal=src.get("ordinal", 0),
            extraction_source=src.get("extraction_source", ""),
            metadata=src.get("metadata", {}) or {},
        )

    def citation_label(self) -> str:
        name = self.title or self.filename or self.doc_id
        if self.page_start:
            if self.page_end and self.page_end != self.page_start:
                return f"{name}, pp. {self.page_start}-{self.page_end}"
            return f"{name}, p. {self.page_start}"
        return name
