"""Token-aware chunking with overlap.

Chunks are measured with the *embedding model's own tokenizer*, not a character
count and not a different model's tokenizer. Getting this wrong is the usual
cause of silent truncation: text past the encoder's max_tokens is dropped by the
tokenizer, so the tail of every oversized chunk is simply not represented in its
vector, and nobody notices because indexing still succeeds.

Strategy, in order of preference:
  1. Split on paragraph boundaries and pack paragraphs up to the token budget.
  2. A paragraph larger than the budget is split on sentence boundaries.
  3. A sentence larger than the budget is split on the token grid.
Overlap is applied between adjacent chunks so a fact spanning a boundary is
retrievable from either side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_core.config import ChunkingSettings
from rag_core.documents import Chunk, Document, stable_id

_PARAGRAPH_RE = re.compile(r"\n\s*\n+")
# Sentence terminator followed by whitespace and a capital/quote/digit. Good
# enough for prose and does not need an NLP dependency.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_WS_RE = re.compile(r"[ \t]+")


class Tokenizerish:
    """Minimal protocol the chunker needs from a tokenizer."""

    def count(self, text: str) -> int: ...
    def split_by_tokens(self, text: str, size: int, overlap: int) -> list[str]: ...


class HFTokenizer(Tokenizerish):
    """Wraps a `tokenizers.Tokenizer` loaded from the embedding model."""

    def __init__(self, tokenizer) -> None:
        self._t = tokenizer

    def count(self, text: str) -> int:
        # no_special: we are budgeting content, and [CLS]/[SEP] are accounted for
        # by leaving headroom in max_tokens.
        return len(self._t.encode(text, add_special_tokens=False).ids)

    def split_by_tokens(self, text: str, size: int, overlap: int) -> list[str]:
        enc = self._t.encode(text, add_special_tokens=False)
        ids = enc.ids
        if len(ids) <= size:
            return [text]
        out: list[str] = []
        step = max(1, size - overlap)
        for start in range(0, len(ids), step):
            window = ids[start : start + size]
            if not window:
                break
            out.append(self._t.decode(window, skip_special_tokens=True))
            if start + size >= len(ids):
                break
        return out


class ApproxTokenizer(Tokenizerish):
    """Fallback when no tokenizer is available (tests, remote embedders).

    ~4 characters per token is the usual English approximation. It is only used
    to make chunking degrade gracefully, never to make a correctness claim.
    """

    CHARS_PER_TOKEN = 4

    def count(self, text: str) -> int:
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def split_by_tokens(self, text: str, size: int, overlap: int) -> list[str]:
        span = size * self.CHARS_PER_TOKEN
        step = max(1, (size - overlap) * self.CHARS_PER_TOKEN)
        return [text[i : i + span] for i in range(0, len(text), step) if text[i : i + span]]


def load_tokenizer(model_path: str | None) -> Tokenizerish:
    if not model_path:
        return ApproxTokenizer()
    try:
        from pathlib import Path

        from tokenizers import Tokenizer

        p = Path(model_path) / "tokenizer.json"
        if not p.is_file():
            return ApproxTokenizer()
        return HFTokenizer(Tokenizer.from_file(str(p)))
    except Exception:
        return ApproxTokenizer()


@dataclass(slots=True)
class _Piece:
    text: str
    tokens: int
    page: int
    char_start: int


def _normalise(text: str) -> str:
    # Tika emits ragged intra-line whitespace from PDF layout; collapse it but
    # keep newlines, which carry the paragraph structure we chunk on.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _segments(text: str, respect_paragraphs: bool) -> list[tuple[str, int]]:
    """Yield (segment, offset_in_text), splitting on paragraphs then sentences."""
    if not respect_paragraphs:
        return [(text, 0)]
    out: list[tuple[str, int]] = []
    cursor = 0
    for para in _PARAGRAPH_RE.split(text):
        if not para.strip():
            cursor += len(para)
            continue
        idx = text.find(para, cursor)
        if idx < 0:
            idx = cursor
        out.append((para, idx))
        cursor = idx + len(para)
    return out


class Chunker:
    def __init__(self, settings: ChunkingSettings, tokenizer: Tokenizerish | None = None) -> None:
        self.s = settings
        self.tokenizer = tokenizer or ApproxTokenizer()

    def _atoms(self, doc: Document) -> list[_Piece]:
        """Break the document into units no larger than the token budget,
        tagged with the page they came from."""
        atoms: list[_Piece] = []
        budget = self.s.max_tokens

        for page in doc.pages:
            text = _normalise(page.text)
            if not text:
                continue
            for seg, offset in _segments(text, self.s.respect_paragraphs):
                if self.tokenizer.count(seg) <= budget:
                    atoms.append(_Piece(seg, self.tokenizer.count(seg), page.number, offset))
                    continue
                # Too big: sentences.
                cursor = offset
                for sent in _SENTENCE_RE.split(seg):
                    if not sent.strip():
                        continue
                    n = self.tokenizer.count(sent)
                    if n <= budget:
                        atoms.append(_Piece(sent, n, page.number, cursor))
                        cursor += len(sent)
                        continue
                    # Still too big: hard split on the token grid.
                    for part in self.tokenizer.split_by_tokens(
                        sent, budget, self.s.overlap_tokens
                    ):
                        atoms.append(
                            _Piece(part, self.tokenizer.count(part), page.number, cursor)
                        )
                        cursor += len(part)
        return atoms

    def chunk(self, doc: Document) -> list[Chunk]:
        atoms = self._atoms(doc)
        if not atoms:
            return []

        budget = self.s.max_tokens
        overlap = self.s.overlap_tokens
        chunks: list[Chunk] = []
        window: list[_Piece] = []
        window_tokens = 0

        def flush() -> None:
            nonlocal window, window_tokens
            if not window:
                return
            text = "\n\n".join(p.text for p in window).strip()
            if len(text) >= self.s.min_chunk_chars:
                ordinal = len(chunks)
                first, last = window[0], window[-1]
                chunks.append(
                    Chunk(
                        chunk_id=stable_id(doc.doc_id, ordinal, text[:200]),
                        doc_id=doc.doc_id,
                        text=text,
                        ordinal=ordinal,
                        page_start=first.page,
                        page_end=last.page,
                        char_start=first.char_start,
                        char_end=last.char_start + len(last.text),
                        token_count=window_tokens,
                        extraction_source=self._page_source(doc, first.page),
                        filename=doc.filename,
                        title=doc.title or doc.filename,
                        source_uri=doc.source_uri,
                        metadata=dict(doc.metadata),
                    )
                )
            # Carry the tail of this window into the next one so a fact split
            # across the boundary stays retrievable from both chunks.
            carried: list[_Piece] = []
            carried_tokens = 0
            for piece in reversed(window):
                if carried_tokens + piece.tokens > overlap:
                    break
                carried.insert(0, piece)
                carried_tokens += piece.tokens
            window = carried
            window_tokens = carried_tokens

        for atom in atoms:
            if window and window_tokens + atom.tokens > budget:
                flush()
                # A single atom bigger than the carried overlap can still exceed
                # the budget; drop the carry rather than emit an oversized chunk.
                if window_tokens + atom.tokens > budget:
                    window, window_tokens = [], 0
            window.append(atom)
            window_tokens += atom.tokens

        # Final flush must not re-carry, or we loop forever.
        if window:
            text = "\n\n".join(p.text for p in window).strip()
            if len(text) >= self.s.min_chunk_chars:
                ordinal = len(chunks)
                first, last = window[0], window[-1]
                chunks.append(
                    Chunk(
                        chunk_id=stable_id(doc.doc_id, ordinal, text[:200]),
                        doc_id=doc.doc_id,
                        text=text,
                        ordinal=ordinal,
                        page_start=first.page,
                        page_end=last.page,
                        char_start=first.char_start,
                        char_end=last.char_start + len(last.text),
                        token_count=window_tokens,
                        extraction_source=self._page_source(doc, first.page),
                        filename=doc.filename,
                        title=doc.title or doc.filename,
                        source_uri=doc.source_uri,
                        metadata=dict(doc.metadata),
                    )
                )
        return chunks

    @staticmethod
    def _page_source(doc: Document, page_number: int) -> str:
        for p in doc.pages:
            if p.number == page_number:
                return p.source
        return "tika"
