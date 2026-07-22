"""Chunking is the part most likely to be quietly wrong, so it gets real tests.

These use ApproxTokenizer and need no model files or running services.
"""

from __future__ import annotations

import pytest

from rag_core.chunking import ApproxTokenizer, Chunker
from rag_core.config import ChunkingSettings
from rag_core.documents import Document, Page


def make_doc(pages: list[str]) -> Document:
    return Document(
        doc_id="doc1",
        source_uri="test.pdf",
        filename="test.pdf",
        title="Test",
        pages=[Page(number=i, text=t) for i, t in enumerate(pages, start=1)],
    )


def settings(**kw) -> ChunkingSettings:
    return ChunkingSettings(**{"max_tokens": 50, "overlap_tokens": 10,
                               "min_chunk_chars": 20, **kw})


def test_empty_document_yields_nothing():
    chunker = Chunker(settings(), ApproxTokenizer())
    assert chunker.chunk(make_doc([""])) == []


def test_short_document_is_one_chunk():
    text = "The retention period for claims documentation is seven years."
    chunks = Chunker(settings(), ApproxTokenizer()).chunk(make_doc([text]))
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].page_start == 1


def test_no_chunk_exceeds_the_token_budget():
    # The whole point: a chunk longer than the encoder's window is silently
    # truncated at embed time, so the tail never makes it into the vector.
    para = " ".join(f"word{i}" for i in range(60))
    doc = make_doc(["\n\n".join([para] * 10)])
    s = settings(max_tokens=50)
    chunks = Chunker(s, ApproxTokenizer()).chunk(doc)
    assert chunks
    for c in chunks:
        assert c.token_count <= s.max_tokens, f"chunk {c.ordinal} is {c.token_count} tokens"


def test_chunks_overlap():
    paras = [f"Paragraph number {i} contains some distinctive filler text here." for i in range(12)]
    doc = make_doc(["\n\n".join(paras)])
    chunks = Chunker(settings(max_tokens=40, overlap_tokens=15), ApproxTokenizer()).chunk(doc)
    assert len(chunks) >= 2
    # Adjacent chunks should share text, or a fact on the boundary is lost.
    shared = any(
        any(line in chunks[i + 1].text for line in chunks[i].text.split("\n\n") if len(line) > 20)
        for i in range(len(chunks) - 1)
    )
    assert shared, "expected adjacent chunks to share content"


def test_page_provenance_is_preserved():
    doc = make_doc(["First page content here, long enough to survive.",
                    "Second page content here, also long enough."])
    chunks = Chunker(settings(max_tokens=15), ApproxTokenizer()).chunk(doc)
    pages = {c.page_start for c in chunks}
    assert pages.issubset({1, 2})
    assert all(c.page_start >= 1 for c in chunks)


def test_ordinals_are_sequential():
    doc = make_doc(["\n\n".join(f"Paragraph {i} with enough text to count." for i in range(20))])
    chunks = Chunker(settings(max_tokens=30), ApproxTokenizer()).chunk(doc)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_deterministic():
    # Re-ingesting a document must overwrite, not duplicate.
    doc = make_doc(["Some stable content that will be chunked identically twice."])
    a = Chunker(settings(), ApproxTokenizer()).chunk(doc)
    b = Chunker(settings(), ApproxTokenizer()).chunk(doc)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_oversized_single_sentence_is_split():
    monster = " ".join(f"token{i}" for i in range(500))  # no sentence breaks at all
    chunks = Chunker(settings(max_tokens=50), ApproxTokenizer()).chunk(make_doc([monster]))
    assert len(chunks) > 1
    assert all(c.token_count <= 50 for c in chunks)


def test_tiny_fragments_are_dropped():
    chunks = Chunker(settings(min_chunk_chars=100), ApproxTokenizer()).chunk(make_doc(["hi"]))
    assert chunks == []
