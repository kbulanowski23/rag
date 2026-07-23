"""`classification` and `acl` are indexed, first-class fields -- not entries in
`metadata`.

That distinction is the whole point. `metadata` is mapped `enabled: false`, so
it is stored and returned but never indexed: a filter on `metadata.classification`
matches NOTHING and reports no error. For a display field that is a curiosity.
For an access control it is the worst failure mode there is, because the system
looks like it is working while enforcing nothing.

Mappings are also fixed at index creation, so these fields have to exist before
a corpus is loaded even though enforcement is not built yet -- adding them later
means reindexing everything.
"""

from __future__ import annotations

from rag_core.config import OpenSearchSettings
from rag_core.documents import Chunk, Document, Page, SearchHit
from rag_core.search.index_mapping import build_mapping


def mapping_props() -> dict:
    m = build_mapping(OpenSearchSettings(), dim=384)
    return m["mappings"]["properties"]


# -- the mapping --------------------------------------------------------------

def test_classification_and_acl_are_indexed_keywords():
    props = mapping_props()
    for field in ("classification", "acl"):
        assert props[field]["type"] == "keyword", f"{field} must be a filterable keyword"


def test_metadata_remains_unindexed():
    # Guards the reason the dedicated fields exist. If someone "fixes" this by
    # enabling metadata, a mapping explosion follows: every key any document
    # ever carries becomes a field in the index.
    assert mapping_props()["metadata"]["enabled"] is False


# -- propagation --------------------------------------------------------------

def test_labels_reach_the_indexed_document():
    body = Chunk(
        chunk_id="c1", doc_id="d1", text="x", ordinal=0, page_start=1, page_end=1,
        classification="SECRET", acl=["team-claims", "cleared-uk"],
    ).to_os_doc()
    assert body["classification"] == "SECRET"
    assert body["acl"] == ["team-claims", "cleared-uk"]


def test_labels_survive_the_round_trip_out_of_opensearch():
    hit = SearchHit.from_os({
        "_id": "c1",
        "_score": 1.0,
        "_source": {"chunk_id": "c1", "doc_id": "d1", "text": "x",
                    "classification": "SECRET", "acl": ["team-claims"]},
    })
    assert hit.classification == "SECRET"
    assert hit.acl == ["team-claims"]


def test_unlabelled_chunks_are_empty_not_missing():
    # An unlabelled corpus must not blow up. Empty is a value the route layer
    # can then decide to treat as deny-by-default.
    hit = SearchHit.from_os({"_source": {"chunk_id": "c", "doc_id": "d", "text": "t"}})
    assert hit.classification == ""
    assert hit.acl == []


def test_chunks_inherit_the_document_label():
    from rag_core.chunking import Chunker
    from rag_core.config import ChunkingSettings

    doc = Document(
        doc_id="d1", source_uri="u", filename="f.txt", title="t",
        pages=[Page(number=1, text="Retention rules. " * 200, source="tika")],
        classification="CONFIDENTIAL", acl=["team-claims"],
    )
    chunks = Chunker(ChunkingSettings()).chunk(doc)
    assert len(chunks) > 1, "need several chunks to prove every one is labelled"
    assert all(c.classification == "CONFIDENTIAL" for c in chunks)
    assert all(c.acl == ["team-claims"] for c in chunks)


def test_a_chunks_labels_are_copied_not_aliased():
    # Mutating the document afterwards must not silently relabel indexed chunks.
    from rag_core.chunking import Chunker
    from rag_core.config import ChunkingSettings

    # Must clear min_chunk_chars (80) or nothing is emitted at all.
    doc = Document(doc_id="d", source_uri="u", filename="f.txt",
                   pages=[Page(number=1, text="Retention rules apply. " * 12, source="tika")],
                   acl=["original"])
    chunks = Chunker(ChunkingSettings()).chunk(doc)
    doc.acl.append("added-later")
    assert chunks[0].acl == ["original"]
