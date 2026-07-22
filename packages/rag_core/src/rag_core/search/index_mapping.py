"""Index mapping and bootstrap.

Deliberately plain: text, a knn_vector, and metadata. No ingest pipeline, no
model id, no neural field. The cluster needs only the `knn` plugin, which is
part of every standard OpenSearch distribution.
"""

from __future__ import annotations

from typing import Any

from rag_core.config import OpenSearchSettings


def build_mapping(s: OpenSearchSettings, dim: int) -> dict[str, Any]:
    return {
        "settings": {
            "index": {
                "number_of_shards": s.num_shards,
                "number_of_replicas": s.num_replicas,
                "knn": True,
                # Raise this only if you actually need recall above ~0.95; it
                # costs query latency linearly.
                "knn.algo_param.ef_search": 100,
            },
            "analysis": {
                "analyzer": {
                    "rag_text": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding", "english_possessive_stemmer", "english_stop", "english_stemmer"],
                    }
                },
                "filter": {
                    "english_stop": {"type": "stop", "stopwords": "_english_"},
                    "english_stemmer": {"type": "stemmer", "language": "english"},
                    "english_possessive_stemmer": {"type": "stemmer", "language": "possessive_english"},
                },
            },
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "text": {
                    "type": "text",
                    "analyzer": "rag_text",
                    # A raw sub-field keeps exact phrase matching available for
                    # identifiers and part numbers that stemming would mangle.
                    "fields": {"raw": {"type": "text", "analyzer": "standard"}},
                },
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": s.knn_space_type,
                        "engine": s.knn_engine,
                        "parameters": {"m": s.hnsw_m, "ef_construction": s.hnsw_ef_construction},
                    },
                },
                "ordinal": {"type": "integer"},
                "page_start": {"type": "integer"},
                "page_end": {"type": "integer"},
                "char_start": {"type": "integer"},
                "char_end": {"type": "integer"},
                "token_count": {"type": "integer"},
                "extraction_source": {"type": "keyword"},
                "filename": {"type": "keyword", "fields": {"text": {"type": "text"}}},
                "title": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 512}}},
                "source_uri": {"type": "keyword"},
                "indexed_at": {"type": "date"},
                # Arbitrary business metadata. `false` means stored and
                # returnable but not indexed -- add explicit fields above for
                # anything you need to filter on.
                "metadata": {"type": "object", "enabled": False},
            },
        },
    }


def build_alias_actions(index: str, alias: str) -> dict[str, Any]:
    return {"actions": [{"add": {"index": index, "alias": alias}}]}
