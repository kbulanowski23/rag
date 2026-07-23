from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    # Per-request overrides of the configured defaults. Everything is optional;
    # omitting a field means "use the deployment's configuration".
    k: int | None = Field(default=None, ge=1, le=50)
    filters: dict[str, Any] | None = None
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    stream: bool = True


class SourceOut(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    title: str = ""
    filename: str = ""
    source_uri: str = ""
    page_start: int = 0
    page_end: int = 0
    score: float = 0.0
    extraction_source: str = ""
    # Returned so a caller can show what it retrieved and why it was allowed.
    # Filtering on these already worked; without them in the response the labels
    # were invisible to the client, which makes an access decision unauditable.
    classification: str = ""
    acl: list[str] = Field(default_factory=list)
    retrievers: dict[str, int] = Field(default_factory=dict)
    text: str = ""


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    cited: list[int]
    model: str
    timings_ms: dict[str, int]


class SearchRequest(BaseModel):
    """Retrieval without generation. Useful for tuning relevance and for a
    plain search UI that does not need an LLM at all."""

    query: str = Field(min_length=1, max_length=1000)
    k: int | None = Field(default=None, ge=1, le=100)
    filters: dict[str, Any] | None = None
    fusion: Literal["rrf", "vector_only", "bm25_only"] | None = None


class SearchResponse(BaseModel):
    query: str
    hits: list[SourceOut]
    took_ms: int


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks_indexed: int
    ocr_pages: int
    failed: int
    errors: list[str]
    timings_ms: dict[str, int]


class HealthComponent(BaseModel):
    name: str
    ok: bool
    detail: str = ""


class HealthResponse(BaseModel):
    ok: bool
    env: str
    components: list[HealthComponent]


class ConfigResponse(BaseModel):
    """Effective non-secret configuration. Exposed so an operator can confirm
    what a running pod actually resolved, which is the first question asked
    whenever behaviour differs between environments."""

    env: str
    llm_provider: str
    llm_model: str
    llm_base_url: str
    embedding_provider: str
    embedding_dim: int
    index: str
    fusion: str
    final_k: int
    rerank_enabled: bool
    ocr_enabled: bool
