"""Single source of truth for configuration.

Every knob is an environment variable so that the same image runs unchanged at
home and in the air-gapped environment. Nested sections use a double-underscore
delimiter: RAG_LLM__MODEL, RAG_OPENSEARCH__HOSTS, and so on.

Nothing in this module reads a file at import time except an optional .env for
local development, which is absent in a container.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["openai_compatible", "azure_openai", "anthropic", "echo"]
EmbeddingProviderName = Literal["local_onnx", "openai_compatible"]
FusionStrategy = Literal["rrf", "vector_only", "bm25_only"]
AuthMode = Literal["none", "header", "oidc"]


class LLMSettings(BaseModel):
    provider: LLMProviderName = "openai_compatible"
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5:14b"
    api_key: str = ""
    # Azure only: pinned API version, and the deployment name is taken from `model`.
    api_version: str = "2024-10-21"

    temperature: float = 0.1
    max_tokens: int = 1500
    top_p: float = 1.0
    timeout_s: float = 120.0

    verify_ssl: bool = True
    ca_bundle: str | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)

    # Reasoning models emit their chain of thought inline before the answer.
    # It is longer than the answer, contradicts itself as it reasons, and leaks
    # the prompt's structure, so it is stripped before the user sees it. Turn
    # this off to see the raw trace when debugging a bad answer. The tags are
    # settable because the convention is not universal.
    strip_reasoning: bool = True
    reasoning_open_tag: str = "<think>"
    reasoning_close_tag: str = "</think>"

    # Extra JSON merged into the request body, for parameters the OpenAI wire
    # format does not define. Stripping a reasoning trace only hides it -- the
    # tokens are still generated and still cost the user the wait. Not
    # generating them is far better, and that switch is vendor-specific:
    #
    #   vLLM / qwen3:  {"chat_template_kwargs": {"enable_thinking": false}}
    #   others:        {"reasoning_effort": "low"}
    #
    # Merged last, so it can override the standard fields when a gateway needs
    # something non-standard. Whether it reaches the model depends on the
    # gateway passing unknown keys through; LiteLLM generally does.
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @field_validator("extra_headers", "extra_body", mode="before")
    @classmethod
    def _parse_json_obj(cls, v: Any) -> Any:
        # Env vars arrive as strings; accept JSON so a ConfigMap can carry them.
        if isinstance(v, str):
            v = v.strip()
            return json.loads(v) if v else {}
        return v

    @field_validator("base_url")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        return v.rstrip("/")


class EmbeddingSettings(BaseModel):
    provider: EmbeddingProviderName = "local_onnx"
    # For local_onnx this is a directory holding model.onnx + tokenizer.json.
    model_path: str = "./deploy/models/bge-small-en-v1.5"
    # For openai_compatible:
    base_url: str = ""
    model: str = ""
    api_key: str = ""

    dim: int = 384
    max_tokens: int = 512
    batch_size: int = 32
    normalize: bool = True
    # Asymmetric models (bge, e5) need different prefixes per side. Empty is fine.
    query_prefix: str = ""
    passage_prefix: str = ""
    timeout_s: float = 60.0
    # onnxruntime intra-op threads; 0 lets ORT decide. Pin this in OpenShift so a
    # CPU-limited pod does not spawn a thread per host core.
    num_threads: int = 0


class OpenSearchSettings(BaseModel):
    hosts: str = "http://localhost:9200"
    index: str = "rag-chunks"
    username: str = ""
    password: str = ""
    verify_certs: bool = True
    ca_certs: str | None = None
    timeout_s: float = 60.0
    bulk_size: int = 200

    num_shards: int = 1
    num_replicas: int = 0
    hnsw_m: int = 16
    hnsw_ef_construction: int = 128
    # `lucene` ships with OpenSearch and needs no extra native library, which is
    # the safest choice for a cluster we do not control. `faiss` and `nmslib` are
    # available if the target cluster has them.
    knn_engine: Literal["lucene", "faiss", "nmslib"] = "lucene"
    knn_space_type: Literal["cosinesimil", "l2", "innerproduct"] = "cosinesimil"

    @property
    def host_list(self) -> list[str]:
        return [h.strip() for h in self.hosts.split(",") if h.strip()]


class ChunkingSettings(BaseModel):
    max_tokens: int = 450
    overlap_tokens: int = 64
    min_chunk_chars: int = 80
    respect_paragraphs: bool = True


class RetrievalSettings(BaseModel):
    fusion: FusionStrategy = "rrf"
    top_k_bm25: int = 50
    top_k_vector: int = 50
    final_k: int = 8
    rrf_k: int = 60
    min_score: float = 0.0
    context_token_budget: int = 6000
    rerank_enabled: bool = False
    rerank_model_path: str = "./deploy/models/bge-reranker-base"
    rerank_candidates: int = 30


class TikaSettings(BaseModel):
    url: str = "http://localhost:9998"
    timeout_s: float = 300.0
    # The tika-server image bundles Tesseract and will quietly OCR any image it
    # is handed. That defeats the routing seam: pages come back above
    # min_chars_per_page, labelled "tika", never reach EasyOCR, and the text is
    # worse. Extraction stays extraction; OCR is a decision the pipeline makes.
    skip_builtin_ocr: bool = True

    @field_validator("url")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        return v.rstrip("/")


class OCRSettings(BaseModel):
    enabled: bool = True
    url: str = "http://localhost:9999"
    timeout_s: float = 600.0
    languages: str = "en"
    # Below this many characters of extracted text per page, assume the page is a
    # scan and send the rendered page image to EasyOCR.
    min_chars_per_page: int = 120

    @field_validator("url")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @property
    def language_list(self) -> list[str]:
        return [x.strip() for x in self.languages.split(",") if x.strip()]


class APISettings(BaseModel):
    cors_origins: str = "http://localhost:3000"
    max_upload_mb: int = 100
    auth_mode: AuthMode = "none"
    # Override the built-in system prompt without rebuilding the image.
    system_prompt_path: str | None = None

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "local"
    log_level: str = "INFO"

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    tika: TikaSettings = Field(default_factory=TikaSettings)
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    api: APISettings = Field(default_factory=APISettings)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide singleton. Import this, never construct Settings directly."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Test hook: force the next get_settings() to re-read the environment."""
    global _settings
    _settings = None
