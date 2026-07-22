"""Configuration is the mechanism the whole air-gap story rests on, so the
nested env-var contract is tested explicitly."""

from __future__ import annotations

import pytest

from rag_core.config import Settings, reset_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("RAG_"):
            monkeypatch.delenv(key, raising=False)
    reset_settings()
    yield
    reset_settings()


def test_defaults_are_usable_without_any_env():
    s = Settings(_env_file=None)
    assert s.llm.provider == "openai_compatible"
    assert s.embedding.dim == 384
    assert s.retrieval.fusion == "rrf"


def test_nested_env_vars_override(monkeypatch):
    monkeypatch.setenv("RAG_LLM__PROVIDER", "azure_openai")
    monkeypatch.setenv("RAG_LLM__MODEL", "gpt-5.2-deployment")
    monkeypatch.setenv("RAG_RETRIEVAL__FINAL_K", "12")
    s = Settings(_env_file=None)
    assert s.llm.provider == "azure_openai"
    assert s.llm.model == "gpt-5.2-deployment"
    assert s.retrieval.final_k == 12


def test_extra_headers_parse_from_json(monkeypatch):
    monkeypatch.setenv("RAG_LLM__EXTRA_HEADERS", '{"X-Tenant":"search"}')
    assert Settings(_env_file=None).llm.extra_headers == {"X-Tenant": "search"}


def test_base_url_trailing_slash_is_stripped(monkeypatch):
    # Otherwise every request URL ends up with a double slash.
    monkeypatch.setenv("RAG_LLM__BASE_URL", "https://llm.internal.corp/v1/")
    assert Settings(_env_file=None).llm.base_url == "https://llm.internal.corp/v1"


def test_comma_separated_lists(monkeypatch):
    monkeypatch.setenv("RAG_OPENSEARCH__HOSTS", "https://a:9200, https://b:9200")
    monkeypatch.setenv("RAG_API__CORS_ORIGINS", "https://x.example,https://y.example")
    s = Settings(_env_file=None)
    assert s.opensearch.host_list == ["https://a:9200", "https://b:9200"]
    assert s.api.cors_origin_list == ["https://x.example", "https://y.example"]


def test_invalid_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("RAG_LLM__PROVIDER", "not_a_provider")
    with pytest.raises(Exception):
        Settings(_env_file=None)
