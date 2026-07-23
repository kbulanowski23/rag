"""The ConfigMap is the deployment's control surface, and Settings is declared
with extra="ignore" -- a misspelled key is not an error, it is silence. The
service starts, reports healthy, and quietly runs on the default value.

That is the worst possible failure mode for a setting like RAG_EMBEDDING__DIM,
so every key shipped in the manifest is checked here against the model that has
to consume it. This runs in CI with no cluster and no network.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from rag_core.config import Settings

MANIFEST = pathlib.Path(__file__).resolve().parents[1] / "deploy/openshift/00-config.yaml"

# Consumed by the Node server in the web pod (see web/app/api/v1/[...path]/route.ts),
# not by rag_core. It is the proxy target, and it has no Settings field by design.
WEB_ONLY_KEYS = {"RAG_API_URL"}


def documents() -> list[dict]:
    return [d for d in yaml.safe_load_all(MANIFEST.read_text(encoding="utf-8")) if d]


def by_kind(kind: str) -> dict:
    return next(d for d in documents() if d.get("kind") == kind)


def binds_to_a_field(key: str) -> bool:
    """Mirror pydantic-settings' env resolution: RAG_<SECTION>__<FIELD>."""
    if not key.startswith("RAG_"):
        return False
    rest = key[len("RAG_"):].lower()
    if "__" not in rest:
        return rest in Settings.model_fields
    section, field = rest.split("__", 1)
    declared = Settings.model_fields.get(section)
    if declared is None:
        return False
    return field in declared.annotation.model_fields


def config_keys() -> list[str]:
    keys = list(by_kind("ConfigMap")["data"])
    keys += list(by_kind("Secret")["stringData"])
    return [k for k in keys if k not in WEB_ONLY_KEYS]


@pytest.mark.parametrize("key", config_keys())
def test_every_configmap_key_binds_to_a_settings_field(key: str):
    assert binds_to_a_field(key), (
        f"{key} matches no field in Settings. pydantic-settings ignores unknown "
        f"variables, so this would be silently dropped in the cluster."
    )


def test_the_web_proxy_target_is_present():
    # The UI is blank without it: every /api/v1 call would go nowhere.
    assert "RAG_API_URL" in by_kind("ConfigMap")["data"]


def test_secrets_are_not_duplicated_in_the_configmap():
    # A credential in the ConfigMap is readable by anyone with get on configmaps
    # and would also be committed to git.
    secret_keys = set(by_kind("Secret")["stringData"])
    assert not (secret_keys & set(by_kind("ConfigMap")["data"]))


def test_embedding_dim_is_set_explicitly():
    # Inheriting the 384 default while pointing at a 1024-dim model builds an
    # index that can never accept the vectors it will be given.
    assert "RAG_EMBEDDING__DIM" in by_kind("ConfigMap")["data"]
