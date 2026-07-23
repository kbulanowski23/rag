from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Response
from starlette.concurrency import run_in_threadpool

from app.deps import Container, get_container
from app.schemas import ConfigResponse, HealthComponent, HealthResponse

log = logging.getLogger(__name__)
router = APIRouter(tags=["ops"])


@router.get("/health/live")
def live() -> dict:
    """Liveness: is the process running. Must not touch dependencies -- if it
    did, an OpenSearch blip would make OpenShift restart every API pod."""
    return {"status": "ok"}


@router.get("/health/ready", response_model=HealthResponse)
async def ready(response: Response, c: Container = Depends(get_container)) -> HealthResponse:
    """Readiness: can this pod actually serve a request."""
    components: list[HealthComponent] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            components.append(HealthComponent(name=name, ok=ok, detail=detail))
        except Exception as e:
            components.append(HealthComponent(name=name, ok=False, detail=str(e)[:200]))

    def opensearch_check():
        info = c.client.info()
        exists = c.admin.exists()
        return exists, f"v{info['version']['number']}" + ("" if exists else "; index missing")

    def embedder_check():
        v = c.embedder.embed_query("readiness probe")
        want = c.settings.embedding.dim
        if len(v) == want:
            return True, f"dim={want}"
        # The single most likely misconfiguration when pointing at a new
        # embedding endpoint, and the least obvious from a bare dimension
        # number: the index mapping is built from RAG_EMBEDDING__DIM and is
        # fixed at creation, so this cannot be fixed by a rollout alone.
        return False, (
            f"RAG_EMBEDDING__DIM={want} but model {c.settings.embedding.model or 'local'!r} "
            f"returned {len(v)}. Set the dim to {len(v)}, then recreate the index "
            f"and re-ingest -- existing vectors are unusable at a different dim."
        )

    def tika_check():
        ok = c.tika.health()
        return ok, "" if ok else "unreachable"

    await run_in_threadpool(check, "opensearch", opensearch_check)
    await run_in_threadpool(check, "embedder", embedder_check)
    await run_in_threadpool(check, "tika", tika_check)

    if c.ocr is not None:
        ocr_ok = await run_in_threadpool(c.ocr.health)
        # OCR being down degrades ingest quality but does not stop search, so it
        # is reported without failing readiness.
        components.append(
            HealthComponent(name="ocr", ok=True, detail="" if ocr_ok else "unreachable (degraded)")
        )

    # The LLM is deliberately not probed here: it is often rate-limited or slow
    # to cold-start, and search must stay available when generation is not.
    ok = all(comp.ok for comp in components)
    if not ok:
        response.status_code = 503
    return HealthResponse(ok=ok, env=c.settings.env, components=components)


@router.get("/health/llm")
async def llm_health(c: Container = Depends(get_container)) -> dict:
    ok = await c.llm.health()
    return {
        "ok": ok,
        "provider": c.settings.llm.provider,
        "model": c.settings.llm.model,
        "base_url": c.settings.llm.base_url,
    }


@router.get("/config", response_model=ConfigResponse)
def effective_config(c: Container = Depends(get_container)) -> ConfigResponse:
    """Non-secret effective configuration. Never add api_key or password here."""
    s = c.settings
    return ConfigResponse(
        env=s.env,
        llm_provider=s.llm.provider,
        llm_model=s.llm.model,
        llm_base_url=s.llm.base_url,
        embedding_provider=s.embedding.provider,
        embedding_dim=s.embedding.dim,
        index=s.opensearch.index,
        fusion=s.retrieval.fusion,
        final_k=s.retrieval.final_k,
        rerank_enabled=s.retrieval.rerank_enabled,
        ocr_enabled=s.ocr.enabled,
    )
