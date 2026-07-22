from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.deps import build_container
from app.routes import chat, health, ingest, search
from rag_core.config import get_settings
from rag_core.logging_setup import configure_logging

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging("rag-api", settings.log_level, json_output=settings.env != "local")
    log.info(
        "starting rag-api",
        extra={
            "env": settings.env,
            "llm_provider": settings.llm.provider,
            "llm_model": settings.llm.model,
            "embedding_provider": settings.embedding.provider,
            "index": settings.opensearch.index,
        },
    )
    # Building the container loads the ONNX session; doing it here rather than
    # lazily means a misconfigured model fails the pod at startup instead of on
    # the first user request.
    app.state.container = build_container()
    try:
        yield
    finally:
        await app.state.container.aclose()
        log.info("rag-api stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Enterprise RAG Search API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Auth is a deliberate seam: today it is a no-op, and Ping/OIDC token
    # validation drops in here without touching any route.
    if settings.api.auth_mode != "none":
        from app.auth import AuthMiddleware

        app.add_middleware(AuthMiddleware, settings=settings)

    api_prefix = "/api/v1"
    app.include_router(health.router, prefix=api_prefix)
    app.include_router(chat.router, prefix=api_prefix)
    app.include_router(search.router, prefix=api_prefix)
    app.include_router(ingest.router, prefix=api_prefix)
    return app


app = create_app()
