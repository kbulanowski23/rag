"""Authentication seam.

Not wired for the MVP -- RAG_API__AUTH_MODE defaults to `none`. It exists so that
adding Ping later is a middleware swap rather than an architectural change, and
so nobody is tempted to sprinkle auth checks through the route handlers.

Two modes are sketched:
  header  -- trust an identity header injected by an upstream proxy (the usual
             OpenShift pattern when a sidecar or the router terminates SSO)
  oidc    -- validate a bearer JWT against a JWKS endpoint

`oidc` intentionally raises rather than half-working: a security control that
silently passes everything is worse than one that is absent.
"""

from __future__ import annotations

import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from rag_core.config import Settings

log = logging.getLogger(__name__)

PUBLIC_PATHS = ("/api/v1/health/live", "/api/v1/health/ready", "/docs", "/openapi.json")


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings) -> None:
        super().__init__(app)
        self.settings = settings
        if settings.api.auth_mode == "oidc":
            raise NotImplementedError(
                "auth_mode=oidc is not implemented. Wire JWKS validation here "
                "before enabling it; do not ship a permissive stub."
            )

    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        if self.settings.api.auth_mode == "header":
            user = request.headers.get("x-forwarded-user") or request.headers.get("x-remote-user")
            if not user:
                return JSONResponse({"detail": "unauthenticated"}, status_code=401)
            # Downstream code can read request.state.user; retrieval filters
            # based on entitlements would be applied from here.
            request.state.user = user

        return await call_next(request)
