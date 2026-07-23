/**
 * Server-side proxy from the web origin to the RAG API.
 *
 * Why this exists rather than a `rewrites()` entry in next.config.mjs: rewrites
 * are evaluated during `next build` and frozen into routes-manifest.json, so the
 * API URL would be whatever the CI runner happened to have and no ConfigMap
 * change could move it. A route handler runs per request, so RAG_API_URL is read
 * from the live environment and `oc set env deployment/rag-web RAG_API_URL=...`
 * takes effect on the next rollout.
 *
 * Two consequences worth knowing:
 *   - The browser only ever talks to the web origin, so the API needs no public
 *     Route and no CORS entry for the UI hostname.
 *   - One image runs in every environment. Nothing about the target is compiled
 *     into the bundle.
 *
 * If you would rather have the browser call the API directly, set
 * NEXT_PUBLIC_API_BASE_URL at build time (see lib/api.ts); absolute URLs bypass
 * this handler entirely, and then the API does need a Route and CORS.
 */

import type { NextRequest } from "next/server";

// Streaming a proxied response is incompatible with any attempt to cache or
// statically analyse it, and SSE must not be buffered.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const fetchCache = "force-no-store";

function target(): string {
  return (process.env.RAG_API_URL || "http://localhost:8000").replace(/\/$/, "");
}

// Hop-by-hop headers describe a single connection and must not be forwarded to
// the next one. `host` in particular would make the API see the browser's
// hostname and mis-route.
const STRIP_REQUEST = ["host", "connection", "keep-alive", "transfer-encoding", "upgrade"];
// Re-encoding is undici's business; passing these through describes a body we
// are no longer sending byte-for-byte.
const STRIP_RESPONSE = ["content-encoding", "content-length", "transfer-encoding", "connection"];

async function proxy(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const url = `${target()}/api/v1/${path.map(encodeURIComponent).join("/")}${req.nextUrl.search}`;

  const headers = new Headers(req.headers);
  for (const h of STRIP_REQUEST) headers.delete(h);

  const hasBody = req.method !== "GET" && req.method !== "HEAD";

  let upstream: Response;
  try {
    upstream = await fetch(url, {
      method: req.method,
      headers,
      // Streamed straight through, so a large upload is never buffered in the
      // web pod's memory. `duplex: "half"` is required by undici whenever the
      // body is a stream rather than a buffer.
      body: hasBody ? req.body : undefined,
      ...(hasBody ? { duplex: "half" } : {}),
      redirect: "manual",
      cache: "no-store",
    } as RequestInit);
  } catch (e) {
    // A connection failure here is nearly always a misconfigured RAG_API_URL,
    // so say so explicitly instead of surfacing a bare 500 from the runtime.
    const detail = e instanceof Error ? e.message : String(e);
    return Response.json(
      { detail: `RAG API unreachable at ${target()}: ${detail}` },
      { status: 502 },
    );
  }

  const out = new Headers(upstream.headers);
  for (const h of STRIP_RESPONSE) out.delete(h);

  // upstream.body is passed through unread, which is what keeps SSE streaming
  // token-by-token instead of arriving as one blob when generation finishes.
  return new Response(upstream.body, { status: upstream.status, headers: out });
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
export const HEAD = proxy;
