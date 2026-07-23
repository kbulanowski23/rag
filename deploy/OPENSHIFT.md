# Deploying to OpenShift

Everything environment-specific is an environment variable. No hostname, model
name, URL or key is compiled into any image, so the same build is promoted from
dev to prod and only the ConfigMap differs.

## The four values that must be right

Everything else in `00-config.yaml` has a working default. These do not:

| Value | Where | Notes |
|---|---|---|
| `RAG_LLM__BASE_URL` + `RAG_LLM__MODEL` | ConfigMap | Your LiteLLM endpoint. URL includes `/v1`. |
| `RAG_EMBEDDING__*` | ConfigMap | Provider, model, and **dim**. See below. |
| `RAG_OPENSEARCH__HOSTS` | ConfigMap | Your cluster. |
| `RAG_LLM__API_KEY` | Secret | Never in git. |

## LiteLLM

LiteLLM speaks the OpenAI wire format, so it needs no special provider:

```yaml
RAG_LLM__PROVIDER: "openai_compatible"
RAG_LLM__BASE_URL: "https://litellm.apps.example/v1"   # /v1 matters
RAG_LLM__MODEL: "gpt-4o"      # the name LiteLLM routes on, not the vendor's
```

`RAG_LLM__MODEL` must be a name from LiteLLM's `model_list`, which is often not
what the upstream vendor calls the model. Confirm with `GET {BASE_URL}/models`.

If the gateway needs routing headers:

```yaml
RAG_LLM__EXTRA_HEADERS: '{"X-Team":"claims"}'
```

If TLS is intercepted, mount the internal CA and set `RAG_LLM__CA_BUNDLE`
rather than turning `RAG_LLM__VERIFY_SSL` off.

## Embeddings — the one that will bite

The embedding dimension defines the OpenSearch vector field. It is fixed when
the index is created and **cannot be changed afterwards**. Getting it wrong is
the most likely first-deploy failure, so readiness reports it explicitly:

```
RAG_EMBEDDING__DIM=384 but model 'text-embedding-3-large' returned 3072.
Set the dim to 3072, then recreate the index and re-ingest.
```

Two supported shapes:

```yaml
# (a) model inside the image -- no network hop per query, cannot drift
RAG_EMBEDDING__PROVIDER: "local_onnx"
RAG_EMBEDDING__MODEL_PATH: "/models/bge-small-en-v1.5"
RAG_EMBEDDING__DIM: "384"

# (b) served from LiteLLM
RAG_EMBEDDING__PROVIDER: "openai_compatible"
RAG_EMBEDDING__BASE_URL: "https://litellm.apps.example/v1"
RAG_EMBEDDING__MODEL: "text-embedding-3-large"
RAG_EMBEDDING__DIM: "3072"
RAG_EMBEDDING__QUERY_PREFIX: ""     # bge-only; wrong for OpenAI models
```

Common dims: `bge-small` 384 · `bge-base`/`e5-base` 768 · `bge-large` 1024 ·
`text-embedding-3-small` 1536 · `text-embedding-3-large` 3072. Do not trust the
table — verify against the endpoint.

**Changing the embedding model invalidates every vector already indexed.** There
is no migration: recreate the index and re-ingest. Vectors from two models are
not comparable, and the failure is silent — retrieval quietly returns nonsense
rather than erroring.

`RAG_EMBEDDING__QUERY_PREFIX` is not cosmetic. bge and e5 were trained
asymmetrically, and dropping the prefix measurably degrades retrieval. It is
also wrong to send a bge prefix to an OpenAI model — clear it when switching.

## Order of operations, first deploy

```bash
oc new-project rag

# 1. Secret first -- the API pod will not start without it.
oc create secret generic rag-secrets \
  --from-literal=RAG_LLM__API_KEY=... \
  --from-literal=RAG_EMBEDDING__API_KEY=... \
  --from-literal=RAG_OPENSEARCH__USERNAME=... \
  --from-literal=RAG_OPENSEARCH__PASSWORD=...

# 2. Config and workloads.
oc apply -f deploy/openshift/00-config.yaml --selector=app.kubernetes.io/part-of=rag
oc apply -f deploy/openshift/10-api.yaml
oc apply -f deploy/openshift/20-tika-ocr.yaml
oc apply -f deploy/openshift/30-web.yaml

# 3. Create the index. Must happen before the first ingest.
oc apply -f deploy/openshift/40-ingest-job.yaml

# 4. Prove it. Exercises extraction, chunking, embedding, indexing, BM25,
#    kNN, fusion, OCR, the LLM and SSE. Ten stages, all must pass.
oc exec deployment/rag-api -- python ops/smoke_test.py --api-url http://localhost:8000

oc get route rag-web
```

The `--selector` in step 2 is deliberate: `00-config.yaml` also contains a
placeholder Secret, and applying it would overwrite the real credentials from
step 1 with `REPLACE-ME`.

## Changing configuration later

```bash
oc edit configmap rag-config
oc rollout restart deployment/rag-api deployment/rag-web
```

Safe to change live: retrieval settings (`FUSION`, `TOP_K_*`, `FINAL_K`,
`CONTEXT_TOKEN_BUDGET`), LLM settings, timeouts, log level.

Requires re-ingest: `RAG_CHUNKING__*` applies only to newly ingested documents,
so changing it leaves a corpus chunked two different ways.

Requires index recreate + re-ingest: `RAG_EMBEDDING__DIM`, or any change of
embedding model.

Note that `Settings` ignores unknown variables — a misspelled key is silence,
not an error, and the service will start happily on the default. The keys in the
manifest are checked against the settings model by `tests/test_openshift_config.py`.

## Networking

The browser only ever talks to `rag-web`. That pod proxies `/api/v1/*` to the
API over cluster DNS (`web/app/api/v1/[...path]/route.ts`, target `RAG_API_URL`).

Consequences:
- The API has **no Route**. It is reachable inside the namespace only.
- CORS never applies to the UI; `RAG_API__CORS_ORIGINS` stays empty.
- One public hostname, and one image per service across all environments.

The web Route carries `haproxy.router.openshift.io/timeout: 300s`. Without it
the router closes the SSE connection at its 30s default and answers appear to
stop mid-sentence. Keep it >= `RAG_LLM__TIMEOUT_S`.

## Troubleshooting

| Symptom | Cause |
|---|---|
| API pod never ready, `dim=N` in readiness | `RAG_EMBEDDING__DIM` disagrees with the model. |
| Answers stop mid-sentence around 30s | Missing SSE timeout annotation on the web Route. |
| UI loads, every request 404s | `RAG_API_URL` unset or wrong; check the web pod's env. |
| UI loads, requests 502 with "RAG API unreachable" | `RAG_API_URL` points somewhere the web pod cannot reach. |
| Retrieval returns irrelevant results | Embedding model changed without re-indexing. |
| Scanned PDFs return nothing | OCR disabled or unreachable; ingest logs show `ocr_pages=0`. |
| A ConfigMap change had no effect | Key typo — unknown variables are ignored silently. |

`GET /api/v1/config` returns the effective non-secret configuration and is the
fastest way to see what a pod actually loaded. The deploy pipeline prints it
after every rollout, so the job log is a record of what each environment ran.
