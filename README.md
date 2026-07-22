# Enterprise RAG Search

Air-gap-capable document search and chat. Documents are extracted (Tika/EasyOCR),
chunked and embedded in Python, and indexed into OpenSearch as plain vectors.
A FastAPI service does hybrid retrieval and answer synthesis against a pluggable LLM.
Next.js provides the chat UI.

## Design rules

1. **No heavyweight orchestration frameworks.** No LangChain / LlamaIndex / Haystack.
   Retrieval glue is written here, in the open, ~400 lines.
2. **Nothing reaches the public internet at runtime.** Models are baked into images.
   No telemetry, no CDN assets, no model-hub downloads.
3. **OpenSearch stays dumb.** It stores text + vectors and runs BM25/kNN. No
   ml-commons, no neural pipelines, no model deployment in the cluster.
   Only the `knn` plugin is required.
4. **Everything is configured, not coded.** One settings layer, env-var driven,
   backed by an OpenShift ConfigMap/Secret. Swapping the LLM is a config change.

## Layout

```
packages/rag_core/     shared library: config, llm, embeddings, opensearch, chunking
services/api/          FastAPI: /chat, /search, /ingest  (no torch)
services/worker/       ingest pipeline: tika -> ocr -> chunk -> embed -> index
services/ocr/          EasyOCR HTTP wrapper (isolated: this is the only torch image)
web/                   Next.js chat frontend
deploy/openshift/      manifests
deploy/models/         baked model artifacts (see deploy/models/README.md)
ops/                   airgap transfer, wheelhouse, index bootstrap
```

## Data flow

**Ingest:** file -> Tika (text + metadata + page structure) -> if text yield is below
threshold, route pages to EasyOCR -> token-aware chunking with overlap -> batch embed
-> `_bulk` into OpenSearch.

**Query:** Next.js -> `POST /chat` -> embed query -> parallel BM25 + kNN -> RRF fusion
-> optional cross-encoder rerank -> context assembly under a token budget -> LLM ->
SSE stream of tokens followed by a citation payload.

## Local development

Prerequisites: Docker Desktop, Python 3.12, Node 20+.

```
cp .env.example .env
python ops/fetch_models.py          # one-time, needs internet; run at home only
docker compose up -d opensearch tika ocr
python -m venv .venv && .venv\Scripts\activate
pip install -e packages/rag_core -r services/api/requirements.txt
python -m ops.bootstrap_index
uvicorn app.main:app --app-dir services/api --reload --port 8000
cd web && npm ci && npm run dev
```

## Air-gapped transfer

See `ops/AIRGAP.md`. Summary: build all images at home, `docker save` them, carry the
tarball plus the wheelhouse and the `npm` cache, load into the internal registry.
Model artifacts travel inside the images — nothing is fetched at runtime.
