# Moving this to the classified environment

Everything here was verified by building and running it, not reasoned about.
Where a number appears, it was measured.

## TL;DR

**No code changes.** Every environment-specific value is an environment
variable. What you change is a ConfigMap, a Secret, and where images are pulled
from.

| | |
|---|---|
| Python packages to mirror | **31** (slim build) or 48 (with local embeddings) |
| npm packages to mirror | **58**, all from `registry.npmjs.org`, none from private or git sources |
| Base images | `ubi8/python-312`, `ubi8/nodejs-20` |
| Third-party images | `apache/tika:3.0.0.0-full`. OpenSearch you already run. |
| HuggingFace | **removable entirely** — see below |
| Frameworks | FastAPI, Pydantic, httpx, opensearch-py, Next.js. No LangChain, no LlamaIndex, no vector-DB SDK. |

The one genuinely hard artifact is the OCR image. Everything else is ordinary.

---

## 1. Removing HuggingFace

You said no HuggingFace. That is achievable completely, and it makes the image
smaller rather than requiring a workaround.

`huggingface_hub` and `hf-xet` are *transitive* dependencies of `tokenizers`,
which is a dependency of the **local ONNX embedder**. Nothing in this codebase
ever calls the Hub — the tokenizer loads from a local file
(`Tokenizer.from_file`, `local_onnx.py:55`) — but in a classified review the
package's presence is a conversation you do not need to have.

Because you serve embeddings from LiteLLM, that whole branch is dead code. The
provider factory imports lazily (`embeddings/__init__.py`), so with
`RAG_EMBEDDING__PROVIDER=openai_compatible` none of it is even loaded:

```
provider: RemoteOpenAIEmbedder
onnxruntime imported: False
tokenizers imported : False
huggingface_hub     : False
```

So build the API and worker with the slim requirements:

```bash
docker build -f services/api/Dockerfile \
  --build-arg REQUIREMENTS=requirements-served.txt \
  --build-arg WITH_MODELS=false \
  -t rag-api:0.1.0 .
```

| | full | slim |
|---|---|---|
| Python packages | 48 | **31** |
| `huggingface_hub`, `hf-xet`, `onnxruntime`, `tokenizers`, `numpy` | present | **absent** |
| Model weights from huggingface.co | 128 MB | **none** |
| Image size | 581 MB | **449 MB** |

**Verified on OpenShift**, not assumed: the slim image passes readiness on all
four components, ingests a scanned PNG through OCR, and answers a question
grounded and cited in 1.3 s.

This also means `ops/fetch_models.py` — the only script that touches
`huggingface.co` — never runs for the API or worker. It is still needed for OCR
(see §4).

### The one trade-off

Without `tokenizers`, chunking falls back to `ApproxTokenizer`, which estimates
tokens as `chars / 4` (`chunking.py:84-97`). Chunk boundaries become slightly
less precise.

This matters only at the edges: `RAG_CHUNKING__MAX_TOKENS=450` is now an
*estimate*, and dense text (tables, code, non-English) has more tokens per
character than prose. If your embedding model's limit is 512, leave headroom —
**set `RAG_CHUNKING__MAX_TOKENS` to 380–400** rather than 450. An overlong chunk
is silently truncated by the embedding service, which loses text from the *end*
of the chunk with no error.

---

## 2. What to mirror

### Python — 31 packages (slim)

All are mainstream PyPI with no compiled exotica. The full closure:

```
annotated-types  anyio  certifi  charset-normalizer  click  Events  fastapi
h11  httpcore  httptools  httpx  idna  opensearch-py  pydantic  pydantic-core
pydantic-settings  python-dateutil  python-dotenv  python-multipart  PyYAML
requests  six  sniffio  starlette  typing-extensions  urllib3  uvicorn  uvloop
watchfiles  websockets  (+ rag-core, built from this repo)
```

`uvloop`, `httptools` and `watchfiles` are `uvicorn[standard]` extras and ship
manylinux wheels. Nothing needs a compiler.

Mirror with your normal tooling, e.g.:

```bash
pip download -r services/api/requirements-served.txt -d wheelhouse/
```

Then set `PIP_INDEX_URL` to your internal index at build time.

### npm — 58 packages

```
dependencies:    next@16.2.11, react@19.2.0, react-dom@19.2.0
devDependencies: typescript@5.7.2, @types/{node,react,react-dom}
```

Three runtime dependencies. No component library, no CSS framework, no icon
pack, no HTTP wrapper — that was deliberate (see the `//` note in
`package.json`). All 58 resolve from `registry.npmjs.org`; **none** come from
git URLs or private registries, which is usually the thing that breaks a mirror.

37 of the 58 are platform-specific optional packages (`@next/swc-*`,
`@img/sharp-*`). Two warnings, both already handled in the repo:

- **Do not run `npm ci --omit=optional`.** It drops `@next/swc-*`, and the build
  then tries to *download* the SWC binary at build time — fatal inside the gap.
- `sharp` carries libvips CVEs and would fail an image scan. It is not skipped
  at install; it is excluded from the traced standalone output via
  `outputFileTracingExcludes` in `next.config.mjs`, so it never reaches the
  runtime image. Safe only because `images.unoptimized` is set.

Set `npm config set registry <your mirror>` before `npm ci`.

### Base images

```
registry.access.redhat.com/ubi8/python-312   (api, worker, ocr)
registry.access.redhat.com/ubi8/nodejs-20    (web)
```

Both Dockerfiles parameterise this, so point them at your mirror without editing
anything:

```bash
--build-arg BASE_IMAGE=nexus.internal/ubi8/python-312:latest
--build-arg NODE_IMAGE=nexus.internal/ubi8/nodejs-20:latest
```

### Third-party images

| Image | Notes |
|---|---|
| `apache/tika:3.0.0.0-full` | Needs the `-full` tag; slim cannot read some office formats |
| OpenSearch | You already run it. Nothing to transfer. |

---

## 3. What runs at runtime: nothing external

The design rule is that no pod fetches anything at runtime. Verified by scanning
all shipped source for external URLs — the **only** hit is the default
`base_url` in `llm/anthropic.py`, used only if you select that provider.

Every outbound connection a running pod makes is to an address you configure:

| To | Setting |
|---|---|
| LiteLLM (chat) | `RAG_LLM__BASE_URL` |
| LiteLLM (embeddings) | `RAG_EMBEDDING__BASE_URL` |
| OpenSearch | `RAG_OPENSEARCH__HOSTS` |
| Tika | `RAG_TIKA__URL` (in-cluster) |
| OCR | `RAG_OCR__URL` (in-cluster) |

Belt and braces, if your scanners care: set `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` on the pods. With the slim build these packages are not
installed at all, so it is decoration rather than defence.

---

## 4. The OCR image — the one hard case

`rag-ocr` is the only build with real internet dependencies:

1. **torch CPU wheels** from `https://download.pytorch.org/whl/cpu`, not PyPI
   (`services/ocr/requirements.txt`). Your mirror must carry the CPU build
   specifically — the PyPI default is the CUDA build, ~2.5 GB larger and useless
   without a GPU.
2. **EasyOCR model weights**, downloaded during the build
   (`services/ocr/Dockerfile:24`) so the container never fetches at runtime.

Three options, in order of preference:

**(a) Build outside the gap and transfer the image.** This is what
`ops/AIRGAP.md` describes and what the design assumes. ~3 GB via `docker save`.
Nothing to mirror.

**(b) Mirror torch CPU + pre-stage the EasyOCR weights.** Needed if policy says
images must be built inside. The weights are ordinary files; stage them into
`/models/easyocr` and delete the download step from the Dockerfile.

**(c) Turn OCR off** — `RAG_OCR__ENABLED=false`, and drop the Deployment.
Scanned documents then index as empty and become invisible to search. Given how
much of a classified corpus is typically scanned, I would not choose this.

OCR is otherwise proven: it runs on OpenShift under the restricted SCC, routes
correctly, and tags chunks `extraction_source=ocr`.

---

## 5. Configuration swaps

Everything below is a ConfigMap value. No rebuild.

| Setting | Sandbox | Yours |
|---|---|---|
| `RAG_LLM__BASE_URL` | in-cluster vLLM | your LiteLLM, **including `/v1`** |
| `RAG_LLM__MODEL` | `isvc-qwen3-8b-fp8` | the name from LiteLLM's `model_list` |
| `RAG_LLM__API_KEY` | SA token | your key, **in the Secret** |
| `RAG_EMBEDDING__PROVIDER` | `openai_compatible` | same |
| `RAG_EMBEDDING__BASE_URL` | stub | your embeddings endpoint |
| `RAG_EMBEDDING__MODEL` | `bge-small-en-v1.5` | your model |
| `RAG_EMBEDDING__DIM` | 384 | **must match — see below** |
| `RAG_EMBEDDING__QUERY_PREFIX` | bge prefix | **clear it** unless bge/e5 |
| `RAG_OPENSEARCH__HOSTS` | in-cluster | your cluster |
| `RAG_API__AUTH_MODE` | `none` | `oidc` (Ping) |

### The dimension

`RAG_EMBEDDING__DIM` defines the OpenSearch vector field. It is **fixed at index
creation and cannot be changed**. Get it from the endpoint, do not trust a table:

```bash
curl -s $BASE_URL/embeddings -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"<your-model>","input":["test"]}' | jq '.data[0].embedding | length'
```

Readiness fails loudly and specifically if it disagrees. I tested index creation
at 384 / 768 / 1024 / 1536 / 3072 / 4096 / 8192 against OpenSearch 2.18 with the
`lucene` engine — all accepted, so whatever your model returns will fit.

Changing the embedding model later invalidates every vector already indexed.
There is no migration: recreate the index and re-ingest. The failure is silent —
retrieval quietly returns nonsense rather than erroring.

### The query prefix

`RAG_EMBEDDING__QUERY_PREFIX` is currently bge's asymmetric prefix. Sending it to
a model not trained with one measurably degrades retrieval and produces no error.
Clear it unless your model is bge or e5.

### If your model reasons

`RAG_LLM__EXTRA_BODY` passes vendor JSON through to the gateway. On qwen3,
`{"chat_template_kwargs":{"enable_thinking":false}}` took time-to-first-token
from 18.3 s to 0.56 s. Other gateways spell it `{"reasoning_effort":"low"}`. If
your model does not reason, leave it empty — the default is a no-op.
`RAG_LLM__STRIP_REASONING` (default on) removes any trace that arrives anyway.

---

## 6. What you may not need

**The web UI.** You have your own. The API is a plain FastAPI service — point
your UI at `/api/v1/chat/stream` (SSE), `/api/v1/search`, `/api/v1/ingest`.
Dropping `rag-web` removes Next.js, all 58 npm packages, and the `ubi8/nodejs-20`
base from the transfer entirely.

If you *do* drop it, note that the web pod is currently what proxies the browser
to the API. Without it the API needs its own Route, and
`RAG_API__CORS_ORIGINS` must list your UI's origin.

**OpenSearch and Dashboards.** Already deployed. Only `RAG_OPENSEARCH__HOSTS`,
credentials and `RAG_OPENSEARCH__CA_CERTS` change. Confirm the `lucene` k-NN
engine is available (it ships with OpenSearch; `faiss`/`nmslib` need a plugin).

---

## 7. Checklist

- [ ] Mirror 31 Python packages (`pip download -r requirements-served.txt`)
- [ ] Mirror `ubi8/python-312`, and `ubi8/nodejs-20` if keeping the web UI
- [ ] Mirror `apache/tika:3.0.0.0-full`
- [ ] Decide OCR: transfer the image (a), mirror torch CPU + weights (b), or off (c)
- [ ] Build API/worker with `REQUIREMENTS=requirements-served.txt WITH_MODELS=false`
- [ ] Confirm the embedding dimension from the live endpoint
- [ ] Clear `RAG_EMBEDDING__QUERY_PREFIX` unless bge/e5
- [ ] Lower `RAG_CHUNKING__MAX_TOKENS` to ~380 for the approximate tokenizer
- [ ] Create `rag-secrets` out of band — never apply the placeholder in `00-config.yaml`
- [ ] Create the index (`python -m app.main bootstrap`) before the first ingest
- [ ] Run `ops/smoke_test.py --api-url ...` — ten stages, all must pass

That last one is the real acceptance test. It exercises extraction, chunking,
embedding, indexing, BM25, kNN, RRF fusion, OCR, the LLM and SSE streaming, and
it is the fastest way to discover a wrong dimension or a key without access to
the model.

---

## 8. Is anything here risky?

Honest assessment of what could still bite:

| Risk | Likelihood | Notes |
|---|---|---|
| Embedding dimension mismatch | **High** | Readiness catches it; recreate the index |
| torch CPU wheels not mirrored | **High** | Only if building OCR inside the gap |
| TLS interception to LiteLLM | Medium | Mount the CA, set `RAG_LLM__CA_BUNDLE` — do not disable verification |
| Egress policy blocks the pod → LiteLLM | Medium | NetworkPolicy / egress firewall, not an app concern |
| `npm ci --omit=optional` used by a shared CI template | Medium | Breaks the build; tell whoever owns the template |
| Third-party images failing under the restricted SCC | Medium | Hit this twice with OpenSearch images; fix is `chgrp -R 0 && chmod -R g=u` |
| Docker-strategy builds disabled by policy | Low | Only affects in-cluster builds |

Two things this repo has *not* been tested against, because a trial cluster
cannot simulate them: **TLS interception** and **egress policy**. Both are
configuration rather than code, but both are where I would expect the first day
at work to go.
