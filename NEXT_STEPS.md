# Start here after the reboot

Written 2026-07-22. Delete this file once `ops/smoke_test.py` passes.

Goal: every stage proven — extraction, **OCR**, chunking, embedding, indexing,
BM25, kNN, fusion, filters, LLM grounding, citations, REST endpoints, SSE
streaming — then the UI answering questions about your own documents.

---

## What you just did

Ran `wsl --install` as Administrator and rebooted. That was the fix for Docker
Desktop's *"Virtualization support not detected"* error, which was misleading:
your BIOS was already correct (`VirtualizationFirmwareEnabled: True`) and WSL2
was simply missing. **Do not change any BIOS settings.**

## Already done — do not redo

- [x] Python 3.12 + `.venv` with onnxruntime 1.20.1
- [x] Embedding model in `deploy/models/bge-small-en-v1.5` (133 MB)
- [x] Embedding + chunking proven (semantic ranking verified)
- [x] OCR test fixture generated in `deploy/fixtures/` (a page with no text layer)
- [x] 31 unit tests passing; frontend builds
- [x] Ollama 0.32.1 installed — **no model pulled yet**

---

# Step 0 — open THREE terminals

The two long jobs (9 GB model pull, ~20 min OCR image build) run in parallel
while you do everything else. Doing them in sequence wastes half an hour.

In **each** git-bash terminal:

```bash
cd /c/Users/Konrad/Desktop/Rag
export PATH="$PATH:$LOCALAPPDATA/Programs/Ollama:$LOCALAPPDATA/Programs/DockerDesktop/resources/bin"
```

Then confirm the reboot worked (any terminal):

```bash
wsl --status            # must NOT say "not installed"
docker info | head -3   # must print server info, not an error
```

If `docker info` errors: open the **Docker Desktop** app and wait for the whale
icon to stop animating. The engine takes ~60 s to boot after login.

---

# Step 1 — Terminal A: pull the LLM (~9 GB)

```bash
ollama pull qwen2.5:14b
```

Leave it running. Move to Terminal B immediately.

---

# Step 2 — Terminal B: build the OCR image (~15–30 min)

This is the big one: it installs CPU torch and pre-downloads the EasyOCR weights
into the image so nothing is fetched at runtime.

```bash
docker compose build ocr
```

Leave it running. Move to Terminal C.

> If the build fails on the Red Hat base image, use a plain Python base for home
> testing — the work build keeps UBI:
> ```bash
> docker compose build --build-arg BASE_IMAGE=python:3.12-slim ocr
> ```

---

# Step 3 — Terminal C: everything else

## 3a. Start OpenSearch and Tika (fast)

```bash
docker compose up -d opensearch tika
docker compose ps            # wait for opensearch => healthy, ~60s
```

## 3b. Create the index

```bash
PYTHONPATH=services/worker .venv/Scripts/python.exe -m app.main bootstrap
```

Expect JSON showing `"exists": true, "chunks": 0`.

## 3c. First proof — no LLM, no OCR needed yet

Run this **now**, while A and B are still going:

```bash
.venv/Scripts/python.exe ops/smoke_test.py --skip-llm --skip-ocr
```

Stages 1–7 must pass. This proves Tika extraction, chunking, embedding,
indexing, BM25, kNN, RRF fusion and filtering all work. If this fails, stop and
fix it — nothing downstream can work.

---

# Step 4 — when Terminal B finishes: start OCR and prove it

```bash
docker compose up -d ocr
docker compose logs -f ocr        # wait for "loading EasyOCR reader for en"
```

First start takes ~60 s to load the models. Then:

```bash
.venv/Scripts/python.exe ops/smoke_test.py --skip-llm
```

Stage 9 is the OCR proof. It:
1. confirms Tika alone extracts ~nothing from `deploy/fixtures/scanned-facilities-policy.png`
2. ingests it and asserts the OCR route actually triggered
3. asserts the recognised text is **searchable** ("badge", "replacement", "security office")
4. asserts chunks are tagged `extraction_source="ocr"`

That is real proof, not just a health check.

---

# Step 5 — when Terminal A finishes: prove the whole thing

```bash
ollama list                       # confirm qwen2.5:14b is there
.venv/Scripts/python.exe ops/smoke_test.py
```

**All ten stages must pass.** This is the milestone. It now additionally proves
the LLM answers are *grounded* (the fact "seven years" comes from the document,
not the model's memory) and that answers carry `[n]` citations — including a
question answered from the **scanned** page.

---

# Step 6 — the API, and the SSE stream the UI depends on

Terminal C:

```bash
.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir services/api --port 8000 --reload
```

Terminal A (the pull is done):

```bash
.venv/Scripts/python.exe ops/smoke_test.py --api-url http://localhost:8000
```

Stage 10 now runs: `/health/ready`, `/config` (and asserts it leaks no secrets),
`/search`, `/index/stats`, and the real `/chat/stream` SSE endpoint — parsing the
exact byte stream the browser will consume. If stage 10 passes, the UI will work
before you even open it.

API docs: <http://localhost:8000/docs>

---

# Step 7 — the UI

Terminal B:

```bash
cd web && npm run dev
```

Open <http://localhost:3000> and:

1. Click **Upload**, choose `deploy/fixtures/scanned-facilities-policy.pdf`
   (the scanned one — proves OCR through the UI)
2. Wait for the chunk count, e.g. *"1 chunks, 1 OCR pages"*
3. Ask: **"How long do badge replacement requests take?"**
4. The answer streams in with `[1]` citations. Click the number to jump to the
   source panel on the right, which shows the page reference, which retriever
   found it (`bm25#2 · knn#1`), and that it came from OCR.

Then upload a real document of your own and ask about it.

## Bulk-ingest a folder

```bash
PYTHONPATH=services/worker .venv/Scripts/python.exe -m app.main ingest /c/path/to/docs --recursive
```

---

# Daily restart (once it all works)

```bash
cd /c/Users/Konrad/Desktop/Rag
export PATH="$PATH:$LOCALAPPDATA/Programs/Ollama:$LOCALAPPDATA/Programs/DockerDesktop/resources/bin"
docker compose up -d opensearch tika ocr
.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir services/api --port 8000 --reload &
cd web && npm run dev
```

Data persists in the `opensearch-data` volume — you do not re-ingest each time.

---

# If something breaks

| Symptom | Command |
|---|---|
| anything at all | `curl localhost:8000/api/v1/health/ready` — names the broken component |
| answers ignore your docs | `curl -X POST localhost:8000/api/v1/search -H "Content-Type: application/json" -d '{"query":"your question"}'` |
| LLM errors | `curl localhost:8000/api/v1/health/llm` |
| wrong model or index | `curl localhost:8000/api/v1/config` |
| nothing indexed | `PYTHONPATH=services/worker .venv/Scripts/python.exe -m app.main stats` |
| OCR not triggering | `curl localhost:9999/health` · `docker compose logs ocr` |
| OCR returns nonsense | raise `RAG_OCR__RENDER_DPI` to 300 in `.env` and restart the ocr container |
| scanned pages skipped | raise `RAG_OCR__MIN_CHARS_PER_PAGE` — pages above it are assumed to have a good text layer |
| OpenSearch won't start | `docker compose logs opensearch` — usually the 2 GB heap vs Docker's memory limit |

`ops/smoke_test.py` flags: `--skip-llm`, `--skip-ocr`, `--api-url URL`, `--keep`
(leaves the test documents in the index so you can inspect them in OpenSearch
Dashboards at <http://localhost:5601> after `docker compose up -d dashboards`).
