# Quickstart — proving the pipeline works

Goal: get `ops/smoke_test.py` printing all-green. That script pushes a document
through Tika, chunking, ONNX embedding, OpenSearch, hybrid retrieval and the LLM,
and asserts a specific fact comes back grounded and cited. When it passes, the
concept is proven end to end.

## Prerequisites

Two things must be installed first. Neither can be worked around.

**Docker Desktop** — OpenSearch, Tika and EasyOCR all run as containers.
<https://www.docker.com/products/docker-desktop/>

**Python 3.12** — `onnxruntime` and `easyocr` publish no 3.14 wheels, and the
work environment is very unlikely to be on 3.14 either.
<https://www.python.org/downloads/release/python-31210/> — tick "Add to PATH".

**WSL2** — Docker Desktop on Windows runs its engine inside WSL2. If it is
missing, Docker Desktop reports *"Virtualization support not detected"*, which is
misleading: on most machines virtualization is already enabled in the BIOS and
the actual missing piece is WSL. In an **Administrator** PowerShell:

```powershell
wsl --install
```

Then **reboot**. Verify with `wsl --status` and `docker info`.

To check which problem you actually have:

```powershell
(Get-CimInstance Win32_Processor).VirtualizationFirmwareEnabled   # False => BIOS
wsl --status                                                       # error => WSL
```

Only if the first returns `False` do you need the BIOS: reboot into UEFI and
enable **SVM Mode** (AMD) or **Intel VT-x** (Intel).

**Ollama** (for the LLM at home) — <https://ollama.com/download/windows>

```powershell
ollama pull qwen2.5:14b     # ~9 GB, comfortable on a 16 GB 5070 Ti
```

### PATH after installing

Docker Desktop and Ollama install to per-user directories and are **not** on PATH
in terminals that were already open. Open a new terminal, or in git-bash:

```bash
export PATH="$PATH:$LOCALAPPDATA/Programs/Ollama:$LOCALAPPDATA/Programs/DockerDesktop/resources/bin"
```

## Setup

```bash
cd /c/Users/Konrad/Desktop/Rag
cp .env.example .env

# 1. Python 3.12 venv + all dependencies (recreates .venv)
bash ops/setup_dev.sh

# 2. Fetch the embedding model (needs internet; do this at home, once)
.venv/Scripts/python.exe ops/fetch_models.py

# 3. Backing services
docker compose up -d opensearch tika
docker compose ps                        # wait for opensearch to be healthy

# 4. Create the index
PYTHONPATH=services/worker .venv/Scripts/python.exe -m app.main bootstrap
```

If you had `.venv` activated in a shell before running `setup_dev.sh`, re-activate
it — the script recreates the directory.

## Prove it

```powershell
# Retrieval only — no LLM needed. Proves extraction, embedding, indexing, search.
python ops\smoke_test.py --skip-llm

# Full path including generation.
ollama serve                             # in another terminal, if not already running
python ops\smoke_test.py
```

All-green means the architecture works. Everything after this is tuning.

## Then run it for real

```powershell
uvicorn app.main:app --app-dir services\api --reload --port 8000
# http://localhost:8000/docs

cd web; npm install; npm run dev
# http://localhost:3000
```

The UI calls `/api/v1/*` on its own origin and Next proxies that to the API
server-side (`web/next.config.mjs`). The target is `RAG_API_URL`, read at server
startup and defaulting to `http://localhost:8000`, so local development needs no
configuration and no CORS entry.

Ingest a folder of your own documents:

```powershell
$env:PYTHONPATH="services\worker"; python -m app.main ingest C:\path\to\docs --recursive
```

## OCR

The OCR image takes 15–30 minutes to build the first time (torch is ~2.5 GB).
Skip it until the rest is green — the pipeline degrades cleanly without it and
only scanned documents are affected.

```powershell
docker compose up -d --build ocr
```

## Useful checks

| Command | Answers |
|---|---|
| `GET /api/v1/config` | what this pod actually resolved — model, index, fusion |
| `GET /api/v1/health/ready` | which dependency is down |
| `GET /api/v1/health/llm` | can it reach the model endpoint |
| `POST /api/v1/search` | retrieval without generation, with per-retriever ranks |
| `python -m app.main stats` | how many documents and chunks are indexed |

## Switching to the work LLM

Change four env vars, nothing else:

```
RAG_LLM__PROVIDER=azure_openai
RAG_LLM__BASE_URL=https://<resource>.openai.azure.com
RAG_LLM__MODEL=<deployment-name>
RAG_LLM__API_KEY=<key>
```

Then `python ops/smoke_test.py` again. That is the whole switch — and it is worth
testing at home against a second endpoint before you rely on it at work.

## Known dependency findings

`npm audit` reports a moderate `postcss` advisory reachable only through Next's
own build tooling. No released Next version resolves it, and `npm audit fix
--force` would downgrade to Next 9. It does not ship in the runtime image
(`.next/standalone` is 14.6 MB and contains neither postcss nor sharp). Document
it rather than chase it.
