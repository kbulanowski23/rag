# Build context is the repo root:  docker build -f services/api/Dockerfile .
#
# UBI is used rather than python:3.12-slim because OpenShift clusters commonly
# restrict images to Red Hat bases, and because ubi-minimal is the one your
# security team will already have scanned.
ARG BASE_IMAGE=registry.access.redhat.com/ubi8/python-312:latest
FROM ${BASE_IMAGE} AS builder

USER 0
WORKDIR /build

COPY services/api/requirements.txt ./requirements.txt
COPY packages/rag_core /build/packages/rag_core

# --no-deps on rag_core: its dependencies are already pinned in requirements.txt
# and we do not want pip resolving a different set behind our back.
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt \
 && /opt/venv/bin/pip install --no-cache-dir --no-deps /build/packages/rag_core


FROM ${BASE_IMAGE}

ARG BUILD_VERSION=0.1.0
LABEL org.opencontainers.image.title="rag-api" \
      org.opencontainers.image.version="${BUILD_VERSION}"

USER 0
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY services/api/app /app/app
# Model weights are baked in. Nothing is fetched at runtime -- this is the whole
# point of the air-gap design. Run ops/fetch_models.py before building.
COPY deploy/models/bge-small-en-v1.5 /models/bge-small-en-v1.5

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RAG_EMBEDDING__MODEL_PATH=/models/bge-small-en-v1.5 \
    # OpenShift enforces CPU limits the ORT thread pool cannot see; pin it.
    RAG_EMBEDDING__NUM_THREADS=2

# OpenShift runs containers as an arbitrary UID in the root group. Group-writable
# permissions are what make that work; a hardcoded `USER 1001` alone does not.
RUN chgrp -R 0 /app /models /opt/venv && chmod -R g=u /app /models /opt/venv

USER 1001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/v1/health/live',timeout=3).status==200 else 1)"

# One worker per pod: scale with replicas, not with in-pod workers. Each worker
# would load its own copy of the ONNX session and multiply the memory footprint.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-keep-alive", "75", "--no-access-log"]
