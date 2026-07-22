# Build context is the repo root:  docker build -f services/worker/Dockerfile .
ARG BASE_IMAGE=registry.access.redhat.com/ubi9/python-312:latest
FROM ${BASE_IMAGE} AS builder

USER 0
WORKDIR /build

COPY services/worker/requirements.txt ./requirements.txt
COPY packages/rag_core /build/packages/rag_core

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt \
 && /opt/venv/bin/pip install --no-cache-dir --no-deps /build/packages/rag_core


FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="rag-worker"

USER 0
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY services/worker/app /app/app
COPY deploy/models/bge-small-en-v1.5 /models/bge-small-en-v1.5

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RAG_EMBEDDING__MODEL_PATH=/models/bge-small-en-v1.5 \
    RAG_EMBEDDING__NUM_THREADS=4

RUN chgrp -R 0 /app /models /opt/venv && chmod -R g=u /app /models /opt/venv

USER 1001

# Overridden by the Job/CronJob spec; `stats` is a harmless default that proves
# connectivity if someone runs the image with no arguments.
ENTRYPOINT ["python", "-m", "app.main"]
CMD ["stats"]
