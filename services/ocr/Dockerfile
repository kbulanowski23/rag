# Build context is the repo root:  docker build -f services/ocr/Dockerfile .
#
# This is the heavy image (~3 GB) because of torch. It is built once and changes
# rarely; keep it out of the frequent build/deploy loop.
ARG BASE_IMAGE=registry.access.redhat.com/ubi9/python-312:latest
FROM ${BASE_IMAGE} AS builder

USER 0
WORKDIR /build

COPY services/ocr/requirements.txt ./requirements.txt

# CPU wheels only. The default index serves the CUDA build, which adds ~2.5 GB
# of libraries this pod will never call.
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
 && /opt/venv/bin/pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt

# Pre-download the EasyOCR weights at build time. Without this the container
# tries to fetch them on first request, which fails in the air-gapped cluster.
ENV EASYOCR_MODULE_PATH=/models/easyocr
ARG OCR_LANGUAGES=en
RUN mkdir -p /models/easyocr \
 && /opt/venv/bin/python -c "\
import os, easyocr; \
langs=os.environ.get('OCR_LANGUAGES','en').split(','); \
easyocr.Reader(langs, gpu=False, model_storage_directory='/models/easyocr', \
               user_network_directory='/models/easyocr', download_enabled=True, verbose=False); \
print('easyocr models cached:', os.listdir('/models/easyocr'))"


FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="rag-ocr"

USER 0
WORKDIR /app

# opencv-python-headless still needs a few shared libraries present.
RUN microdnf install -y libglvnd-glx libgomp mesa-libGL \
 && microdnf clean all || true

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /models/easyocr /models/easyocr
COPY services/ocr/app /app/app

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EASYOCR_MODULE_PATH=/models/easyocr \
    RAG_OCR__GPU=false \
    RAG_OCR__RENDER_DPI=200 \
    # torch sizes its thread pool from the host core count and ignores the pod
    # CPU limit, causing severe contention. Pin it.
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

RUN chgrp -R 0 /app /models /opt/venv && chmod -R g=u /app /models /opt/venv

USER 1001
EXPOSE 9999

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:9999/health',timeout=5).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9999", "--workers", "1"]
