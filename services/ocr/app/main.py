"""EasyOCR behind a small HTTP API.

Isolated from every other service because of what it drags in: torch, torchvision
and the detection/recognition weights. Keeping it here means the API and worker
images stay ~200 MB instead of ~3 GB, which matters a great deal when images have
to be carried across an air gap on physical media.

PDF page rendering uses pypdfium2 rather than PyMuPDF: PyMuPDF is AGPL, which
most enterprise legal reviews will not clear. pypdfium2 is Apache/BSD.

Models: EasyOCR downloads weights on first use by default. That is fatal in an
air-gapped environment, so the Dockerfile pre-downloads them into
EASYOCR_MODULE_PATH and this service runs with download_enabled=False.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

logging.basicConfig(
    level=os.getenv("RAG_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("rag-ocr")

MODEL_DIR = os.getenv("EASYOCR_MODULE_PATH", "/models/easyocr")
DEFAULT_LANGS = [x.strip() for x in os.getenv("RAG_OCR__LANGUAGES", "en").split(",") if x.strip()]
USE_GPU = os.getenv("RAG_OCR__GPU", "false").lower() == "true"
RENDER_DPI = int(os.getenv("RAG_OCR__RENDER_DPI", "200"))
MAX_PAGES = int(os.getenv("RAG_OCR__MAX_PAGES", "500"))

_readers: dict[str, object] = {}
_lock = threading.Lock()


def get_reader(languages: list[str]):
    """One Reader per language set, created once.

    Constructing a Reader loads the detection and recognition models -- several
    seconds and a few hundred MB. Never do it per request.
    """
    import easyocr

    key = ",".join(sorted(languages))
    with _lock:
        if key not in _readers:
            log.info("loading EasyOCR reader for %s (gpu=%s)", key, USE_GPU)
            _readers[key] = easyocr.Reader(
                languages,
                gpu=USE_GPU,
                model_storage_directory=MODEL_DIR,
                user_network_directory=MODEL_DIR,
                # Must stay False: a download attempt in the air-gapped
                # environment hangs until timeout instead of failing fast.
                download_enabled=False,
                verbose=False,
            )
        return _readers[key]


class ImageRequest(BaseModel):
    image_base64: str
    languages: list[str] = Field(default_factory=lambda: list(DEFAULT_LANGS))


class PdfRequest(BaseModel):
    pdf_base64: str
    pages: list[int] | None = None   # 1-based; None means every page
    languages: list[str] = Field(default_factory=lambda: list(DEFAULT_LANGS))


class PageResult(BaseModel):
    page: int
    text: str
    mean_confidence: float


class OcrResult(BaseModel):
    text: str
    mean_confidence: float
    boxes: int


class PdfResult(BaseModel):
    pages: list[PageResult]


def _run_ocr(image_bytes: bytes, languages: list[str]) -> tuple[str, float, int]:
    from PIL import Image

    reader = get_reader(languages)
    with Image.open(io.BytesIO(image_bytes)) as img:
        array = np.array(img.convert("RGB"))

    # detail=1 gives (bbox, text, confidence); paragraph=False keeps the
    # confidences, which we surface so downstream can flag unreliable pages.
    results = reader.readtext(array, detail=1, paragraph=False)
    if not results:
        return "", 0.0, 0

    # Sort top-to-bottom then left-to-right so the text reads in natural order;
    # EasyOCR's native ordering follows detection, not layout.
    def sort_key(item):
        bbox = item[0]
        top = min(p[1] for p in bbox)
        left = min(p[0] for p in bbox)
        return (round(top / 12), left)   # bucket rows to tolerate skew

    results.sort(key=sort_key)
    lines = [str(text) for _, text, _ in results if str(text).strip()]
    confs = [float(conf) for _, _, conf in results if conf is not None]
    return "\n".join(lines), (sum(confs) / len(confs) if confs else 0.0), len(results)


def _render_pdf_pages(pdf_bytes: bytes, pages: list[int] | None) -> list[tuple[int, bytes]]:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        total = len(doc)
        wanted = pages or list(range(1, total + 1))
        wanted = [p for p in wanted if 1 <= p <= total][:MAX_PAGES]
        out: list[tuple[int, bytes]] = []
        scale = RENDER_DPI / 72.0
        for number in wanted:
            page = doc[number - 1]
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil()
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            out.append((number, buf.getvalue()))
        return out
    finally:
        doc.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the default reader so the first real request is not the one that pays
    # the model-loading cost, and so a missing model file fails at startup.
    try:
        get_reader(DEFAULT_LANGS)
    except Exception:
        log.exception("could not preload EasyOCR models from %s", MODEL_DIR)
    yield


app = FastAPI(title="RAG OCR Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "languages": DEFAULT_LANGS, "gpu": USE_GPU, "loaded": list(_readers)}


@app.post("/ocr", response_model=OcrResult)
async def ocr_image(req: ImageRequest) -> OcrResult:
    try:
        data = base64.b64decode(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid base64: {e}") from e

    try:
        text, conf, boxes = await run_in_threadpool(_run_ocr, data, req.languages)
    except Exception as e:
        log.exception("ocr failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    return OcrResult(text=text, mean_confidence=conf, boxes=boxes)


@app.post("/ocr/pdf", response_model=PdfResult)
async def ocr_pdf(req: PdfRequest) -> PdfResult:
    try:
        data = base64.b64decode(req.pdf_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid base64: {e}") from e

    def work() -> list[PageResult]:
        out: list[PageResult] = []
        for number, png in _render_pdf_pages(data, req.pages):
            text, conf, _ = _run_ocr(png, req.languages)
            out.append(PageResult(page=number, text=text, mean_confidence=conf))
        return out

    try:
        pages = await run_in_threadpool(work)
    except Exception as e:
        log.exception("pdf ocr failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    return PdfResult(pages=pages)
