"""Client for the EasyOCR sidecar.

The OCR service is a separate pod for one reason: EasyOCR needs torch, and torch
is ~2.5 GB. Keeping it behind HTTP means the API and worker images stay small,
are quick to move across the air gap, and can scale independently from the thing
that actually needs the CPU (or GPU).
"""

from __future__ import annotations

import base64
import logging

import httpx

from rag_core.config import OCRSettings

log = logging.getLogger(__name__)


class OCRError(RuntimeError):
    pass


class OCRClient:
    def __init__(self, settings: OCRSettings) -> None:
        self.s = settings
        self.client = httpx.Client(
            base_url=settings.url, timeout=httpx.Timeout(settings.timeout_s, connect=10.0)
        )

    def health(self) -> bool:
        try:
            return self.client.get("/health", timeout=5.0).status_code == 200
        except httpx.HTTPError:
            return False

    def ocr_image(self, image_bytes: bytes, languages: list[str] | None = None) -> tuple[str, float]:
        """Returns (text, mean_confidence)."""
        payload = {
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            "languages": languages or self.s.language_list,
        }
        try:
            r = self.client.post("/ocr", json=payload)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as e:
            raise OCRError(f"ocr HTTP {e.response.status_code}: {e.response.text[:300]}") from e
        except httpx.HTTPError as e:
            raise OCRError(f"ocr unreachable: {e}") from e
        return data.get("text", ""), float(data.get("mean_confidence") or 0.0)

    def ocr_pdf(self, pdf_bytes: bytes, pages: list[int] | None = None,
                languages: list[str] | None = None) -> dict[int, tuple[str, float]]:
        """OCR selected pages of a PDF. The service renders the pages itself so
        that no PDF rasteriser is needed in the worker image."""
        payload = {
            "pdf_base64": base64.b64encode(pdf_bytes).decode("ascii"),
            "pages": pages,
            "languages": languages or self.s.language_list,
        }
        try:
            r = self.client.post("/ocr/pdf", json=payload)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as e:
            raise OCRError(f"ocr HTTP {e.response.status_code}: {e.response.text[:300]}") from e
        except httpx.HTTPError as e:
            raise OCRError(f"ocr unreachable: {e}") from e
        return {
            int(p["page"]): (p.get("text", ""), float(p.get("mean_confidence") or 0.0))
            for p in data.get("pages", [])
        }

    def close(self) -> None:
        self.client.close()
