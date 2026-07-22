"""Apache Tika client.

We ask Tika for XHTML rather than plain text, because the XHTML carries page
boundaries (`<div class="page">`) that plain text throws away. Page numbers are
what make a citation actionable -- "it's in the 400-page handbook somewhere" is
not a citation.

Tika is used as a pure HTTP service (tika-server). We never embed the Java
library, and nothing here needs a JVM in the Python image.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from html.parser import HTMLParser
from urllib.parse import quote

import httpx

from rag_core.config import TikaSettings
from rag_core.documents import Page

log = logging.getLogger(__name__)


class TikaError(RuntimeError):
    pass


# Control characters would let a crafted filename inject header lines; quotes
# and backslashes would break out of the quoted-string.
_HEADER_UNSAFE = re.compile(r'[\x00-\x1f\x7f"\\]')


def content_disposition(filename: str) -> str:
    """Build a Content-Disposition value that survives ASCII header encoding.

    Real filenames off a shared drive carry curly apostrophes, accents, em
    dashes and non-Latin scripts. HTTP header values are ASCII, and httpx
    raises UnicodeEncodeError rather than guessing -- which turns an ordinary
    document into a 500 before Tika is ever reached.

    RFC 6266: an ASCII-folded `filename` that older parsers can read, plus
    `filename*` carrying the true name percent-encoded as UTF-8. We only send
    this so Tika can pick a parser by extension when the content type is a lie,
    so the fallback losing an accent costs nothing -- but the extension must
    survive, which is why the folded name is repaired rather than dropped.
    """
    folded = unicodedata.normalize("NFKD", filename).encode("ascii", "ignore").decode("ascii")
    folded = _HEADER_UNSAFE.sub("", folded).strip()
    if "." in folded:
        stem, _, ext = folded.rpartition(".")
        if not stem:
            # Nothing ASCII survived left of the dot (e.g. a fully CJK name).
            folded = f"upload.{ext}"
    elif not folded:
        folded = "upload"
    return f"attachment; filename=\"{folded}\"; filename*=UTF-8''{quote(filename, safe='')}"


class _PageParser(HTMLParser):
    """Pulls per-page text out of Tika's XHTML.

    Written against stdlib HTMLParser rather than bringing in lxml or
    BeautifulSoup: the markup Tika emits is machine-generated and regular, and
    this keeps the air-gapped dependency list one entry shorter.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pages: list[list[str]] = []
        self._depth_stack: list[str] = []
        self._in_body = False
        self._skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        attr = dict(attrs)
        if tag == "body":
            self._in_body = True
        elif tag in ("script", "style"):
            self._skip += 1
        elif tag == "div" and attr.get("class") == "page":
            self.pages.append([])
        elif tag in ("p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4"):
            self._break()

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        elif tag in ("p", "div", "tr", "li", "h1", "h2", "h3", "h4"):
            self._break()

    def _break(self) -> None:
        if self.pages and self.pages[-1] and self.pages[-1][-1] != "\n":
            self.pages[-1].append("\n")

    def handle_data(self, data: str) -> None:
        if not self._in_body or self._skip or not data.strip():
            return
        if not self.pages:
            # Formats without page structure (html, txt, docx) get one page.
            # This must be gated on there being real content: the whitespace
            # between <body> and the first <div class="page"> would otherwise
            # open a phantom page 1 and shift every page number by one.
            self.pages.append([])
        self.pages[-1].append(data)

    def result(self) -> list[str]:
        out = []
        for parts in self.pages:
            text = "".join(parts)
            text = re.sub(r"\n{3,}", "\n\n", text)
            out.append(text.strip())
        return out


class TikaClient:
    def __init__(self, settings: TikaSettings) -> None:
        self.s = settings
        self.client = httpx.Client(
            base_url=settings.url, timeout=httpx.Timeout(settings.timeout_s, connect=10.0)
        )

    def health(self) -> bool:
        try:
            return self.client.get("/tika", timeout=5.0).status_code == 200
        except httpx.HTTPError:
            return False

    def extract_pages(self, data: bytes, filename: str, content_type: str = "") -> list[Page]:
        headers = {"Accept": "text/html"}
        if content_type:
            headers["Content-Type"] = content_type
        # Lets Tika pick a parser by extension when the content type is a lie,
        # which it frequently is for files coming out of a shared drive.
        headers["Content-Disposition"] = content_disposition(filename)
        if self.s.skip_builtin_ocr:
            headers["X-Tika-OCRskipOcr"] = "true"
        try:
            r = self.client.put("/tika", content=data, headers=headers)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise TikaError(f"tika HTTP {e.response.status_code} for {filename}: {e.response.text[:300]}") from e
        except httpx.HTTPError as e:
            raise TikaError(f"tika unreachable for {filename}: {e}") from e

        parser = _PageParser()
        parser.feed(r.text)
        texts = parser.result()
        return [Page(number=i, text=t, source="tika") for i, t in enumerate(texts, start=1)]

    def extract_metadata(self, data: bytes, filename: str, content_type: str = "") -> dict:
        headers = {"Accept": "application/json"}
        if content_type:
            headers["Content-Type"] = content_type
        headers["Content-Disposition"] = content_disposition(filename)
        try:
            r = self.client.put("/meta", content=data, headers=headers)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            log.warning("tika metadata failed for %s: %s", filename, e)
            return {}

    def detect_type(self, data: bytes, filename: str) -> str:
        try:
            r = self.client.put(
                "/detect/stream",
                content=data,
                headers={"Content-Disposition": content_disposition(filename)},
            )
            r.raise_for_status()
            return r.text.strip()
        except httpx.HTTPError:
            return ""

    def close(self) -> None:
        self.client.close()
