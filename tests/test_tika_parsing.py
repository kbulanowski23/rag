"""Tika's XHTML is where page numbers come from, so the parser is tested against
representative output rather than trusted."""

from __future__ import annotations

from rag_core.extraction.tika import _PageParser, content_disposition

PDF_XHTML = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><meta name="Content-Type" content="application/pdf"/><title>Handbook</title></head>
<body>
<div class="page"><p>Page one text about retention.</p><p>More on page one.</p></div>
<div class="page"><p>Page two covers escalation.</p></div>
<div class="page"><p>Page three is nearly blank.</p></div>
</body></html>"""

DOCX_XHTML = """<html><body><p>A document with no page divisions.</p>
<p>Second paragraph.</p></body></html>"""


def parse(xhtml: str) -> list[str]:
    p = _PageParser()
    p.feed(xhtml)
    return p.result()


def test_pages_are_separated():
    pages = parse(PDF_XHTML)
    assert len(pages) == 3
    assert "retention" in pages[0]
    assert "escalation" in pages[1]
    # Page one's content must not bleed into page two, or citations point at
    # the wrong page.
    assert "escalation" not in pages[0]


def test_paragraphs_within_a_page_are_kept_together():
    pages = parse(PDF_XHTML)
    assert "Page one text" in pages[0] and "More on page one" in pages[0]


def test_formats_without_pages_produce_one_page():
    pages = parse(DOCX_XHTML)
    assert len(pages) == 1
    assert "no page divisions" in pages[0]
    assert "Second paragraph" in pages[0]


def test_script_and_style_content_is_dropped():
    pages = parse(
        "<html><body><div class='page'><script>var x=1;</script>"
        "<style>p{color:red}</style><p>Real content.</p></div></body></html>"
    )
    assert "Real content." in pages[0]
    assert "var x" not in pages[0]
    assert "color:red" not in pages[0]


def test_empty_body():
    assert parse("<html><body></body></html>") == []


# -- Content-Disposition ------------------------------------------------------
# Filenames arrive from users and shared drives, not from us. httpx encodes
# header values as ASCII and raises, so a curly apostrophe in a filename used to
# fail the ingest with a 500 before Tika was ever contacted.


def encodable(filename: str) -> str:
    value = content_disposition(filename)
    value.encode("ascii")  # what httpx does; raises if we got this wrong
    return value


def test_ascii_filename_is_unchanged_in_the_fallback():
    assert 'filename="handbook.pdf"' in encodable("handbook.pdf")


def test_curly_apostrophe_survives():
    value = encodable("Konrad’s policy.pdf")
    assert 'filename="Konrads policy.pdf"' in value
    # The true name is still carried, percent-encoded.
    assert "filename*=UTF-8''Konrad%E2%80%99s%20policy.pdf" in value


def test_accents_fold_to_ascii():
    assert 'filename="resume.docx"' in encodable("résumé.docx")


def test_extension_survives_a_fully_non_ascii_name():
    # Tika picks a parser by extension, so losing it would change the outcome.
    assert 'filename="upload.pdf"' in encodable("政策文件.pdf")


def test_quotes_and_control_characters_cannot_inject_a_header():
    value = encodable('bad"name\r\nX-Injected: 1.pdf')
    assert "X-Injected: 1.pdf" in value  # kept as text...
    assert "\r" not in value and "\n" not in value  # ...but not as a header
    assert value.count('"') == 2
