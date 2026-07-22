"""Generate a fake 'scanned' document for testing the OCR path.

Testing OCR properly needs a document with **no text layer** -- a page that is
purely pixels. Real scanned PDFs are awkward to keep in a repo, so this renders
one: text drawn into an image, saved as PNG and as an image-only PDF.

Tika will extract essentially nothing from these, which is exactly the condition
that triggers the OCR route in IngestPipeline. If OCR is broken, the document
ingests with zero chunks and the smoke test fails loudly.

    python ops/make_scan_fixture.py

Needs Pillow (requirements-dev.txt). Output goes to deploy/fixtures/.
Both files are committed, so this normally only needs running once.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "deploy" / "fixtures"

# Mild noise and imperfect alignment are deliberate: a pixel-perfect render is
# an unrealistically easy target and would not exercise OCR the way a real scan
# does. The facts below are what the smoke test asserts on.
LINES = [
    ("ACME CORPORATION", 46, True),
    ("Facilities Access Policy", 32, True),
    ("", 20, False),
    ("Document reference: FAC-2019-114", 26, False),
    ("", 14, False),
    ("Badge replacement requests are processed within", 26, False),
    ("three business days by the Security Office.", 26, False),
    ("", 14, False),
    ("Visitors must be escorted at all times while", 26, False),
    ("inside the secure data hall on level 4.", 26, False),
    ("", 14, False),
    ("After-hours access requires written approval", 26, False),
    ("from the Facilities Director, extension 2280.", 26, False),
]


def build_image():
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1240, 1000  # roughly A4 at 150 dpi
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    def font_for(size: int, bold: bool):
        # DejaVu ships with Pillow; fall back to the bitmap default if a build
        # lacks it, which still OCRs acceptably.
        for name in (("DejaVuSans-Bold.ttf", "arialbd.ttf") if bold
                     else ("DejaVuSans.ttf", "arial.ttf")):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    y = 70
    for text, size, bold in LINES:
        if text:
            draw.text((90, y), text, fill=(15, 15, 15), font=font_for(size, bold))
        y += size + 14

    # A faint border and a scanner-ish grey speckle, so the input is not a
    # pristine synthetic bitmap.
    draw.rectangle([(40, 40), (width - 40, height - 40)], outline=(190, 190, 190), width=2)
    import random

    random.seed(7)
    px = img.load()
    for _ in range(4000):
        x, yy = random.randrange(width), random.randrange(height)
        v = px[x, yy]
        if v == (255, 255, 255):
            g = random.randrange(225, 250)
            px[x, yy] = (g, g, g)
    return img


def main() -> int:
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("Pillow is required:  pip install -r requirements-dev.txt", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = build_image()

    png = OUT_DIR / "scanned-facilities-policy.png"
    img.save(png, format="PNG")
    print(f"wrote {png.relative_to(REPO_ROOT)} ({png.stat().st_size / 1024:.0f} KB)")

    # An image-only PDF: Pillow embeds the bitmap with no text layer at all,
    # which is precisely what a flatbed scanner produces.
    pdf = OUT_DIR / "scanned-facilities-policy.pdf"
    img.convert("RGB").save(pdf, format="PDF", resolution=150.0)
    print(f"wrote {pdf.relative_to(REPO_ROOT)} ({pdf.stat().st_size / 1024:.0f} KB)")

    print("\nThese have NO text layer. Tika will return ~nothing for them;")
    print("only the OCR path can extract their content.")
    print("\nTest with:  python ops/smoke_test.py   (OCR stage runs automatically)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
