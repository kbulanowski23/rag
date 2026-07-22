"""Lets `pytest` run from the repo root without installing rag_core first.

In CI and in the images, rag_core is a proper installed package; this is purely
a developer convenience so a fresh clone can run the tests immediately.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "packages" / "rag_core" / "src"))
