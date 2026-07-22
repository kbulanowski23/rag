"""rag_core — shared primitives for the RAG platform.

Import from the subpackages (rag_core.llm, rag_core.search, ...) rather than
from here, so that a process which never embeds does not import onnxruntime.
"""

__version__ = "0.1.0"

from rag_core.config import Settings, get_settings

__all__ = ["Settings", "get_settings", "__version__"]
