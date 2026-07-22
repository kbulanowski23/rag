"""Optional cross-encoder reranking, on onnxruntime.

Off by default (RAG_RETRIEVAL__RERANK_ENABLED=false). Turn it on once the base
retrieval is measurably working -- a reranker papers over retrieval problems and
makes them harder to diagnose, and it adds real latency: it scores every
candidate against the query, so it is O(candidates) forward passes per search.

If the model directory is absent, this degrades to a no-op rather than failing
the request; a missing optional model must never take the search path down.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from rag_core.config import RetrievalSettings
from rag_core.documents import SearchHit

log = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_path: str, max_tokens: int = 512) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        root = Path(model_path).expanduser().resolve()
        model_file = next(
            (root / c for c in ("model.onnx", "onnx/model.onnx") if (root / c).is_file()), None
        )
        if model_file is None:
            raise FileNotFoundError(f"no ONNX model under {root}")

        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_tokens)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.session = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self.session.get_inputs()}
        self._lock = threading.Lock()

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        pairs = [(query, h.text) for h in hits]
        with self._lock:
            encs = self.tokenizer.encode_batch(pairs)
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.array(
                    [e.type_ids for e in encs], dtype=np.int64
                )
            feeds = {k: v for k, v in feeds.items() if k in self._input_names}
            logits = self.session.run(None, feeds)[0]

        scores = logits[:, 0] if logits.ndim == 2 and logits.shape[1] == 1 else logits.reshape(len(hits), -1)[:, -1]
        for hit, score in zip(hits, scores):
            hit.score = float(score)
            hit.retrievers["rerank"] = 1
        return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]


_reranker: CrossEncoderReranker | None = None
_tried = False


def get_reranker(settings: RetrievalSettings) -> CrossEncoderReranker | None:
    global _reranker, _tried
    if _tried:
        return _reranker
    _tried = True
    try:
        _reranker = CrossEncoderReranker(settings.rerank_model_path)
        log.info("reranker loaded from %s", settings.rerank_model_path)
    except Exception as e:
        log.warning("reranking requested but unavailable (%s); continuing without it", e)
        _reranker = None
    return _reranker
