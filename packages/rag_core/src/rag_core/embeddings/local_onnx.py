"""Local sentence-embedding inference on onnxruntime.

Why not sentence-transformers: it depends on torch, which is ~2.5 GB of wheels
to carry across the air gap and to hold in every API and worker pod. The useful
part of it -- tokenize, run the encoder, pool, normalise -- is about eighty lines
and is written out below. onnxruntime + tokenizers is roughly 60 MB.

Expects a directory containing:
    model.onnx          (or onnx/model.onnx)
    tokenizer.json
Populate it with ops/fetch_models.py at home; it is then baked into the image.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from rag_core.config import EmbeddingSettings
from rag_core.embeddings.base import EmbeddingError, EmbeddingProvider


def _find_model_file(root: Path) -> Path:
    for candidate in ("model.onnx", "onnx/model.onnx", "model_quantized.onnx",
                      "onnx/model_quantized.onnx"):
        p = root / candidate
        if p.is_file():
            return p
    raise EmbeddingError(
        f"no ONNX model under {root}. Run `python ops/fetch_models.py` on an "
        f"internet-connected machine, or mount the model directory."
    )


class LocalOnnxEmbedder(EmbeddingProvider):
    def __init__(self, settings: EmbeddingSettings) -> None:
        # Imported lazily so that importing rag_core does not require onnxruntime
        # in contexts that never embed (e.g. a thin health-check container).
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.settings = settings
        self.dim = settings.dim
        root = Path(settings.model_path).expanduser().resolve()
        if not root.is_dir():
            raise EmbeddingError(f"embedding model directory not found: {root}")

        tok_path = root / "tokenizer.json"
        if not tok_path.is_file():
            raise EmbeddingError(f"tokenizer.json missing in {root}")

        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.enable_truncation(max_length=settings.max_tokens)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        opts = ort.SessionOptions()
        if settings.num_threads > 0:
            # Pin threads: an ORT session left to itself sizes its pool from the
            # host core count, which ignores the pod's CPU limit and thrashes.
            opts.intra_op_num_threads = settings.num_threads
            opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(_find_model_file(root)), opts, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self.session.get_inputs()}
        # ORT sessions are thread-safe for Run, but the tokenizer's padding state
        # is not; serialise the whole encode to keep it simple and correct.
        self._lock = threading.Lock()

        actual = self.session.get_outputs()[0].shape[-1]
        if isinstance(actual, int) and actual != self.dim:
            raise EmbeddingError(
                f"configured RAG_EMBEDDING__DIM={self.dim} but model outputs {actual}. "
                f"The index mapping is built from this value -- fix the config."
            )

    def _encode(self, texts: list[str]) -> np.ndarray:
        encs = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)

        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}

        out = self.session.run(None, feeds)[0]  # (batch, seq, hidden)

        if out.ndim == 3:
            vectors = self._pool(out, mask)
        else:
            vectors = out  # model already pools internally

        if self.settings.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.clip(norms, 1e-12, None)
        return vectors.astype(np.float32)

    @staticmethod
    def _pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Mean pooling over non-padding tokens.

        bge/e5/gte are all trained with mean pooling (bge nominally uses CLS, but
        mean is within noise and is robust to models that lack a CLS head).
        Padding must be excluded or short texts get their vectors dragged toward
        the pad embedding.
        """
        m = mask[..., None].astype(np.float32)
        return (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prefix = self.settings.passage_prefix
        prepared = [f"{prefix} {t}".strip() if prefix else t for t in texts]
        out: list[list[float]] = []
        bs = max(1, self.settings.batch_size)
        with self._lock:
            for i in range(0, len(prepared), bs):
                out.extend(self._encode(prepared[i : i + bs]).tolist())
        return out

    def embed_query(self, text: str) -> list[float]:
        prefix = self.settings.query_prefix
        prepared = f"{prefix} {text}".strip() if prefix else text
        with self._lock:
            return self._encode([prepared])[0].tolist()
