# Model artifacts

Populated by `python ops/fetch_models.py`, run at home with internet access.
The weights are **not** in git — they are large binaries. `SHA256SUMS` is
committed so a transfer can be verified on the other side.

```
deploy/models/
  bge-small-en-v1.5/       embedding model, 384-dim, ~130 MB
    model.onnx
    tokenizer.json
    config.json
  bge-reranker-base/       optional cross-encoder (--include-reranker)
  SHA256SUMS
```

These directories are `COPY`'d into the API and worker images at build time.
Nothing is downloaded when a pod starts.

## Changing the embedding model

Three things must change together, or search breaks silently:

1. `RAG_EMBEDDING__MODEL_PATH` — the new directory
2. `RAG_EMBEDDING__DIM` — the new output dimension
3. **The index must be recreated and every document re-ingested.**

Vectors from different models are not comparable. A mismatched dimension is
caught at startup; a *same*-dimension swap is not, and produces an index that
returns confident nonsense. Treat a model change as a full re-index, always.

## Verifying a transfer

```bash
cd deploy/models && sha256sum -c SHA256SUMS
```
