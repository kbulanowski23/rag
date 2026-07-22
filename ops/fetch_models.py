"""Download model artifacts for baking into images.

Run this ONCE, at home, on a machine with internet access. The downloaded files
are committed to deploy/models/ (or carried across the air gap alongside the
images) and are then baked in at build time. Nothing downloads at runtime.

    python ops/fetch_models.py
    python ops/fetch_models.py --include-reranker

Only the standard library is used, so this works before any dependency is
installed. Files come from the HuggingFace CDN over plain HTTPS -- if your home
network blocks it, download the four files by hand and drop them in the target
directory; nothing else about the process changes.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

HF = "https://huggingface.co"
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "deploy" / "models"

# The embedding model. 384 dimensions, ~130 MB, strong retrieval quality for its
# size, permissive licence. If you change it, update RAG_EMBEDDING__DIM to match
# and rebuild the index -- vectors from a different model are not comparable.
EMBEDDING_MODEL = {
    "name": "bge-small-en-v1.5",
    "repo": "BAAI/bge-small-en-v1.5",
    "files": [
        "onnx/model.onnx",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
    ],
}

RERANKER_MODEL = {
    "name": "bge-reranker-base",
    "repo": "BAAI/bge-reranker-base",
    "files": ["onnx/model.onnx", "tokenizer.json", "config.json"],
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  = {dest.relative_to(REPO_ROOT)} (already present)")
        return
    # ASCII only: the default Windows console codepage is cp1252 and raises
    # UnicodeEncodeError on anything outside it, which would crash the download.
    print(f"  -> GET {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "rag-fetch-models/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f, length=1024 * 256)
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"failed to download {url}: HTTP {e.code}") from e
    except urllib.error.URLError as e:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"failed to reach {url}: {e.reason}") from e
    tmp.replace(dest)
    print(f"    -> {dest.relative_to(REPO_ROOT)} ({dest.stat().st_size / 1e6:.1f} MB)")


def fetch(model: dict) -> Path:
    target = MODELS_DIR / model["name"]
    print(f"\n{model['repo']} -> {target.relative_to(REPO_ROOT)}")
    for remote in model["files"]:
        # Flatten onnx/model.onnx to model.onnx so the loader finds it either way.
        local = target / ("model.onnx" if remote.endswith("model.onnx") else Path(remote).name)
        download(f"{HF}/{model['repo']}/resolve/main/{remote}", local)
    return target


def write_manifest(paths: list[Path]) -> None:
    """A checksum manifest so the air-gapped side can verify the transfer.

    Media-based transfers do get corrupted, and a truncated ONNX file fails in a
    confusing way at pod startup rather than an obvious one at copy time.
    """
    lines = []
    for root in paths:
        for f in sorted(root.rglob("*")):
            if f.is_file() and f.suffix != ".part":
                digest = hashlib.sha256(f.read_bytes()).hexdigest()
                lines.append(f"{digest}  {f.relative_to(MODELS_DIR).as_posix()}")
    manifest = MODELS_DIR / "SHA256SUMS"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {manifest.relative_to(REPO_ROOT)} ({len(lines)} files)")
    print("verify on the target with:  cd deploy/models && sha256sum -c SHA256SUMS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch model artifacts for offline use")
    parser.add_argument("--include-reranker", action="store_true")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fetched = [fetch(EMBEDDING_MODEL)]
    if args.include_reranker:
        fetched.append(fetch(RERANKER_MODEL))
    write_manifest(fetched)

    print("\nNext: python -m app.main bootstrap   (from services/worker)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
