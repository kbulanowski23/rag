#!/usr/bin/env bash
# Local dev environment setup (Windows/git-bash or Linux).
#
# Recreates .venv on Python 3.12 and installs everything needed to run the API
# and the ingest worker. Python 3.13+ is rejected on purpose: onnxruntime and
# easyocr publish no wheels for it.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"

# Docker Desktop and Ollama install user-local on Windows and are not on PATH
# in shells that were open before the installers ran. git-bash sets USERNAME,
# not USER, so fall back rather than assume either exists.
WHO="${USER:-${USERNAME:-$(whoami 2>/dev/null | sed 's#.*\\\\##')}}"
LOCALPROGS="/c/Users/$WHO/AppData/Local/Programs"
export PATH="$PATH:$LOCALPROGS/Ollama:$LOCALPROGS/DockerDesktop/resources/bin"

find_py312() {
  for c in "py -3.12" "python3.12" "$LOCALPROGS/Python/Python312/python.exe"; do
    if $c --version >/dev/null 2>&1; then echo "$c"; return 0; fi
  done
  return 1
}

PY="$(find_py312)" || {
  echo "ERROR: Python 3.12 not found. Install it from python.org and re-run." >&2
  exit 1
}
echo "using: $PY ($($PY --version 2>&1))"

if [ -d .venv ]; then
  echo "removing existing .venv"
  rm -rf .venv
fi

echo "creating .venv"
$PY -m venv .venv

VPY="$REPO/.venv/Scripts/python.exe"
[ -f "$VPY" ] || VPY="$REPO/.venv/bin/python"

echo "installing dependencies (this takes a few minutes)"
"$VPY" -m pip install --quiet --upgrade pip setuptools wheel
"$VPY" -m pip install --quiet -r services/api/requirements.txt
"$VPY" -m pip install --quiet --no-deps -e packages/rag_core
"$VPY" -m pip install --quiet pytest

echo
echo "installed:"
"$VPY" -c "import onnxruntime, tokenizers, fastapi, opensearchpy, numpy; \
print('  onnxruntime', onnxruntime.__version__); \
print('  tokenizers ', tokenizers.__version__); \
print('  fastapi    ', fastapi.__version__); \
print('  numpy      ', numpy.__version__)"

echo
echo "next:"
echo "  .venv/Scripts/python.exe ops/fetch_models.py"
echo "  docker compose up -d opensearch tika"
echo "  .venv/Scripts/python.exe ops/smoke_test.py --skip-llm"
