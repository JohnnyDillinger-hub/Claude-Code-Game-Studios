#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/Claude-Code-Game-Studios}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv-vllm}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_VERSION="${VLLM_VERSION:-0.19.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

"$PYTHON_BIN" -m venv "$VENV_PATH"
"$VENV_PATH/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_PATH/bin/python" -m pip install --upgrade --index-url "$TORCH_INDEX_URL" torch torchvision torchaudio
"$VENV_PATH/bin/python" -m pip install --upgrade \
  "vllm==$VLLM_VERSION" \
  huggingface_hub \
  ninja
"$VENV_PATH/bin/python" - <<'PY'
import importlib.metadata as md
print("vllm", md.version("vllm"))
PY
