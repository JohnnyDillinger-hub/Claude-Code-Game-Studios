#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/Claude-Code-Game-Studios}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv-deepspeed}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEEPSPEED_VERSION="${DEEPSPEED_VERSION:-0.18.9}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.57.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
CUDA_NVCC_VERSION="${CUDA_NVCC_VERSION:-12.8.93}"

rm -rf "$VENV_PATH"
"$PYTHON_BIN" -m venv "$VENV_PATH"
PIP_NO_CACHE_DIR=1 "$VENV_PATH/bin/python" -m pip install --upgrade pip setuptools wheel
PIP_NO_CACHE_DIR=1 "$VENV_PATH/bin/python" -m pip install --upgrade --index-url "$TORCH_INDEX_URL" torch torchvision torchaudio
PIP_NO_CACHE_DIR=1 "$VENV_PATH/bin/python" -m pip install --upgrade \
  "nvidia-cuda-nvcc-cu12==$CUDA_NVCC_VERSION" \
  "deepspeed==$DEEPSPEED_VERSION" \
  "transformers==$TRANSFORMERS_VERSION" \
  accelerate \
  sentencepiece \
  safetensors \
  ninja
"$VENV_PATH/bin/python" - <<'PY'
import site
import shutil
from pathlib import Path

site_packages = None
for candidate in site.getsitepackages():
    if "site-packages" in candidate:
        site_packages = Path(candidate)
        break

if site_packages is None:
    raise SystemExit("Could not resolve site-packages for DeepSpeed virtualenv")

nvcc_bin = site_packages / "nvidia" / "cuda_nvcc" / "bin" / "nvcc"
runtime_bin = site_packages / "nvidia" / "cuda_runtime" / "bin"
runtime_bin.mkdir(parents=True, exist_ok=True)
target_link = runtime_bin / "nvcc"
if nvcc_bin.exists():
    if target_link.exists() or target_link.is_symlink():
        target_link.unlink()
    target_link.symlink_to(nvcc_bin)
    print("linked", target_link, "->", nvcc_bin)
else:
    system_nvcc = shutil.which("nvcc")
    if system_nvcc is None:
        for candidate in [Path("/usr/local/cuda/bin/nvcc"), *sorted(Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse=True)]:
            if candidate.exists():
                system_nvcc = str(candidate)
                break
    if system_nvcc is not None:
        if target_link.exists() or target_link.is_symlink():
            target_link.unlink()
        target_link.symlink_to(system_nvcc)
        print("linked", target_link, "->", system_nvcc)
    else:
        print(f"warning: no packaged or system nvcc found at {nvcc_bin}")
PY
"$VENV_PATH/bin/python" - <<'PY'
import importlib.metadata as md
print("deepspeed", md.version("deepspeed"))
print("transformers", md.version("transformers"))
PY
