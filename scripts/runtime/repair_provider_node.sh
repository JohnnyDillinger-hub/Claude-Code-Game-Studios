#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/Claude-Code-Game-Studios}"
PROVIDER_BOOTSTRAP_MIN_FREE_DISK_GIB="${PROVIDER_BOOTSTRAP_MIN_FREE_DISK_GIB:-10}"
export REPO_ROOT PROVIDER_BOOTSTRAP_MIN_FREE_DISK_GIB

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <runtime> [<runtime> ...]" >&2
  exit 2
fi

source "$REPO_ROOT/scripts/runtime/provider_bootstrap_common.sh"

provider_bootstrap_preflight_disk_space

for runtime in "$@"; do
  case "$runtime" in
    deepspeed)
      PYTHON_BIN="${REPO_ROOT}/.venv-deepspeed/bin/python"
      EXECUTABLE_PATH="${REPO_ROOT}/.venv-deepspeed/bin/deepspeed"
      if [[ ! -x "$PYTHON_BIN" ]]; then
        echo "provider repair: missing DeepSpeed python executable at $PYTHON_BIN" >&2
        exit 2
      fi
      if [[ ! -e "$EXECUTABLE_PATH" ]]; then
        echo "provider repair: missing DeepSpeed entrypoint at $EXECUTABLE_PATH" >&2
        exit 2
      fi
      "$PYTHON_BIN" -c '
import json
import sys
from pathlib import Path

from cluster.orchestrator.runtime_adapters import repair_packaged_runtime_nvcc_link

executable_path = sys.argv[1]
result = repair_packaged_runtime_nvcc_link(executable_path, allow_version_shim=True)
runtime_bin = result.get("runtime_bin")
nvcc_path = result.get("nvcc_path")
target = Path(runtime_bin) / "nvcc" if runtime_bin else None
print(json.dumps(result, sort_keys=True))
if target is None or not target.exists() or nvcc_path is None:
    raise SystemExit(2)
' "$EXECUTABLE_PATH"
      provider_bootstrap_preflight_disk_space
      ;;
    *)
      echo "provider repair: unsupported runtime '$runtime'" >&2
      exit 2
      ;;
  esac
done
