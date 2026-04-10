#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/Claude-Code-Game-Studios}"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <runtime> [<runtime> ...]" >&2
  exit 2
fi

cd "$REPO_ROOT"

for runtime in "$@"; do
  case "$runtime" in
    vllm)
      bash scripts/runtime/install_vllm.sh
      ;;
    sglang)
      bash scripts/runtime/install_sglang.sh
      ;;
    deepspeed)
      bash scripts/runtime/install_deepspeed.sh
      ;;
    ollama)
      echo "provider bootstrap: runtime '$runtime' is tracked but has no installer script yet" >&2
      ;;
    *)
      echo "provider bootstrap: unsupported runtime '$runtime'" >&2
      exit 2
      ;;
  esac
done
