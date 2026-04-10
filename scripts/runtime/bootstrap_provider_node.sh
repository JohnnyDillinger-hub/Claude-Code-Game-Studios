#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/Claude-Code-Game-Studios}"
PROVIDER_BOOTSTRAP_MIN_FREE_DISK_GIB="${PROVIDER_BOOTSTRAP_MIN_FREE_DISK_GIB:-10}"
PROVIDER_BOOTSTRAP_RETRIES="${PROVIDER_BOOTSTRAP_RETRIES:-2}"
PROVIDER_BOOTSTRAP_TIMEOUT_SECONDS="${PROVIDER_BOOTSTRAP_TIMEOUT_SECONDS:-3600}"
PROVIDER_BOOTSTRAP_PAUSE_SECONDS="${PROVIDER_BOOTSTRAP_PAUSE_SECONDS:-5}"
export REPO_ROOT PROVIDER_BOOTSTRAP_MIN_FREE_DISK_GIB PROVIDER_BOOTSTRAP_RETRIES PROVIDER_BOOTSTRAP_TIMEOUT_SECONDS PROVIDER_BOOTSTRAP_PAUSE_SECONDS

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <runtime> [<runtime> ...]" >&2
  exit 2
fi

source "$REPO_ROOT/scripts/runtime/provider_bootstrap_common.sh"

provider_bootstrap_preflight_disk_space

for runtime in "$@"; do
  provider_bootstrap_install_runtime_with_retry "$runtime"
  provider_bootstrap_preflight_disk_space
  sleep 2
done
