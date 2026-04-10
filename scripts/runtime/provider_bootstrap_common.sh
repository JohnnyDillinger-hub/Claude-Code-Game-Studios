#!/usr/bin/env bash
set -euo pipefail

provider_bootstrap_log() {
  printf '[provider-bootstrap] %s\n' "$*" >&2
}

provider_bootstrap_repo_root() {
  printf '%s\n' "${REPO_ROOT:-$HOME/Claude-Code-Game-Studios}"
}

provider_bootstrap_min_free_disk_gib() {
  printf '%s\n' "${PROVIDER_BOOTSTRAP_MIN_FREE_DISK_GIB:-10}"
}

provider_bootstrap_retry_count() {
  printf '%s\n' "${PROVIDER_BOOTSTRAP_RETRIES:-2}"
}

provider_bootstrap_timeout_seconds() {
  printf '%s\n' "${PROVIDER_BOOTSTRAP_TIMEOUT_SECONDS:-3600}"
}

provider_bootstrap_pause_seconds() {
  printf '%s\n' "${PROVIDER_BOOTSTRAP_PAUSE_SECONDS:-5}"
}

provider_bootstrap_free_gib_for_path() {
  local path="$1"
  df -Pk "$path" | awk 'NR==2 { printf "%d\n", int($4 / 1024 / 1024) }'
}

provider_bootstrap_cleanup_disk_pressure() {
  provider_bootstrap_log "running disk cleanup before runtime install"
  apt-get clean >/dev/null 2>&1 || true
  rm -rf /var/lib/apt/lists/* >/dev/null 2>&1 || true
  rm -rf /tmp/pip-* /tmp/tmp.* >/dev/null 2>&1 || true
  rm -rf /root/.cache/pip/http-v2 /root/.cache/pip/selfcheck >/dev/null 2>&1 || true
  rm -rf /root/.cache/pip/wheels >/dev/null 2>&1 || true
  if command -v journalctl >/dev/null 2>&1; then
    journalctl --vacuum-time=1d >/dev/null 2>&1 || true
  fi
}

provider_bootstrap_preflight_disk_space() {
  local repo_root
  repo_root="$(provider_bootstrap_repo_root)"
  local min_free_gib
  min_free_gib="$(provider_bootstrap_min_free_disk_gib)"
  local paths=("$repo_root" "/" "/var" "/tmp")
  local path
  local free_gib
  local seen=0
  for path in "${paths[@]}"; do
    [[ -e "$path" ]] || continue
    seen=1
    free_gib="$(provider_bootstrap_free_gib_for_path "$path")"
    provider_bootstrap_log "preflight disk check path=$path free=${free_gib}GiB threshold=${min_free_gib}GiB"
    if (( free_gib < min_free_gib )); then
      provider_bootstrap_cleanup_disk_pressure
      free_gib="$(provider_bootstrap_free_gib_for_path "$path")"
      provider_bootstrap_log "post-cleanup disk check path=$path free=${free_gib}GiB threshold=${min_free_gib}GiB"
      if (( free_gib < min_free_gib )); then
        provider_bootstrap_log "disk pressure remains after cleanup on $path"
        return 1
      fi
    fi
  done
  if [[ "$seen" -eq 0 ]]; then
    provider_bootstrap_log "no filesystem paths were available for disk preflight"
  fi
}

provider_bootstrap_install_runtime_once() {
  local runtime="$1"
  local repo_root
  repo_root="$(provider_bootstrap_repo_root)"
  cd "$repo_root"
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
      provider_bootstrap_log "runtime 'ollama' is tracked but has no installer script yet"
      ;;
    *)
      provider_bootstrap_log "unsupported runtime '$runtime'"
      return 2
      ;;
  esac
}

provider_bootstrap_install_runtime_with_retry() {
  local runtime="$1"
  local attempt=1
  local max_attempts
  max_attempts="$(provider_bootstrap_retry_count)"
  local timeout_seconds
  timeout_seconds="$(provider_bootstrap_timeout_seconds)"
  local pause_seconds
  pause_seconds="$(provider_bootstrap_pause_seconds)"

  while true; do
    provider_bootstrap_log "installing runtime '$runtime' attempt $attempt/$max_attempts"
    if timeout "$timeout_seconds" bash -lc '
      set -euo pipefail
      source "$REPO_ROOT/scripts/runtime/provider_bootstrap_common.sh"
      provider_bootstrap_install_runtime_once "$1"
    ' _ "$runtime"; then
      return 0
    fi
    if (( attempt >= max_attempts )); then
      provider_bootstrap_log "runtime '$runtime' install failed after $attempt attempt(s)"
      return 1
    fi
    provider_bootstrap_cleanup_disk_pressure
    sleep "$pause_seconds"
    attempt=$((attempt + 1))
  done
}
