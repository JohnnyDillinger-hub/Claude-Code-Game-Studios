#!/usr/bin/env bash

set -u -o pipefail

failures=0

pass() {
  printf '[pass] %s\n' "$1"
}

fail() {
  printf '[fail] %s\n' "$1" >&2
  failures=$((failures + 1))
}

check_command() {
  local name=$1

  if command -v "$name" >/dev/null 2>&1; then
    pass "found $name at $(command -v "$name")"
  else
    fail "missing required command: $name"
  fi
}

check_command git
check_command nvidia-smi
check_command ollama
check_command claude

if command -v ollama >/dev/null 2>&1; then
  if ollama_output="$(ollama list 2>&1)"; then
    if printf '%s\n' "$ollama_output" | awk 'NR > 1 { print $1 }' | grep -qx 'qwen3-coder:30b'; then
      pass "found Ollama model qwen3-coder:30b"
    else
      fail "Ollama is available, but qwen3-coder:30b is not installed"
    fi
  else
    fail "unable to query Ollama models; is Ollama running? Output: $ollama_output"
  fi
fi

if [ "$failures" -gt 0 ]; then
  printf '\nLocal stack smoke test failed with %d issue(s).\n' "$failures" >&2
  exit 1
fi

printf '\nLocal stack smoke test passed.\n'
