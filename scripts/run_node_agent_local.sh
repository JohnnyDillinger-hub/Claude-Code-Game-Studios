#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m cluster.node_agent.daemon \
  --node-id local-demo-agent \
  --host 127.0.0.1 \
  --bind-host 127.0.0.1 \
  --port 8787 \
  --lease-duration-seconds 1800 \
  --cached-model qwen3-coder:30b \
  --label site=local \
  --label role=demo-node \
  --trust-tier trusted \
  --network-tier lan
