#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STATE_FILE="$ROOT_DIR/production/session-state/cluster-phase2-demo.json"
PORT=8797

rm -f "$STATE_FILE"

echo "[1/6] Save demo inventory into persistent registry"
python3 -m cluster.orchestrator.clusterctl save-registry \
  --state-file "$STATE_FILE"

echo
echo "[2/6] Start a local node-agent on port $PORT"
python3 -m cluster.node_agent.daemon \
  --node-id local-demo-agent \
  --host 127.0.0.1 \
  --bind-host 127.0.0.1 \
  --port "$PORT" \
  --lease-duration-seconds 300 \
  --cached-model gemma3:12b \
  --label site=local \
  --label role=demo-node \
  > /tmp/cluster-phase2-node-agent.log 2>&1 &
NODE_AGENT_PID=$!
trap 'kill $NODE_AGENT_PID >/dev/null 2>&1 || true' EXIT
sleep 1

echo
echo "[3/6] Simulate one heartbeat into the persistent registry"
python3 -m cluster.orchestrator.clusterctl heartbeat-once \
  --inventory-url "http://127.0.0.1:${PORT}/inventory" \
  --state-file "$STATE_FILE"

echo
echo "[4/6] Show registry state from disk"
python3 -m cluster.orchestrator.clusterctl load-registry \
  --state-file "$STATE_FILE"

echo
echo "[5/6] Dry-run a remote launch"
python3 -m cluster.orchestrator.clusterctl launch-agent \
  --state-file "$STATE_FILE" \
  --local-node-id local-lab \
  --agent-id qwen-remote-demo \
  --profile qwen-coder-30b \
  --dry-run

echo
echo "[6/6] Reload registry via summary view"
python3 -m cluster.orchestrator.clusterctl show-inventory \
  --state-file "$STATE_FILE"
