#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Demo inventory"
python3 -m cluster.orchestrator.clusterctl show-inventory

echo
echo "[2/4] Local-fit schedule (gemma3:12b)"
python3 -m cluster.orchestrator.clusterctl schedule-agent \
  --agent-id gemma-local-demo \
  --profile gemma3-12b

echo
echo "[3/4] Spillover schedule (qwen3-coder:30b)"
python3 -m cluster.orchestrator.clusterctl schedule-agent \
  --agent-id qwen-remote-demo \
  --profile qwen-coder-30b

echo
echo "[4/4] Sequential placements on the 4-GPU remote node"
python3 - <<'PY'
from cluster.demo.load_demo import load_demo_local_node, load_demo_remote_nodes
from cluster.models import AgentRequest
from cluster.orchestrator.scheduler import reserve_gpu_capacity, schedule_agent

local = reserve_gpu_capacity(load_demo_local_node(), gpu_index=0, amount_mib=36000)
remote_nodes = load_demo_remote_nodes()
four_gpu = next(node for node in remote_nodes if node.node_id == "remote-4gpu-a")
other_nodes = [node for node in remote_nodes if node.node_id != "remote-4gpu-a"]
request = AgentRequest(
    agent_id="multi-agent-demo",
    model_id="google/gemma-3-12b-pt",
    required_vram_mib=10000,
)

for turn in range(1, 5):
    decision = schedule_agent(local, [four_gpu, *other_nodes], request)
    print(
        f"placement {turn}: node={decision.node_id} gpu={decision.gpu_index} "
        f"source={decision.source} free_vram={decision.available_vram_mib}"
    )
    four_gpu = reserve_gpu_capacity(
        four_gpu,
        gpu_index=decision.gpu_index,
        amount_mib=request.required_vram_mib,
    )
PY
