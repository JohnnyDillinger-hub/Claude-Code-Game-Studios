# Cluster Phase 3 Runtime Launch

Phase 3 turns the Phase 2 SSH launch path into a real backend-aware runtime
launcher.

## What It Does

- keeps Phase 1 and Phase 2 inventory, lease, and scheduling behavior intact
- launches one real runtime target per agent
- keeps the Phase 1 rule of one agent per one GPU
- writes a per-agent session record under
  `production/session-state/remote-workers/`

## Backends

### Ollama

For `ollama` profiles, the remote worker now:

1. chooses a deterministic per-GPU port
2. starts a dedicated `ollama serve` process bound to that single GPU
3. waits for `/api/tags`
4. warms the requested model with a short generate call
5. writes the endpoint, logs, and PID into the session file

This avoids relying on a shared system Ollama daemon when GPU affinity matters.

### vLLM

For `vllm` profiles, the remote worker now:

1. chooses a deterministic per-GPU port
2. starts `python -m vllm.entrypoints.openai.api_server`
3. waits for `/health` or `/v1/models`
4. writes the endpoint, logs, and PID into the session file

Phase 3 still fixes `tensor_parallel_size=1`. Multi-GPU-per-agent stays out of
scope.

### Python / Hugging Face

For `python-hf` profiles, the worker runs a one-shot probe on the target GPU and
records the output. This keeps the research backend usable without pretending it
is already a long-lived clustered service.

## Session Files

Each launched worker writes JSON such as:

```json
{
  "status": "launched",
  "agent_id": "qwen-worker-a",
  "node_id": "cluster-pro6000",
  "backend": "ollama",
  "model": "qwen3-coder:30b",
  "gpu_index": 0,
  "listen_port": 17434,
  "endpoint_url": "http://127.0.0.1:17434",
  "server_pid": 12345,
  "single_gpu_only": true
}
```

## Main Command

Dry-run:

```bash
python3 -m cluster.orchestrator.clusterctl launch-agent \
  --state-file production/session-state/cluster-registry.json \
  --local-node-id cluster-5060ti \
  --agent-id qwen-remote-a \
  --profile qwen-coder-30b \
  --dry-run
```

Execute:

```bash
python3 -m cluster.orchestrator.clusterctl launch-agent \
  --state-file production/session-state/cluster-registry.json \
  --local-node-id cluster-5060ti \
  --agent-id qwen-remote-a \
  --profile qwen-coder-30b \
  --ssh-user root
```

## Still Out of Scope

- multi-GPU-per-agent
- tensor parallel above `1`
- distributed KV cache
- production auth, billing, and hardened remote trust boundaries
- public-internet P2P discovery
