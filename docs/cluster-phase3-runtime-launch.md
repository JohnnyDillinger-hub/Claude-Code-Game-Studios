# Cluster Phase 3 Runtime Launch

Phase 3 turns the Phase 2 SSH launch path into a real backend-aware runtime
launcher.

## What It Does

- keeps Phase 1 and Phase 2 inventory, lease, and scheduling behavior intact
- launches one real runtime target per agent
- keeps the Phase 1 rule of one agent per one placement target
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

1. chooses a deterministic port for the selected GPU group
2. starts `python -m vllm.entrypoints.openai.api_server`
3. waits for `/health` or `/v1/models`
4. writes the endpoint, logs, and PID into the session file

Single-node tensor parallel is now supported for dedicated `vllm` profiles such
as `qwen-coder-30b-vllm-tp2` and `qwen-coder-30b-vllm-tp4`, where one agent can
reserve multiple GPUs on the same node.

### SGLang

For `sglang` profiles, the remote worker now:

1. chooses a deterministic port for the selected GPU group
2. starts `python -m sglang.launch_server`
3. waits for `/health` or `/v1/models`
4. writes the endpoint, logs, and PID into the session file

This uses the same single-node multi-GPU reservation model as `vllm`, with
dedicated profiles such as `qwen-coder-30b-sglang-tp2` and
`qwen-coder-30b-sglang-tp4`.

`launch-agent` also supports request-level CUDA graph overrides for SGLang:

- `--cuda-graph-mode profile-default`
- `--cuda-graph-mode enabled`
- `--cuda-graph-mode disabled`
- `--cuda-graph-max-bs N`

This lets the client keep the profile default for a known-good node, or
explicitly force the runtime to start with or without CUDA graph capture for a
particular launch.

`launch-agent` now also emits a top-level `deployment` object in its JSON
response. This gives the future client API a stable, serializable contract for
the requested profile, placement constraints, and launch preferences.

### TensorRT-LLM

For `tensorrt-llm` profiles, the remote worker now:

1. chooses a deterministic port for the selected GPU group
2. starts `trtllm-serve serve`
3. waits for `/health` or `/v1/models`
4. writes the endpoint, logs, and PID into the session file

The current first-class profiles use the `TensorRT-LLM` OpenAI-compatible
server in `pytorch` backend mode so they can launch directly from a model path
without requiring a prebuilt engine. Single-node multi-GPU reservations are now
supported for profiles such as `qwen-coder-30b-trtllm-tp2` and
`qwen-coder-30b-trtllm-tp4`. The runtime adapter also prepends packaged CUDA,
TensorRT, and Torch library directories from the target virtualenv into
`LD_LIBRARY_PATH` so `trtllm-serve` can resolve shared objects such as
`libcublasLt.so.13` and `libnvinfer.so.10` on freshly provisioned nodes.

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
  "node_id": "cluster-5090x2",
  "backend": "vllm",
  "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
  "gpu_index": 0,
  "gpu_indices": [0, 1],
  "tensor_parallel_size": 2,
  "listen_port": 18020,
  "endpoint_url": "http://127.0.0.1:18020",
  "server_pid": 12345,
  "single_gpu_only": false
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
  --profile qwen-coder-30b-vllm-tp2 \
  --probe-remote-sessions
```

Probe and register a live node over SSH:

```bash
python3 -m cluster.orchestrator.clusterctl probe-remote-node \
  --node-id cluster-5090x4-live \
  --host 209.50.14.20 \
  --ssh-user root \
  --ssh-port 39693 \
  --repo-root '$HOME/Claude-Code-Game-Studios' \
  --state-file production/session-state/cluster-registry-live.json
```

Inspect remote session occupancy:

```bash
python3 -m cluster.orchestrator.clusterctl list-remote-sessions \
  --state-file production/session-state/cluster-registry.json
```

If a previous dedicated worker is already holding a GPU, the optional
`--probe-remote-sessions` check excludes that GPU before scheduling so the
control plane fails early with an occupancy reason instead of discovering the
conflict only after the SSH launch attempt.

## Still Out of Scope

- multi-node tensor parallel or pipeline parallel
- distributed KV cache
- production auth, billing, and hardened remote trust boundaries
- public-internet P2P discovery
