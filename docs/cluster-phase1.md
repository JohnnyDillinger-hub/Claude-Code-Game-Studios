# Cluster Phase 1

Phase 1 adds a narrow cluster/node layer for inventory collection and
deterministic single-GPU placement decisions when the local machine does not
have enough free VRAM.

## What Phase 1 Does

- Discovers local GPU inventory through `nvidia-smi`
- Represents local and remote nodes with typed inventory records
- Tracks `available_until` lease windows
- Maintains a registry of known nodes
- Schedules one agent onto one GPU at a time
- Prefers local execution first
- Falls back to remote nodes when local free VRAM is insufficient
- Treats multi-GPU nodes as multiple independent single-GPU placement targets
- Exposes a minimal node-agent HTTP service with `/health` and `/inventory`

## What Phase 1 Does Not Do

- Peer-to-peer discovery on the public internet
- NAT traversal or hole punching
- Auth, billing, or marketplace features
- Multi-GPU-per-agent placement
- Tensor parallel or distributed inference
- Remote model orchestration beyond a launcher stub
- Production-grade security hardening

## Main Commands

Probe the current machine:

```bash
python3 -m cluster.orchestrator.clusterctl probe-local
```

Show demo inventory:

```bash
python3 -m cluster.orchestrator.clusterctl show-inventory
```

Schedule a single agent:

```bash
python3 -m cluster.orchestrator.clusterctl schedule-agent \
  --agent-id qwen-worker \
  --profile qwen-coder-30b
```

Prune expired nodes from the demo inventory:

```bash
python3 -m cluster.orchestrator.clusterctl prune-expired
```

## Demo Data

The demo inventory lives under:

- `cluster/demo/local_node.json`
- `cluster/demo/remote_nodes.json`

It includes:

- one local node
- two active remote nodes
- one expired remote node
- one 4-GPU remote node
- mixed cached model sets
- mixed lease windows

## Scheduling Rules

Phase 1 scheduling is deterministic:

1. Prefer a local GPU if any local GPU has enough free VRAM
2. Otherwise consider active remote GPUs
3. Prefer nodes with the requested model already cached
4. Prefer the longest remaining lease
5. Break ties by highest free VRAM
6. Break final ties stably by `node_id`, then GPU index

Reservation is explicit. The scheduler itself does not silently mutate
inventory. Use `reserve_gpu_capacity(...)` when you want to model multiple
independent placements against the same node inventory in memory.
