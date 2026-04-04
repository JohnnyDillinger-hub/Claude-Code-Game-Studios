# Cluster Architecture Phase 1

Phase 1 is intentionally small. It adds just enough structure to reason about
GPU nodes and place single-GPU agents without pretending the project already has
a full mesh runtime.

## Layers

### `cluster/models.py`

Typed records for:

- `LeaseInfo`
- `GPUInventory`
- `NodeInventory`
- `AgentRequest`
- `PlacementDecision`

These models are the contract shared by the probe, node-agent, registry, demo
data, and scheduler.

### `cluster/node_agent/`

- `probe_gpu.py`
  - wraps `nvidia-smi`
  - returns structured per-GPU inventory
- `daemon.py`
  - minimal HTTP inventory service
  - exposes `/health` and `/inventory`
  - supports CLI args or a JSON config file

### `cluster/orchestrator/`

- `registry.py`
  - stores node inventory snapshots
  - prunes expired nodes
  - returns deterministic active node lists
- `scheduler.py`
  - deterministic single-GPU placement policy
  - local-first
  - cache-aware
  - lease-aware
  - explicit reservation helper for repeated scheduling calls
- `launcher.py`
  - Phase 1 stub only
  - builds the command that would be run locally or remotely
- `clusterctl.py`
  - operator CLI for probing, viewing inventory, scheduling, and pruning

## Placement Model

Phase 1 placement is exactly:

- one agent
- one GPU
- one node

The scheduler evaluates each GPU independently. A 4-GPU node can therefore host
up to 4 independent Phase 1 agents as long as each placement is tracked through
an explicit reservation step or a fresh inventory snapshot.

## Lease Handling

`available_until` is part of every node inventory. Expired nodes are ignored by
the scheduler and can be pruned out of the registry.

This is the only lease mechanic in Phase 1. There is no renewal protocol,
heartbeat protocol, or distributed consensus layer yet.

## Integration Boundary

This layer is infra-only and additive:

- it does not replace `.claude/**`
- it does not refactor the current AI/CUDA lab layout
- it does not implement remote Ollama/vLLM execution

The intended near-term workflow is:

1. collect local + remote inventory
2. compute a deterministic placement
3. hand the result to a human or a future orchestration layer

That keeps the first pass testable and reversible.
