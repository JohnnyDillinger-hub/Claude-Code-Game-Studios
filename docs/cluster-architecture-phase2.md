# Cluster Architecture Phase 2

Phase 2 keeps the Phase 1 scheduler intact and adds two new concerns:

1. freshness of node inventory
2. persistence of cluster state

## New Components

### `cluster/node_agent/heartbeat.py`

Defines:

- `HeartbeatPayload`
- inventory fetch helpers
- HTTP heartbeat POST helper
- file-backed heartbeat helper for local demos
- periodic heartbeat service for node agents

This is still intentionally lightweight. There is no full orchestrator daemon
yet. For local demos and tests, heartbeats can land in a shared registry state
file.

### `cluster/orchestrator/state_store.py`

Adds persistent registry save/load with atomic replacement on write.

This is the boundary between in-memory scheduling logic and a durable cluster
snapshot.

### `cluster/orchestrator/model_profiles.py`

Loads model/runtime profiles from `model_profiles.yaml`.

This removes hard-coded VRAM assumptions from the CLI layer and prepares the
launcher for backend-aware command building.

### `cluster/orchestrator/launcher.py`

Phase 2 launcher behavior:

- validates lease safety before launch
- builds deterministic commands
- supports:
  - local launch
  - SSH dry-run
  - SSH execution

The launched worker remains a placeholder runtime entrypoint rather than a full
remote orchestrator. That keeps Phase 2 honest about scope.

## Phase 1 vs Phase 2

### Phase 1

- static inventory
- in-memory registry
- deterministic placement
- launcher stub

### Phase 2

- heartbeat updates
- persistent registry snapshot
- runtime profiles
- lease-aware launch validation
- real SSH command path

## Still Out of Scope

- P2P node discovery on the public internet
- marketplace or billing
- multi-GPU-per-agent
- distributed inference
- production hardening of remote execution
- secure credential lifecycle management

Those belong to later phases, not this one.
