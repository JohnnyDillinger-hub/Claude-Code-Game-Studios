# Cluster Architecture Phase 3

Phase 3 adds a backend-aware worker layer on top of the Phase 2 control plane.

## Main Component

### `cluster/orchestrator/remote_worker.py`

The remote worker is no longer a placeholder. It now handles three launch
shapes:

- `ollama-server`
- `vllm-server`
- `python-hf-probe`

## Runtime Model

### Dedicated per-GPU server

For `ollama` and `vllm`, Phase 3 launches a dedicated backend process per agent
and per GPU. This keeps the existing scheduler assumption intact:

- one agent
- one GPU
- one target node

The worker chooses deterministic per-GPU ports from the profile metadata and
stores a session record on disk.

### Session record

The worker writes:

- endpoint URL
- chosen port
- backend PID
- stdout/stderr log paths
- warmup result when applicable

This gives the control plane something concrete to inspect after launch without
claiming there is already a full distributed task fabric.

## Safety Behavior

Phase 3 still depends on:

- scheduler placement from Phase 1
- lease safety from Phase 2

The new worker adds:

- reuse of a live session for the same agent
- conflict detection when another agent already owns the same GPU slot
- backend health checks before a launch is considered successful

## What Phase 3 Still Does Not Do

- coordinate multi-GPU inference
- implement tensor parallel over more than one GPU
- build a persistent orchestrator daemon
- solve auth, billing, or marketplace concerns
- expose secure public-internet node discovery
