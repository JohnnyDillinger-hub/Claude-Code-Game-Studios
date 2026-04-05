# Cluster Phase 2

For the dedicated backend runtime launch added after Phase 2, see
`docs/cluster-phase3-runtime-launch.md`.

Phase 2 extends the Phase 1 inventory and scheduler layer with:

- heartbeat-based node updates
- persistent registry state on disk
- runtime model profiles
- a real SSH-based dry-run launch path
- lease-aware launch safety checks

Phase 2 still stays intentionally narrow:

- one agent maps to one GPU
- one launch target is one node
- no tensor parallel
- no multi-GPU-per-agent
- no public-internet P2P or billing layer

## What Is New

### Heartbeats

Node agents can now emit heartbeat updates that contain:

- current node inventory
- current `available_until`
- heartbeat interval

For local demos, heartbeats can update a persistent registry file directly.

### Persistent Registry

Registry state can now be saved and loaded from disk through
`cluster/orchestrator/state_store.py`.

The format is plain JSON and uses atomic replace on write.

### Runtime Profiles

Profiles live in:

- `cluster/orchestrator/model_profiles.yaml`

Each profile describes:

- model name
- required free VRAM
- runtime class
- preferred backend
- launch metadata

### Launch Path

`launch-agent` can now:

1. compute a placement
2. verify that the lease is still safe
3. build a deterministic local or SSH launch command
4. optionally execute it

Phase 2 established the lease-aware SSH launch surface. The dedicated backend
runtime launched by that path is documented separately in
`docs/cluster-phase3-runtime-launch.md`.

## Main Commands

Heartbeat once into a persistent registry:

```bash
python3 -m cluster.orchestrator.clusterctl heartbeat-once \
  --inventory-url http://127.0.0.1:8787/inventory \
  --state-file production/session-state/cluster-registry.json
```

Save demo inventory to persistent state:

```bash
python3 -m cluster.orchestrator.clusterctl save-registry \
  --state-file production/session-state/cluster-registry.json
```

Load saved registry:

```bash
python3 -m cluster.orchestrator.clusterctl load-registry \
  --state-file production/session-state/cluster-registry.json
```

Dry-run a remote launch:

```bash
python3 -m cluster.orchestrator.clusterctl launch-agent \
  --state-file production/session-state/cluster-registry.json \
  --local-node-id local-lab \
  --agent-id qwen-remote \
  --profile qwen-coder-30b \
  --dry-run
```

## State File Format

The persistent state store writes JSON in this shape:

```json
{
  "version": 1,
  "records": [
    {
      "node": { "...": "NodeInventory" },
      "last_heartbeat_at": "2026-04-04T00:00:00Z",
      "heartbeat_interval_seconds": 30,
      "source": "heartbeat"
    }
  ]
}
```

## Safety Rule

Phase 2 launch is blocked if:

- the chosen placement is already expired
- the chosen placement has less than the configured minimum lease window left

The default safety threshold is `30` seconds.
