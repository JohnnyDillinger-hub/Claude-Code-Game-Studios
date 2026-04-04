---
name: cluster-status
description: "Inspect the current Phase 1 node inventory, lease windows, GPU availability, and cached-model state."
argument-hint: "[optional: local-file remote-file]"
user-invocable: true
allowed-tools: Read, Glob, Grep, Bash
---

When this skill is invoked:

1. Read:
   - `CLAUDE.md`
   - `docs/cluster-phase1.md`
   - `docs/cluster-architecture-phase1.md`

2. Inspect current inventory through the repo tools first:
   - `python3 -m cluster.orchestrator.clusterctl show-inventory`
   - `python3 -m cluster.orchestrator.clusterctl load-registry --state-file ...` when a persistent registry exists
   - if a live node-agent is running, you may also inspect `/health` and `/inventory`

3. Report:
   - active vs expired nodes
   - per-node GPU counts
   - per-GPU free VRAM
   - cached models when present
   - obvious capacity gaps for local-first execution

Use this output shape:

```markdown
## Cluster Status
- [key fact]

## Nodes
- [node summary]

## Capacity Notes
- [placement or lease note]

## Exact Next Commands
[commands]
```

Rules:
- Keep the answer operational.
- Do not invent remote orchestration that Phase 1 does not implement.
- In Phase 2, call out heartbeat freshness and persistent registry state when they matter.
- Call out expired nodes explicitly instead of quietly ignoring them.
