---
name: launch-remote-agent
description: "Compute a Phase 2 placement and build or execute a lease-aware single-GPU remote launch command."
argument-hint: "[agent-id] [profile]"
user-invocable: true
allowed-tools: Read, Glob, Grep, Bash
---

When this skill is invoked:

1. Read:
   - `CLAUDE.md`
   - `docs/cluster-phase2.md`
   - `docs/cluster-architecture-phase2.md`

2. Use the repo CLI instead of inventing launch logic:
   - `python3 -m cluster.orchestrator.clusterctl launch-agent ...`

3. Prefer `--dry-run` first unless the user explicitly wants execution.

4. Report:
   - selected placement
   - whether launch is local or remote
   - exact command
   - whether lease safety passed or blocked the launch

Use this output shape:

```markdown
## Request
- [agent + profile]

## Placement
- [node + gpu + reason]

## Launch
- [dry-run command or execution result]

## Safety
- [lease / cache / stale-node note]
```

Rules:
- Phase 2 is still single-agent single-GPU only.
- Do not claim distributed or multi-GPU launch support.
- If launch is blocked by lease expiry or stale state, say so directly.
