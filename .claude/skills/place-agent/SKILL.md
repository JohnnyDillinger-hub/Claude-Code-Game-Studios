---
name: place-agent
description: "Request a deterministic Phase 1 placement for a single-GPU agent based on model and VRAM requirements."
argument-hint: "[agent-id] [profile-or-vram]"
user-invocable: true
allowed-tools: Read, Glob, Grep, Bash
---

When this skill is invoked:

1. Read:
   - `CLAUDE.md`
   - `docs/cluster-phase1.md`
   - `docs/cluster-architecture-phase1.md`

2. Use the repo scheduler instead of ad-hoc reasoning:
   - `python3 -m cluster.orchestrator.clusterctl schedule-agent ...`

3. Accept either:
   - a known profile such as `qwen-coder-30b`, `gemma3-12b`, or `gemma-pt-12b`
   - or an explicit `--vram-required-mib` value

4. Report:
   - whether placement is local or remote
   - selected node and GPU
   - whether the model is already cached
   - the reason the scheduler chose that target

Use this output shape:

```markdown
## Request
- [agent + model + VRAM]

## Placement Decision
- [decision summary]

## Constraints
- [expired node, local insufficiency, cache note, or lease note]

## Exact Next Commands
[commands]
```

Rules:
- Phase 1 is one agent to one GPU only.
- Do not claim tensor parallel, sharding, or remote execution support.
- If capacity is insufficient everywhere, say so clearly and show the exact rejected request.
