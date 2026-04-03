---
name: benchmark-long-context
description: "Plans or reviews long-context evaluations across the main coding model and the Gemma research contour, with emphasis on comparable settings, prompt formats, and result metadata."
argument-hint: "[optional benchmark scope]"
user-invocable: true
allowed-tools: Read, Glob, Grep, Bash
---

When this skill is invoked:

1. **Read the local context**:
   - `CLAUDE.md`
   - `docs/ai-lab-role-map.md`
   - `docs/local-ollama-claude-setup.md`
   - `research/long_context/README.md`

2. **Identify the comparison set**:
   - main coding model path
   - Gemma PT or Gemma interactive comparison path
   - prompt format
   - context lengths

3. **Refuse sloppy comparisons**:
   - if prompt formats differ, say so
   - if context sizes differ, say so
   - if latency or memory capture is missing, say so
   - if the benchmark is synthetic vs realistic, say so

4. **Produce a compact benchmark plan**:

```markdown
## Models
- [model]

## Context Lengths
- [lengths]

## Benchmarks
- [benchmark or probe]

## Required Metadata
- model
- prompt format
- context length
- latency
- peak memory
- score

## Exact Next Commands
[commands]

## Comparability Risks
- [risk]
```

Rules:
- Prefer a tiny smoke benchmark before a full matrix.
- Comparability is mandatory.
- Keep the first pass small enough to finish on one local machine.
