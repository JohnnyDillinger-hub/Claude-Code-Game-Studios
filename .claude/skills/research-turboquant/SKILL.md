---
name: research-turboquant
description: "Sets up or reviews a small TurboQuant-style research pass around Gemma PT: establish a BF16 baseline first, then compare compressed or quantized variants without overstating results."
argument-hint: "[optional experiment note]"
user-invocable: true
allowed-tools: Read, Glob, Grep, Bash
---

When this skill is invoked:

1. **Read the local context**:
   - `CLAUDE.md`
   - `docs/local-ollama-claude-setup.md`
   - `research/turboquant/README.md`

2. **Verify the PT research backend exists**:
   - check `scripts/gemma_pt/load_gemma_pt.py`
   - check `scripts/gemma_pt/infer_gemma_pt.py`
   - check whether `.venv-gemma` is mentioned or already present

3. **Establish baseline first**:
   - confirm the goal is baseline BF16-style inference on `google/gemma-3-12b-pt`
   - do not discuss quantized wins before a baseline exists

4. **If evaluating a quantized idea**, require:
   - a named baseline
   - a named compressed variant
   - task quality metric
   - latency metric
   - memory metric

5. **Return a concise experiment brief**:

```markdown
## Goal
[one paragraph]

## Baseline
- [what to run first]

## Variant
- [compressed or quantized idea]

## Required Metrics
- quality
- latency
- memory

## Exact Next Commands
[commands]

## Risks
- [risk]
```

Rules:
- Baseline before compression.
- No claims of paper reproduction from a local prototype.
- Prefer one small measurable experiment over a broad speculative plan.
