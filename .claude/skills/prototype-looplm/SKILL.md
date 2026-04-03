---
name: prototype-looplm
description: "Scopes a minimal LoopLM-style adapter experiment around Gemma PT, keeping the experiment honest about what it does and does not reproduce."
argument-hint: "[optional task family]"
user-invocable: true
allowed-tools: Read, Glob, Grep, Bash
---

When this skill is invoked:

1. **Read the local context**:
   - `CLAUDE.md`
   - `docs/ai-lab-role-map.md`
   - `research/looplm/README.md`

2. **Frame the experiment conservatively**:
   - explicitly call it a proxy or adapter experiment
   - do not call it a reproduction unless the evidence supports that claim

3. **Require a minimal ablation plan**:
   - baseline
   - loop depth variants, such as `K=1/2/4`
   - quality metric
   - latency overhead

4. **Return a reversible prototype plan**:

```markdown
## Hypothesis
[one paragraph]

## Minimal Experiment
- baseline
- adapter insertion point
- loop depths

## Metrics
- task quality
- latency
- memory

## Exact Next Commands
[commands]

## What This Is Not
- [non-goal or non-claim]
```

Rules:
- Keep the first pass small and honest.
- Measure latency overhead explicitly.
- Prefer adapter-style changes over deep architecture churn in the first prototype.
