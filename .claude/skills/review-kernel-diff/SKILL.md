---
name: review-kernel-diff
description: "Reviews CUDA, Triton, fused op, and low-level performance diffs with a findings-first workflow focused on correctness, synchronization, memory movement, and measurable performance risk."
argument-hint: "[path-to-file-or-directory]"
user-invocable: true
allowed-tools: Read, Glob, Grep, Bash
---

When this skill is invoked:

1. **Read the target diff or files in full**.

2. **Read project guidance**:
   - `CLAUDE.md`
   - relevant benchmark or profiling docs if they exist

3. **Check correctness risks first**:
   - Host/device boundary mistakes
   - Shape, stride, dtype, and indexing errors
   - Out-of-bounds access risks
   - Missing synchronization or incorrect stream assumptions
   - Invalid assumptions about determinism or numeric stability
   - Missing error handling after launches or runtime calls

4. **Check performance risks second**:
   - Excess host-device copies
   - Uncoalesced memory access patterns
   - Shared-memory or register pressure concerns
   - Launch configuration issues
   - Occupancy-limiting choices
   - Redundant synchronization or serialization points

5. **Check validation coverage**:
   - Unit or property tests for correctness
   - Benchmark coverage before/after
   - Regression checks for edge shapes and boundary conditions

6. **Output findings first** in this format:

```markdown
## Findings
1. [severity] [file:line] [issue]

## Correctness Risks
- [risk]

## Performance Risks
- [risk]

## Validation Gaps
- [missing test or benchmark]

## Verdict
[APPROVED / CHANGES REQUIRED / NEEDS MEASUREMENT]
```

Rules:
- Findings come before summary.
- Prefer concrete file and line references.
- Do not say "optimize" without naming what to measure or change.
- If no findings exist, say so explicitly and mention residual measurement gaps.
