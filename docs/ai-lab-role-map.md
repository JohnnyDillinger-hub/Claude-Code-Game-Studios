# AI Lab Role Map

This fork keeps the existing `.claude` studio structure, but the active working
model is much smaller. Instead of treating the repo as a game studio, reinterpret
the most useful concepts as an AI/CUDA lab workflow.

## Role Mapping

| AI/CUDA Lab Role | Existing Studio Concepts to Reuse | What It Means in This Fork |
|------------------|-----------------------------------|----------------------------|
| Research lead | `technical-director`, `lead-programmer`, `architecture-decision` | Owns experiment direction, decides what is worth building, sets success criteria, and keeps the lab coherent |
| Model engineer | `ai-programmer`, `tools-programmer`, `lead-programmer` | Works on model wrappers, prompting utilities, eval harnesses, data plumbing, and local runtime integration |
| CUDA kernel engineer | `engine-programmer` | Owns CUDA/C++ extensions, custom kernels, fused ops, memory movement, launch configuration, and low-level correctness |
| Profiler / reviewer | `performance-analyst`, `code-review` | Measures performance, reviews diffs, checks correctness risk, and verifies that claimed speedups are backed by numbers |

## Suggested Workflow

1. Research lead frames the experiment and the acceptance criteria.
2. Model engineer wires up the runtime, scripts, or evaluation harness.
3. CUDA kernel engineer implements low-level accelerators only where justified.
4. Profiler / reviewer checks correctness first, then performance evidence.

## Practical Interpretation

- The existing game-specific roles remain in the repository as legacy template material.
- For this fork, treat the roles above as the active mental model.
- Use `/bootstrap-ai-lab` for stack setup and `/review-kernel-diff` for low-level review.
