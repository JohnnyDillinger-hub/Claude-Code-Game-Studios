# Claude Code AI/CUDA Lab

This fork uses the existing `.claude/**` structure as its runtime, but the
active domain is now a local AI/CUDA lab rather than a game studio template.

## Local Stack

- **OS**: Ubuntu 22.04
- **Version Control**: Git
- **GPU Runtime**: NVIDIA CUDA via `nvidia-smi`
- **Local Model Runtime**: Ollama
- **Primary Local Model**: `qwen3-coder:30b`
- **Interactive Agent**: Claude Code
- **Working Languages**: Python, C++, CUDA, Bash

## Model Topology

- **Primary coding runtime**: keep `qwen3-coder:30b` as the main local coding backend.
- **Optional secondary interactive reviewer**: `gemma3:12b` can be used for read-only review or analysis when running a local Ollama-backed workflow.
- **Offline research backend**: keep `google/gemma-3-12b-pt` in a separate Hugging Face / Transformers runtime for controlled inference, long-context evaluation, quantization experiments, and adapter-based research.

The Gemma PT backend is a second contour, not a replacement for the main coding path.

## First Session

@docs/local-ollama-claude-setup.md
@docs/ai-lab-role-map.md

- Run `/bootstrap-ai-lab` to inspect the local stack and identify setup gaps.
- Use `/review-kernel-diff` for CUDA kernels, fused ops, extensions, and other
  low-level performance-sensitive diffs.
- Use `/research-turboquant`, `/benchmark-long-context`, and `/prototype-looplm`
  for the Gemma-backed research contour.

## Working Principles

- Preserve `.claude/**`; prefer additive changes in `docs/`, `scripts/`, and lab code.
- Keep changes small, explicit, and reversible.
- Validate the local stack before heavier work with
  `bash scripts/smoke_test_local_stack.sh`.
- For CUDA and performance-sensitive changes, report both correctness risk and
  performance risk.

## Reference Material

@.claude/docs/coding-standards.md
@.claude/docs/context-management.md
@.claude/docs/coordination-rules.md

## Legacy Note

The repository still contains game-studio agents, docs, and workflows. Treat
them as reusable structure and examples, not as the active domain model for
this fork.
