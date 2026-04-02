---
name: bootstrap-ai-lab
description: "Minimal local AI/CUDA lab bootstrap. Inspects the machine, verifies the CUDA + Ollama + Claude Code stack, and outputs exact next commands to reach a usable local setup."
argument-hint: "[optional: quick|full]"
user-invocable: true
allowed-tools: Read, Glob, Grep, Bash
---

When this skill is invoked:

1. **Read the local project guidance first**:
   - `CLAUDE.md`
   - `docs/local-ollama-claude-setup.md`
   - `docs/ai-lab-role-map.md`

2. **Inspect the local stack**:
   - Check `git`
   - Check `nvidia-smi`
   - Check `ollama`
   - Check `claude`
   - If `ollama` exists, check whether `qwen3-coder:30b` appears in `ollama list`

3. **If present, prefer the repo smoke test**:
   - Run `bash scripts/smoke_test_local_stack.sh`
   - Capture the exact failures instead of paraphrasing them away

4. **Summarize the current state**:
   - What is already working
   - What is missing
   - Whether the machine is ready for local CUDA/model experiments

5. **Output exact next commands**:
   - Use the commands from `docs/local-ollama-claude-setup.md`
   - Keep the fix list short and ordered
   - If everything is ready, say so explicitly and suggest the next useful check

Use this output format:

```markdown
## Local Stack Status
- [working item]
- [missing item]

## Smoke Test
- [result]

## Exact Next Commands
[shell commands here]

## Notes
- [short note]
```

Rules:
- Keep this workflow practical and local-first.
- Do not redesign the repo.
- Prefer exact shell commands over generic advice.
