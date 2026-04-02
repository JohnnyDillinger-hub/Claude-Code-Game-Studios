# Local Ollama + Claude Code Setup

This repository now targets a minimal local AI/CUDA workflow on Ubuntu 22.04.
Claude Code remains the main interactive environment in the repo, while Ollama
hosts the local model runtime.

## What This Setup Assumes

- Ubuntu 22.04
- NVIDIA driver already installed and visible through `nvidia-smi`
- Enough VRAM for `qwen3-coder:30b`
- Repository checked out locally

## 1. Install base tools

```bash
sudo apt-get update
sudo apt-get install -y git curl
```

## 2. Verify the NVIDIA stack

```bash
nvidia-smi
```

If this fails, stop and fix the driver/runtime first.

## 3. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## 4. Start Ollama with a 64k context window

In terminal 1:

```bash
OLLAMA_CONTEXT_LENGTH=64000 ollama serve
```

## 5. Pull the local model

In terminal 2:

```bash
ollama pull qwen3-coder:30b
ollama list
```

You should see `qwen3-coder:30b` in the model list.

## 6. Install Claude Code

```bash
curl -fsSL claude.ai/install.sh | bash
```

Verify:

```bash
claude --version
```

## 7. Run the repository smoke test

From the repository root:

```bash
bash scripts/smoke_test_local_stack.sh
```

## 8. Start working in the repo

From the repository root:

```bash
claude
```

Suggested first actions inside Claude Code:

- `/bootstrap-ai-lab`
- `/review-kernel-diff [path]`

## Full Command Sequence

```bash
cd /path/to/Claude-Code-Game-Studios
sudo apt-get update
sudo apt-get install -y git curl
nvidia-smi
curl -fsSL https://ollama.com/install.sh | sh
```

Then in terminal 1:

```bash
cd /path/to/Claude-Code-Game-Studios
OLLAMA_CONTEXT_LENGTH=64000 ollama serve
```

Then in terminal 2:

```bash
cd /path/to/Claude-Code-Game-Studios
ollama pull qwen3-coder:30b
curl -fsSL claude.ai/install.sh | bash
bash scripts/smoke_test_local_stack.sh
claude
```
