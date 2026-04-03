# TurboQuant Research Track

This directory is for small, local TurboQuant-style experiments around the
Gemma PT research backend.

## Current Scope

- establish a reproducible baseline for `google/gemma-3-12b-pt`
- compare one compressed or quantized variant at a time
- track quality, latency, and memory together

## Expected Artifacts

- notes and experiment briefs in markdown
- machine-readable results in `research/turboquant/runs/`

## Current Bootstrap

The initial bootstrap uses:

- `scripts/gemma_pt/load_gemma_pt.py`
- `scripts/gemma_pt/infer_gemma_pt.py`

Add dedicated quantization runners later only after the baseline path is stable.
