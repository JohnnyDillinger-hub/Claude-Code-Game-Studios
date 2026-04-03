# Long-Context Research Track

This directory is for long-context smoke tests and later benchmark harnesses
that compare the main coding model path against the Gemma research contour.

## Current Scope

- start with small smoke runs before a large benchmark matrix
- keep model, prompt format, context length, latency, memory, and score together
- prefer comparable settings over broad coverage

## Expected Artifacts

- benchmark plans and notes in markdown
- machine-readable results in `research/long_context/results/`

## Current Bootstrap

Use the Gemma PT loader/inference scripts first. Add benchmark-specific runners
only after the result schema is settled.
