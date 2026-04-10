# Developer Update Monitoring

This slice is for system developers and operators only.

It reports update status for externally versioned components that the project
depends on, without affecting normal client or user flows.

## What It Tracks

The current default report tracks:

- `vLLM`
- `SGLang`
- `DeepSpeed`
- `TensorRT-LLM`
- `Ollama`
- the runtime profile catalog in
  [cluster/orchestrator/model_profiles.yaml](/Users/ivandry/GitHub/Claude-Code-Game-Studios/cluster/orchestrator/model_profiles.yaml)

Each tracker reports:

- the detected/local version
- the latest known upstream version when available
- whether an update is available
- a short normalized change summary when a release note snippet is available
- source references used to produce the report

## CLI

Run the developer-only report with:

```bash
python3 -m cluster.orchestrator.clusterctl developer-update-report
```

Useful flags:

- `--format json|text`
- `--offline`
- `--timeout-seconds N`
- `--repo-root PATH`

Examples:

```bash
python3 -m cluster.orchestrator.clusterctl developer-update-report --offline
python3 -m cluster.orchestrator.clusterctl developer-update-report --format text
```

## Output Behavior

- The report is best-effort.
- If an upstream lookup fails, the component is marked unavailable or unknown
  instead of failing the whole command.
- Offline mode skips upstream lookups entirely and only reports local state.
- Stable JSON output is intended for scripts and CI.

## Intended Use

This report is meant for:

- developers checking whether local runtime dependencies are drifting
- operators reviewing release cadence and update pressure
- CI or automation that needs a compact, stable dependency health snapshot

It is not part of the user-facing mesh, provider, or launch flows.

