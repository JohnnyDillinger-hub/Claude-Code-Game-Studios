# Providers Phase 3

Phase 3 adds an external capacity layer on top of the existing mesh and runtime launch work.

The key rule is:

- A provider offer is not a mesh node.
- A provisioned VM or Pod is not yet schedulable.
- Only a joined node-agent heartbeat becomes an active cluster node.

## What Is Implemented

The current implementation adds a provider service under [cluster/providers](/Users/ivandry/GitHub/Claude-Code-Game-Studios/cluster/providers):

- normalized provider models
- built-in blueprints
- JSON-backed provisioning jobs
- CLI discovery and provisioning scaffolding
- real `Vast` create-path when `VAST_API_KEY` is configured
- real `Runpod` create-path when `RUNPOD_API_KEY` is configured
- dry-run provider resource creation stubs for Nebius
- bootstrap bundle generation for future auto-join
- post-join runtime bootstrap plans for provider-created nodes
- best-effort SSH runtime bootstrap once a joined node exposes provider SSH details

For `Vast`, a non-dry-run `providers-provision` request now uses the real
provider create endpoint when `VAST_API_KEY` is present. Without that key, the
command must stay in `--dry-run`.

For `Runpod`, a non-dry-run `providers-provision` request now uses the real Pod
create endpoint when `RUNPOD_API_KEY` is present and the request includes a
blueprint, template, or direct pod config. Without that key, the command must
stay in `--dry-run`.

Supported provider adapters:

- `vast`
- `runpod`
- `nebius`

Current provider commands in [clusterctl.py](/Users/ivandry/GitHub/Claude-Code-Game-Studios/cluster/orchestrator/clusterctl.py):

- `providers-list-offers`
- `providers-list-blueprints`
- `providers-provision`
- `providers-jobs`
- `providers-reconcile-jobs`
- `providers-destroy`

`providers-provision` can now also wait for join confirmation in the same call:

- `--wait-for-join`
- `--join-state-file`
- `--join-timeout-seconds`
- `--join-poll-interval-seconds`

`providers-jobs` now supports operator-oriented filters:

- `--provider`
- `--resource-id`
- `--resource-status`

`providers-destroy` uses a preview-first safety flow:

- without `--confirm`, it prints a dry-run destroy plan and does not mutate the job store
- with `--confirm`, it destroys the concrete provider resource and marks the job as `destroyed`
- `--resource-id` can be used instead of `--job-id` when the external resource id is known

## Discovery Vs Provisioning Vs Joining

Discovery:

- Lists normalized external capacity from provider adapters.
- Returns `ProviderOffer`.
- Does not affect cluster scheduling.

Provisioning:

- Creates a `ProvisionJob`.
- Selects an offer or blueprint.
- Generates a `BootstrapBundle`.
- Returns a dry-run `ProvisionedResource` when no real provider call is made.
- Can create a real Vast instance and normalize its SSH details into `ProvisionedResource`.

Joining:

- Still depends on the existing node agent starting and heartbeating into the registry.
- Can now be waited for directly during `providers-provision` when the caller passes `--wait-for-join`.
- Can now be confirmed from provider jobs through `providers-reconcile-jobs`.
- Marks the job as `joined`, captures the matched `node_id`, and stores a snapshot of the joined node.
- Can now carry a separate `runtime_bootstrap_status` alongside the join status.
- When a joined node still lacks the runtime stack requested by the blueprint, the provider service can start a background runtime bootstrap over SSH.
- Only after that does the resource become `NodeInventory` and participate in scheduling.

Destroying:

- Can be previewed through `providers-destroy` without changing state.
- Can be executed through `providers-destroy --confirm` to destroy a provisioned Vast resource or a stored dry-run placeholder.
- Keeps the provider job as a durable record with `status=destroyed`.

Runtime bootstrap:

- Is driven by blueprint-level `runtime_stack` and optional `preferred_launch_profile`.
- Uses a staged flow that waits briefly after join, performs disk preflight/cleanup, and then installs runtimes with retries and timeouts through [bootstrap_provider_node.sh](/Users/ivandry/GitHub/Claude-Code-Game-Studios/scripts/runtime/bootstrap_provider_node.sh).
- Is recorded separately from provider provisioning in the `ProvisionJob`.
- Still treats model preloading as deferred work; the current phase focuses on getting runtime environments onto the joined node without overwhelming fresh provider disks or SSH services.
- Can surface `stabilizing`, `starting`, `repairing`, and `ready` states during bootstrap.

## What Is Deferred

This phase intentionally does not implement:

- autoscaling
- budget policy
- provider-side billing controls
- full secret management UX
- automatic NAT traversal
- full provider create flow for `nebius`
- background async waiting/polling for cluster join after provisioning
- full post-join runtime bootstrap coverage for `ollama` and `tensorrt-llm`
- automatic model artifact preloading after runtime installation
- automatic background garbage collection of destroyed provider job records

## Managed Vs Bring Your Own Provider

The intended default UX is managed-first:

- end users talk only to our control plane
- provider credentials stay server-side
- the GUI does not need direct provider registration for basic usage

Later we can add a bring-your-own-account mode where a user connects their own provider credentials to the control plane, but the GUI still should not hold those secrets directly.
