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
- dry-run provider resource creation stubs for the remaining providers
- bootstrap bundle generation for future auto-join

For `Vast`, a non-dry-run `providers-provision` request now uses the real
provider create endpoint when `VAST_API_KEY` is present. Without that key, the
command must stay in `--dry-run`.

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

`providers-provision` can now also wait for join confirmation in the same call:

- `--wait-for-join`
- `--join-state-file`
- `--join-timeout-seconds`
- `--join-poll-interval-seconds`

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
- Only after that does the resource become `NodeInventory` and participate in scheduling.

## What Is Deferred

This phase intentionally does not implement:

- autoscaling
- budget policy
- provider-side billing controls
- full secret management UX
- automatic NAT traversal
- full provider create flows in production mode for `runpod` and `nebius`
- background async waiting/polling for cluster join after provisioning

## Managed Vs Bring Your Own Provider

The intended default UX is managed-first:

- end users talk only to our control plane
- provider credentials stay server-side
- the GUI does not need direct provider registration for basic usage

Later we can add a bring-your-own-account mode where a user connects their own provider credentials to the control plane, but the GUI still should not hold those secrets directly.
