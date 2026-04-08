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
- dry-run provider resource creation stubs
- bootstrap bundle generation for future auto-join

Supported provider adapters:

- `vast`
- `runpod`
- `nebius`

Current provider commands in [clusterctl.py](/Users/ivandry/GitHub/Claude-Code-Game-Studios/cluster/orchestrator/clusterctl.py):

- `providers-list-offers`
- `providers-list-blueprints`
- `providers-provision`
- `providers-jobs`

## Discovery Vs Provisioning Vs Joining

Discovery:

- Lists normalized external capacity from provider adapters.
- Returns `ProviderOffer`.
- Does not affect cluster scheduling.

Provisioning:

- Creates a `ProvisionJob`.
- Selects an offer or blueprint.
- Generates a `BootstrapBundle`.
- May return a dry-run `ProvisionedResource`.

Joining:

- Is not automatic yet.
- Will happen when the created VM or Pod starts the existing node agent and heartbeats into the registry.
- Only after that does the resource become `NodeInventory` and participate in scheduling.

## What Is Deferred

This phase intentionally does not implement:

- autoscaling
- budget policy
- provider-side billing controls
- full secret management UX
- automatic NAT traversal
- full provider create flows in production mode
- cluster join confirmation from provider jobs

## Managed Vs Bring Your Own Provider

The intended default UX is managed-first:

- end users talk only to our control plane
- provider credentials stay server-side
- the GUI does not need direct provider registration for basic usage

Later we can add a bring-your-own-account mode where a user connects their own provider credentials to the control plane, but the GUI still should not hold those secrets directly.
