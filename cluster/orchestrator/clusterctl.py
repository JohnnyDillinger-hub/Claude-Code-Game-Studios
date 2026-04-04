from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
from pathlib import Path
import sys
from typing import Sequence

from cluster.demo import LOCAL_NODE_PATH, REMOTE_NODES_PATH
from cluster.models import AgentRequest, NodeInventory, load_node_inventory_file, utc_now
from cluster.node_agent.daemon import main as node_agent_main
from cluster.node_agent.heartbeat import (
    apply_heartbeat_to_state_file,
    build_heartbeat_payload,
    fetch_inventory_from_url,
)
from cluster.node_agent.probe_gpu import ProbeError, build_local_inventory, parse_label_items
from cluster.orchestrator.launcher import launch_agent
from cluster.orchestrator.model_profiles import get_runtime_profile, load_runtime_profiles
from cluster.orchestrator.registry import NodeRegistry
from cluster.orchestrator.scheduler import schedule_agent
from cluster.orchestrator.state_store import RegistryStateStore


DEFAULT_STATE_FILE = Path("production/session-state/cluster-registry.json")


def _resolve_inventory_path(path: str | None, default_path: Path) -> Path:
    return Path(path) if path else default_path


def _load_local_and_remote(
    local_file: str | None,
    remote_file: str | None,
) -> tuple[NodeInventory, list[NodeInventory]]:
    local_path = _resolve_inventory_path(local_file, LOCAL_NODE_PATH)
    remote_path = _resolve_inventory_path(remote_file, REMOTE_NODES_PATH)
    local_node = load_node_inventory_file(local_path)[0]
    remote_nodes = load_node_inventory_file(remote_path)
    return local_node, remote_nodes


def _load_registry_from_args(args: argparse.Namespace) -> NodeRegistry:
    if getattr(args, "state_file", None):
        return RegistryStateStore(args.state_file).load()
    local_node, remote_nodes = _load_local_and_remote(
        getattr(args, "local_file", None),
        getattr(args, "remote_file", None),
    )
    return NodeRegistry([local_node, *remote_nodes])


def _resolve_local_node_for_registry(args: argparse.Namespace, registry: NodeRegistry) -> NodeInventory:
    local_node_id = getattr(args, "local_node_id", None)
    if local_node_id:
        record = registry.get_record(local_node_id)
        if record is None:
            raise ValueError(f"Local node id {local_node_id!r} is not present in the registry")
        return record.node
    nodes = registry.list_nodes()
    if not nodes:
        raise ValueError("Registry is empty; cannot determine a local node")
    return nodes[0]


def _build_request(args: argparse.Namespace) -> AgentRequest:
    profile = get_runtime_profile(args.profile) if args.profile else None
    required_vram = args.vram_required_mib or (
        profile.required_free_vram_mib if profile else None
    )
    model_id = args.model_id or (profile.model_name if profile else None)
    if required_vram is None:
        raise ValueError("Provide --vram-required-mib or use a known --profile")
    labels = parse_label_items(args.label or [])
    return AgentRequest(
        agent_id=args.agent_id,
        required_vram_mib=int(required_vram),
        model_id=str(model_id) if model_id else None,
        labels=tuple(sorted(labels.items())),
        trust_tier=args.trust_tier,
        network_tier=args.network_tier,
    )


def _print_inventory_summary(nodes: list[NodeInventory]) -> None:
    now = datetime.now(tz=UTC)
    for node in nodes:
        status = "expired" if node.is_expired(now=now) else "active"
        cached = ", ".join(node.cached_models) if node.cached_models else "-"
        print(
            f"{node.node_id} [{status}] host={node.host} gpus={node.gpu_count} "
            f"available_until={node.to_dict()['available_until']} cached_models={cached}"
        )
        for gpu in node.gpus:
            util = f"{gpu.utilization_pct}%" if gpu.utilization_pct is not None else "n/a"
            print(
                f"  - gpu {gpu.index}: free={gpu.free_memory_mib} MiB / "
                f"total={gpu.total_memory_mib} MiB util={util}"
            )


def build_parser() -> argparse.ArgumentParser:
    profiles = load_runtime_profiles()
    parser = argparse.ArgumentParser(description="Phase 2 cluster control CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_parser = subparsers.add_parser("probe-local", help="Probe local node inventory as JSON.")
    probe_parser.add_argument("--node-id", default="local-node")
    probe_parser.add_argument("--host")
    probe_parser.add_argument("--lease-duration-seconds", type=int, default=3600)
    probe_parser.add_argument("--cached-model", action="append", default=[])
    probe_parser.add_argument("--label", action="append", default=[])
    probe_parser.add_argument("--trust-tier")
    probe_parser.add_argument("--network-tier")

    show_parser = subparsers.add_parser("show-inventory", help="Show inventory from demo files or state.")
    show_parser.add_argument("--local-file")
    show_parser.add_argument("--remote-file")
    show_parser.add_argument("--state-file")

    schedule_parser = subparsers.add_parser("schedule-agent", help="Compute a PlacementDecision.")
    schedule_parser.add_argument("--local-file")
    schedule_parser.add_argument("--remote-file")
    schedule_parser.add_argument("--state-file")
    schedule_parser.add_argument("--local-node-id")
    schedule_parser.add_argument("--agent-id", default="agent-demo")
    schedule_parser.add_argument("--profile", choices=sorted(profiles))
    schedule_parser.add_argument("--model-id")
    schedule_parser.add_argument("--vram-required-mib", type=int)
    schedule_parser.add_argument("--label", action="append", default=[])
    schedule_parser.add_argument("--trust-tier")
    schedule_parser.add_argument("--network-tier")

    prune_parser = subparsers.add_parser("prune-expired", help="Prune expired or stale nodes.")
    prune_parser.add_argument("--local-file")
    prune_parser.add_argument("--remote-file")
    prune_parser.add_argument("--state-file")
    prune_parser.add_argument("--write-back")
    prune_parser.add_argument("--stale-after-seconds", type=int)

    start_parser = subparsers.add_parser("start-node-agent", help="Start the Phase 2 node agent.")
    start_parser.add_argument("--config")
    start_parser.add_argument("--node-id")
    start_parser.add_argument("--host")
    start_parser.add_argument("--bind-host")
    start_parser.add_argument("--port")
    start_parser.add_argument("--available-until")
    start_parser.add_argument("--lease-duration-seconds")
    start_parser.add_argument("--cached-model", action="append", default=[])
    start_parser.add_argument("--label", action="append", default=[])
    start_parser.add_argument("--trust-tier")
    start_parser.add_argument("--network-tier")
    start_parser.add_argument("--no-allow-empty", action="store_true")
    start_parser.add_argument("--heartbeat-interval-seconds")
    start_parser.add_argument("--heartbeat-url")
    start_parser.add_argument("--heartbeat-state-file")
    start_parser.add_argument("--heartbeat-timeout-seconds")

    heartbeat_parser = subparsers.add_parser(
        "heartbeat-once",
        help="Fetch node inventory and update a persistent registry snapshot.",
    )
    heartbeat_parser.add_argument(
        "--inventory-url",
        default="http://127.0.0.1:8787/inventory",
    )
    heartbeat_parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE))
    heartbeat_parser.add_argument("--heartbeat-interval-seconds", type=int, default=30)
    heartbeat_parser.add_argument("--dry-run", action="store_true")

    launch_parser = subparsers.add_parser(
        "launch-agent",
        help="Schedule an agent and print or execute the launch command.",
    )
    launch_parser.add_argument("--local-file")
    launch_parser.add_argument("--remote-file")
    launch_parser.add_argument("--state-file")
    launch_parser.add_argument("--local-node-id")
    launch_parser.add_argument("--agent-id", default="agent-demo")
    launch_parser.add_argument("--profile", required=True, choices=sorted(profiles))
    launch_parser.add_argument("--model-id")
    launch_parser.add_argument("--vram-required-mib", type=int)
    launch_parser.add_argument("--label", action="append", default=[])
    launch_parser.add_argument("--trust-tier")
    launch_parser.add_argument("--network-tier")
    launch_parser.add_argument("--dry-run", action="store_true")
    launch_parser.add_argument("--ssh-user")
    launch_parser.add_argument("--ssh-port", type=int)
    launch_parser.add_argument("--repo-root")
    launch_parser.add_argument("--min-lease-remaining-seconds", type=int, default=30)

    save_parser = subparsers.add_parser("save-registry", help="Save registry state to disk.")
    save_parser.add_argument("--local-file")
    save_parser.add_argument("--remote-file")
    save_parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE))

    load_parser = subparsers.add_parser("load-registry", help="Load registry state from disk.")
    load_parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE))

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "probe-local":
        try:
            labels = parse_label_items(args.label)
            inventory = build_local_inventory(
                node_id=args.node_id,
                host=args.host,
                lease_duration_seconds=args.lease_duration_seconds,
                cached_models=args.cached_model,
                labels=labels,
                trust_tier=args.trust_tier,
                network_tier=args.network_tier,
                allow_empty=True,
            )
        except (ProbeError, ValueError) as exc:
            print(f"probe-local error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(inventory.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "show-inventory":
        registry = _load_registry_from_args(args)
        _print_inventory_summary(registry.list_nodes())
        return 0

    if args.command == "schedule-agent":
        registry = _load_registry_from_args(args)
        local_node = _resolve_local_node_for_registry(args, registry)
        remote_nodes = [
            node for node in registry.list_nodes() if node.node_id != local_node.node_id
        ]
        request = _build_request(args)
        decision = schedule_agent(local_node, remote_nodes, request)
        print(json.dumps(decision.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "prune-expired":
        registry = _load_registry_from_args(args)
        removed = registry.prune_expired(stale_after_seconds=args.stale_after_seconds)
        remaining = registry.list_nodes()
        if args.write_back:
            RegistryStateStore(args.write_back).save(registry)
        print(
            json.dumps(
                {
                    "removed": removed,
                    "remaining_count": len(remaining),
                    "remaining_node_ids": [node.node_id for node in remaining],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "start-node-agent":
        forwarded_args: list[str] = []
        for name in [
            "config",
            "node_id",
            "host",
            "bind_host",
            "port",
            "available_until",
            "lease_duration_seconds",
            "trust_tier",
            "network_tier",
            "heartbeat_interval_seconds",
            "heartbeat_url",
            "heartbeat_state_file",
            "heartbeat_timeout_seconds",
        ]:
            value = getattr(args, name)
            if value is not None:
                forwarded_args.extend([f"--{name.replace('_', '-')}", str(value)])
        for value in args.cached_model:
            forwarded_args.extend(["--cached-model", value])
        for value in args.label:
            forwarded_args.extend(["--label", value])
        if args.no_allow_empty:
            forwarded_args.append("--no-allow-empty")
        return node_agent_main(forwarded_args)

    if args.command == "heartbeat-once":
        inventory = fetch_inventory_from_url(args.inventory_url)
        payload = build_heartbeat_payload(
            inventory,
            heartbeat_interval_seconds=args.heartbeat_interval_seconds,
            sent_at=utc_now(),
        )
        if args.dry_run:
            print(json.dumps(payload.to_dict(), indent=2, sort_keys=True))
            return 0
        result = apply_heartbeat_to_state_file(args.state_file, payload)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "launch-agent":
        registry = _load_registry_from_args(args)
        local_node = _resolve_local_node_for_registry(args, registry)
        remote_nodes = [
            node for node in registry.list_nodes() if node.node_id != local_node.node_id
        ]
        request = _build_request(args)
        profile = get_runtime_profile(args.profile)
        decision = schedule_agent(local_node, remote_nodes, request)
        if not decision.is_placed:
            print(
                json.dumps(
                    {
                        "placement": decision.to_dict(),
                        "launch": {
                            "status": "blocked",
                            "reason": decision.reason,
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        try:
            result = launch_agent(
                request,
                decision,
                profile,
                dry_run=args.dry_run,
                ssh_user=args.ssh_user,
                ssh_port=args.ssh_port,
                repo_root=args.repo_root,
                min_lease_remaining_seconds=args.min_lease_remaining_seconds,
            )
        except ValueError as exc:
            print(
                json.dumps(
                    {
                        "placement": decision.to_dict(),
                        "launch": {
                            "status": "blocked",
                            "reason": str(exc),
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        print(
            json.dumps(
                {
                    "placement": decision.to_dict(),
                    "launch": result.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if result.status in {"ready", "launched"} else 1

    if args.command == "save-registry":
        local_node, remote_nodes = _load_local_and_remote(args.local_file, args.remote_file)
        registry = NodeRegistry([local_node, *remote_nodes])
        store = RegistryStateStore(args.state_file)
        store.save(registry)
        print(
            json.dumps(
                {
                    "status": "saved",
                    "state_file": str(Path(args.state_file)),
                    "record_count": len(registry.list_nodes()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "load-registry":
        store = RegistryStateStore(args.state_file)
        registry = store.load()
        print(json.dumps(registry.to_state_dict(), indent=2, sort_keys=True))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
