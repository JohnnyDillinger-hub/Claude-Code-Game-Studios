from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence

from cluster.demo import LOCAL_NODE_PATH, REMOTE_NODES_PATH
from cluster.models import (
    AgentDeploymentSpec,
    AgentRequest,
    CUDA_GRAPH_MODE_VALUES,
    NodeInventory,
    RuntimeLaunchPreferences,
    load_node_inventory_file,
    utc_now,
)
from cluster.node_agent.daemon import main as node_agent_main
from cluster.node_agent.heartbeat import (
    apply_heartbeat_to_state_file,
    build_heartbeat_payload,
    fetch_inventory_from_url,
)
from cluster.node_agent.probe_gpu import ProbeError, build_local_inventory, parse_label_items
from cluster.orchestrator.launcher import build_remote_ssh_argv, launch_agent
from cluster.orchestrator.model_profiles import (
    RuntimeProfile,
    get_runtime_profile,
    load_runtime_profiles,
)
from cluster.orchestrator.remote_sessions import RemoteSessionClaim
from cluster.orchestrator.registry import NodeRegistry
from cluster.orchestrator.scheduler import schedule_agent
from cluster.orchestrator.state_store import RegistryStateStore
from cluster.providers.base import ProviderError
from cluster.providers.models import ProvisionRequest
from cluster.providers.service import DEFAULT_JOBS_FILE, ProviderService


DEFAULT_STATE_FILE = Path("production/session-state/cluster-registry.json")
DEFAULT_REMOTE_SESSION_DIR = "production/session-state/remote-workers"
DEFAULT_REPO_ROOT = "$HOME/Claude-Code-Game-Studios"
DEFAULT_SSH_USER = "root"
PROVIDER_CHOICES = ("vast", "runpod", "nebius")
CUDA_GRAPH_MODE_CHOICES = CUDA_GRAPH_MODE_VALUES


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


def _resolve_node_access(
    node: NodeInventory | None,
    *,
    ssh_user: str | None,
    ssh_port: int | None,
    repo_root: str | None,
) -> tuple[str | None, int | None, str]:
    access = node.access if node is not None else None
    resolved_user = ssh_user or (access.ssh_user if access is not None else None) or DEFAULT_SSH_USER
    resolved_port = ssh_port if ssh_port is not None else (access.ssh_port if access is not None else None)
    resolved_repo_root = (
        repo_root
        or (access.repo_root if access is not None else None)
        or DEFAULT_REPO_ROOT
    )
    return resolved_user, resolved_port, resolved_repo_root


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
    required_gpu_count = args.gpu_count_required or (
        profile.required_gpu_count if profile else 1
    )
    model_id = args.model_id or (profile.model_name if profile else None)
    if required_vram is None:
        raise ValueError("Provide --vram-required-mib or use a known --profile")
    if int(required_gpu_count) <= 0:
        raise ValueError("required_gpu_count must be at least 1")
    labels = parse_label_items(args.label or [])
    return AgentRequest(
        agent_id=args.agent_id,
        required_vram_mib=int(required_vram),
        required_gpu_count=int(required_gpu_count),
        model_id=str(model_id) if model_id else None,
        labels=tuple(sorted(labels.items())),
        trust_tier=args.trust_tier,
        network_tier=args.network_tier,
    )


def _build_launch_preferences(args: argparse.Namespace) -> RuntimeLaunchPreferences | None:
    cuda_graph_mode = getattr(args, "cuda_graph_mode", "profile-default")
    cuda_graph_max_bs = getattr(args, "cuda_graph_max_bs", None)
    if cuda_graph_mode == "profile-default" and cuda_graph_max_bs is None:
        return None
    return RuntimeLaunchPreferences(
        cuda_graph_mode=cuda_graph_mode,
        cuda_graph_max_bs=cuda_graph_max_bs,
    )


def _build_deployment_spec(args: argparse.Namespace) -> AgentDeploymentSpec:
    request = _build_request(args)
    return AgentDeploymentSpec(
        agent_id=request.agent_id,
        profile=str(args.profile),
        required_vram_mib=request.required_vram_mib,
        required_gpu_count=request.required_gpu_count,
        model_id=request.model_id,
        labels=request.labels,
        trust_tier=request.trust_tier,
        network_tier=request.network_tier,
        launch_preferences=_build_launch_preferences(args),
    )


def _parse_provider_options(items: Sequence[str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Provider option must use key=value syntax: {item!r}")
        key, raw_value = item.split("=", 1)
        normalized = raw_value.strip()
        lowered = normalized.lower()
        if lowered in {"true", "false"}:
            value: object = lowered == "true"
        else:
            try:
                value = int(normalized)
            except ValueError:
                try:
                    value = float(normalized)
                except ValueError:
                    value = normalized
        parsed[key.strip()] = value
    return parsed


def _build_provision_request(args: argparse.Namespace) -> ProvisionRequest:
    labels = parse_label_items(args.label or [])
    public_ip: bool | None = True if getattr(args, "public_ip", False) else None
    return ProvisionRequest(
        provider=args.provider,
        blueprint_id=getattr(args, "blueprint", None),
        offer_id=getattr(args, "offer_id", None),
        region=getattr(args, "region", None),
        gpu_count=getattr(args, "gpu_count", None),
        public_ip=public_ip,
        volume_gb=getattr(args, "volume_gb", None),
        preemptible_ok=bool(getattr(args, "preemptible_ok", False)),
        cached_models=tuple(getattr(args, "cached_model", []) or ()),
        labels=tuple(sorted(labels.items())),
        trust_tier=getattr(args, "trust_tier", None),
        network_tier=getattr(args, "network_tier", None),
        dry_run=bool(getattr(args, "dry_run", False)),
        provider_options=_parse_provider_options(getattr(args, "provider_option", []) or ()),
    )


def _apply_launch_profile_overrides(
    profile: RuntimeProfile,
    deployment_spec: AgentDeploymentSpec,
) -> RuntimeProfile:
    overrides: dict[str, object] = {}
    launch_preferences = deployment_spec.launch_preferences
    if launch_preferences is not None:
        if profile.runtime_adapter != "sglang-server":
            raise ValueError("CUDA graph overrides are currently supported only for SGLang profiles")
        cuda_graph_mode = launch_preferences.cuda_graph_mode or "profile-default"
        cuda_graph_max_bs = launch_preferences.cuda_graph_max_bs
        if cuda_graph_mode == "enabled":
            overrides["disable_cuda_graph"] = False
        elif cuda_graph_mode == "disabled":
            overrides["disable_cuda_graph"] = True
        if cuda_graph_max_bs is not None:
            overrides["cuda_graph_max_bs"] = int(cuda_graph_max_bs)
    return profile.with_launch_overrides(overrides)


def _build_remote_inventory_probe_command(
    *,
    node_id: str,
    host: str,
    lease_duration_seconds: int,
    cached_models: Sequence[str],
    labels: Sequence[str],
    trust_tier: str | None,
    network_tier: str | None,
    ssh_user: str | None,
    ssh_port: int | None,
    repo_root: str | None,
) -> list[str]:
    command = [
        "python3",
        "-m",
        "cluster.orchestrator.clusterctl",
        "probe-local",
        "--node-id",
        node_id,
        "--host",
        host,
        "--lease-duration-seconds",
        str(lease_duration_seconds),
        "--include-capabilities",
    ]
    if ssh_user is not None:
        command.extend(["--ssh-user", ssh_user])
    if ssh_port is not None:
        command.extend(["--ssh-port", str(ssh_port)])
    if repo_root is not None:
        command.extend(["--repo-root", repo_root])
    if trust_tier is not None:
        command.extend(["--trust-tier", trust_tier])
    if network_tier is not None:
        command.extend(["--network-tier", network_tier])
    for value in cached_models:
        command.extend(["--cached-model", value])
    for value in labels:
        command.extend(["--label", value])
    return command


def _probe_remote_node_inventory(
    *,
    node_id: str,
    host: str,
    ssh_user: str | None,
    ssh_port: int | None,
    repo_root: str,
    lease_duration_seconds: int,
    cached_models: Sequence[str],
    labels: Sequence[str],
    trust_tier: str | None,
    network_tier: str | None,
) -> NodeInventory:
    remote_command = _build_remote_inventory_probe_command(
        node_id=node_id,
        host=host,
        lease_duration_seconds=lease_duration_seconds,
        cached_models=cached_models,
        labels=labels,
        trust_tier=trust_tier,
        network_tier=network_tier,
        ssh_user=ssh_user,
        ssh_port=ssh_port,
        repo_root=repo_root,
    )
    command = build_remote_ssh_argv(
        host=host,
        remote_command=remote_command,
        ssh_user=ssh_user,
        ssh_port=ssh_port,
        repo_root=repo_root,
        gpu_index=None,
    )
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError(f"Remote node probe for {node_id} did not return a JSON object")
    return NodeInventory.from_dict(payload)


def _build_remote_session_probe_command(session_dir: str) -> str:
    script = """
import json
import sys
from pathlib import Path

SESSION_OK = {"starting", "launched", "reused"}
session_dir = Path(sys.argv[1]).expanduser()
payloads = []
claims = []
if session_dir.exists():
    for candidate in sorted(session_dir.glob("*.json")):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        payloads.append(payload)
        status = str(payload.get("status") or "")
        agent_id = str(payload.get("agent_id") or "").strip()
        node_id = str(payload.get("node_id") or "").strip()
        raw_gpu_indices = payload.get("gpu_indices")
        if isinstance(raw_gpu_indices, list) and raw_gpu_indices:
            try:
                gpu_indices = [int(item) for item in raw_gpu_indices]
            except (TypeError, ValueError):
                gpu_indices = []
        else:
            gpu_index = payload.get("gpu_index")
            if gpu_index is None:
                gpu_indices = []
            else:
                try:
                    gpu_indices = [int(gpu_index)]
                except (TypeError, ValueError):
                    gpu_indices = []
        if status not in SESSION_OK or not agent_id or not node_id or not gpu_indices:
            continue
        for gpu_index in gpu_indices:
            claim = {
                "agent_id": agent_id,
                "node_id": node_id,
                "gpu_index": gpu_index,
                "status": status,
            }
            if payload.get("backend") is not None:
                claim["backend"] = str(payload["backend"])
            if payload.get("model") is not None:
                claim["model"] = str(payload["model"])
            if payload.get("listen_port") is not None:
                try:
                    claim["listen_port"] = int(payload["listen_port"])
                except (TypeError, ValueError):
                    pass
            if payload.get("session_file") is not None:
                claim["session_file"] = str(payload["session_file"])
            claims.append(claim)

print(json.dumps({
    "session_dir": str(session_dir.resolve()),
    "session_count": len(payloads),
    "sessions": payloads,
    "claims": claims,
}, sort_keys=True))
""".strip()
    return (
        f"python3 -c {shlex.quote(script)} {shlex.quote(session_dir)}"
    )


def _build_remote_session_stop_command(session_dir: str, agent_id: str) -> str:
    script = """
import json
import os
import signal
import sys
from pathlib import Path
import re

def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)

session_dir = Path(sys.argv[1]).expanduser()
agent_id = sys.argv[2]
session_path = session_dir / f"{sanitize(agent_id)}.json"
try:
    payload = json.loads(session_path.read_text(encoding="utf-8"))
except FileNotFoundError:
    print(json.dumps({
        "status": "not-found",
        "agent_id": agent_id,
        "session_file": str(session_path),
    }, sort_keys=True))
    raise SystemExit(0)

server_pid = payload.get("server_pid")
stopped_pid = None
if server_pid is not None:
    try:
        stopped_pid = int(server_pid)
        os.kill(stopped_pid, signal.SIGTERM)
    except (OSError, TypeError, ValueError):
        stopped_pid = None

payload["status"] = "stopped"
payload["reused"] = False
notes = str(payload.get("notes") or "").strip()
payload["notes"] = (
    f"{notes} Session marked stopped via remote session control.".strip()
    if notes
    else "Session marked stopped via remote session control."
)
session_path.parent.mkdir(parents=True, exist_ok=True)
session_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
result = {
    "status": "stopped",
    "agent_id": agent_id,
    "session_file": str(session_path),
    "payload": payload,
}
if stopped_pid is not None:
    result["server_pid"] = stopped_pid
print(json.dumps(result, sort_keys=True))
""".strip()
    return (
        f"python3 -c {shlex.quote(script)} {shlex.quote(session_dir)} {shlex.quote(agent_id)}"
    )


def _fetch_remote_session_claims_for_node(
    node: NodeInventory,
    *,
    ssh_user: str | None,
    ssh_port: int | None,
    repo_root: str,
    session_dir: str,
) -> tuple[list[dict[str, object]], list[RemoteSessionClaim]]:
    resolved_ssh_user, resolved_ssh_port, resolved_repo_root = _resolve_node_access(
        node,
        ssh_user=ssh_user,
        ssh_port=ssh_port,
        repo_root=repo_root,
    )
    command = build_remote_ssh_argv(
        host=node.host,
        remote_command=_build_remote_session_probe_command(session_dir),
        ssh_user=resolved_ssh_user,
        ssh_port=resolved_ssh_port,
        repo_root=resolved_repo_root,
        gpu_index=None,
    )
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError(f"Remote session probe on {node.node_id} did not return a JSON object")
    sessions_raw = payload.get("sessions", [])
    claims_raw = payload.get("claims", [])
    if not isinstance(sessions_raw, list) or not isinstance(claims_raw, list):
        raise ValueError(f"Remote session probe on {node.node_id} returned malformed JSON")

    claims: list[RemoteSessionClaim] = []
    for item in claims_raw:
        if not isinstance(item, dict):
            continue
        try:
            claims.append(
                RemoteSessionClaim(
                    agent_id=str(item["agent_id"]),
                    node_id=str(item["node_id"]),
                    gpu_index=int(item["gpu_index"]),
                    status=str(item["status"]),
                    backend=(
                        str(item["backend"]) if item.get("backend") is not None else None
                    ),
                    model=str(item["model"]) if item.get("model") is not None else None,
                    listen_port=(
                        int(item["listen_port"])
                        if item.get("listen_port") is not None
                        else None
                    ),
                    session_file=(
                        str(item["session_file"])
                        if item.get("session_file") is not None
                        else None
                    ),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Remote session probe on {node.node_id} returned an invalid claim: {exc}"
            ) from exc
    return [dict(item) for item in sessions_raw if isinstance(item, dict)], claims


def _filter_nodes_by_remote_session_claims(
    nodes: list[NodeInventory],
    *,
    request_agent_id: str,
    ssh_user: str | None,
    ssh_port: int | None,
    repo_root: str,
    session_dir: str,
) -> tuple[list[NodeInventory], list[dict[str, object]]]:
    filtered_nodes: list[NodeInventory] = []
    claim_summaries: list[dict[str, object]] = []
    for node in nodes:
        _, claims = _fetch_remote_session_claims_for_node(
            node,
            ssh_user=ssh_user,
            ssh_port=ssh_port,
            repo_root=repo_root,
            session_dir=session_dir,
        )
        blocked_gpu_indices = {
            claim.gpu_index for claim in claims if claim.agent_id != request_agent_id
        }
        filtered_nodes.append(
            node.with_gpus(gpu for gpu in node.gpus if gpu.index not in blocked_gpu_indices)
        )
        claim_summaries.extend(claim.to_dict() for claim in claims)
    return filtered_nodes, claim_summaries


def _print_inventory_summary(nodes: list[NodeInventory]) -> None:
    now = datetime.now(tz=timezone.utc)
    for node in nodes:
        status = "expired" if node.is_expired(now=now) else "active"
        cached = ", ".join(node.cached_models) if node.cached_models else "-"
        print(
            f"{node.node_id} [{status}] host={node.host} gpus={node.gpu_count} "
            f"available_until={node.to_dict()['available_until']} cached_models={cached}"
        )
        if node.access is not None:
            access_bits = []
            if node.access.ssh_user is not None:
                access_bits.append(f"user={node.access.ssh_user}")
            if node.access.ssh_port is not None:
                access_bits.append(f"port={node.access.ssh_port}")
            if node.access.repo_root is not None:
                access_bits.append(f"repo_root={node.access.repo_root}")
            if access_bits:
                print(f"  access: {' '.join(access_bits)}")
        if node.system_info is not None:
            system_bits = []
            if node.system_info.python_version is not None:
                system_bits.append(f"python={node.system_info.python_version}")
            if node.system_info.driver_version is not None:
                system_bits.append(f"driver={node.system_info.driver_version}")
            if node.system_info.cuda_version is not None:
                system_bits.append(f"cuda={node.system_info.cuda_version}")
            if system_bits:
                print(f"  system: {' '.join(system_bits)}")
        if node.runtime_capabilities:
            runtime_bits = []
            for capability in node.runtime_capabilities:
                version = capability.version if capability.version is not None else "missing"
                runtime_bits.append(f"{capability.name}={version}")
            print(f"  runtimes: {', '.join(runtime_bits)}")
        if node.topology is not None:
            topology_text = (
                "single-node-mgpu"
                if node.topology.single_node_multi_gpu
                else "single-gpu"
            )
            if node.topology.interconnect is not None:
                topology_text = f"{topology_text} interconnect={node.topology.interconnect}"
            print(f"  topology: {topology_text}")
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
    probe_parser.add_argument("--ssh-user")
    probe_parser.add_argument("--ssh-port", type=int)
    probe_parser.add_argument("--repo-root")
    probe_parser.add_argument("--include-capabilities", action="store_true")

    remote_probe_parser = subparsers.add_parser(
        "probe-remote-node",
        help="Probe a remote node over SSH and optionally upsert it into the registry state.",
    )
    remote_probe_parser.add_argument("--node-id", required=True)
    remote_probe_parser.add_argument("--host", required=True)
    remote_probe_parser.add_argument("--lease-duration-seconds", type=int, default=3600)
    remote_probe_parser.add_argument("--cached-model", action="append", default=[])
    remote_probe_parser.add_argument("--label", action="append", default=[])
    remote_probe_parser.add_argument("--trust-tier")
    remote_probe_parser.add_argument("--network-tier")
    remote_probe_parser.add_argument("--ssh-user")
    remote_probe_parser.add_argument("--ssh-port", type=int)
    remote_probe_parser.add_argument("--repo-root")
    remote_probe_parser.add_argument("--state-file")

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
    schedule_parser.add_argument("--gpu-count-required", type=int)
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
    launch_parser.add_argument("--gpu-count-required", type=int)
    launch_parser.add_argument("--label", action="append", default=[])
    launch_parser.add_argument("--trust-tier")
    launch_parser.add_argument("--network-tier")
    launch_parser.add_argument("--dry-run", action="store_true")
    launch_parser.add_argument("--ssh-user")
    launch_parser.add_argument("--ssh-port", type=int)
    launch_parser.add_argument("--repo-root")
    launch_parser.add_argument("--session-dir", default=DEFAULT_REMOTE_SESSION_DIR)
    launch_parser.add_argument("--probe-remote-sessions", action="store_true")
    launch_parser.add_argument("--min-lease-remaining-seconds", type=int, default=30)
    launch_parser.add_argument(
        "--cuda-graph-mode",
        choices=CUDA_GRAPH_MODE_CHOICES,
        default="profile-default",
    )
    launch_parser.add_argument("--cuda-graph-max-bs", type=int)

    session_list_parser = subparsers.add_parser(
        "list-remote-sessions",
        help="Inspect remote worker sessions over SSH.",
    )
    session_list_parser.add_argument("--local-file")
    session_list_parser.add_argument("--remote-file")
    session_list_parser.add_argument("--state-file")
    session_list_parser.add_argument("--node-id")
    session_list_parser.add_argument("--ssh-user")
    session_list_parser.add_argument("--ssh-port", type=int)
    session_list_parser.add_argument("--repo-root")
    session_list_parser.add_argument("--session-dir", default=DEFAULT_REMOTE_SESSION_DIR)

    session_stop_parser = subparsers.add_parser(
        "stop-remote-session",
        help="Stop a remote worker session over SSH.",
    )
    session_stop_parser.add_argument("--local-file")
    session_stop_parser.add_argument("--remote-file")
    session_stop_parser.add_argument("--state-file")
    session_stop_parser.add_argument("--node-id", required=True)
    session_stop_parser.add_argument("--agent-id", required=True)
    session_stop_parser.add_argument("--ssh-user")
    session_stop_parser.add_argument("--ssh-port", type=int)
    session_stop_parser.add_argument("--repo-root")
    session_stop_parser.add_argument("--session-dir", default=DEFAULT_REMOTE_SESSION_DIR)

    provider_offers_parser = subparsers.add_parser(
        "providers-list-offers",
        help="List normalized provider offers from external capacity adapters.",
    )
    provider_offers_parser.add_argument("--provider", choices=sorted(PROVIDER_CHOICES))
    provider_offers_parser.add_argument("--gpu-name")
    provider_offers_parser.add_argument("--min-gpu-count", type=int)
    provider_offers_parser.add_argument("--min-vram-gb", type=float)
    provider_offers_parser.add_argument("--max-price-hourly", type=float)
    provider_offers_parser.add_argument("--region")
    provider_offers_parser.add_argument("--preemptible-ok", action="store_true")

    provider_blueprints_parser = subparsers.add_parser(
        "providers-list-blueprints",
        help="List built-in provider provisioning blueprints.",
    )
    provider_blueprints_parser.add_argument("--provider", choices=sorted((*PROVIDER_CHOICES, "any")))
    provider_blueprints_parser.add_argument("--model-id")

    provider_provision_parser = subparsers.add_parser(
        "providers-provision",
        help="Create a provider provisioning job and emit its bootstrap contract.",
    )
    provider_provision_parser.add_argument("--provider", required=True, choices=sorted((*PROVIDER_CHOICES, "any")))
    provider_provision_parser.add_argument("--blueprint")
    provider_provision_parser.add_argument("--offer-id")
    provider_provision_parser.add_argument("--region")
    provider_provision_parser.add_argument("--gpu-count", type=int)
    provider_provision_parser.add_argument("--public-ip", action="store_true")
    provider_provision_parser.add_argument("--volume-gb", type=int)
    provider_provision_parser.add_argument("--preemptible-ok", action="store_true")
    provider_provision_parser.add_argument("--cached-model", action="append", default=[])
    provider_provision_parser.add_argument("--label", action="append", default=[])
    provider_provision_parser.add_argument("--trust-tier")
    provider_provision_parser.add_argument("--network-tier")
    provider_provision_parser.add_argument("--provider-option", action="append", default=[])
    provider_provision_parser.add_argument("--dry-run", action="store_true")
    provider_provision_parser.add_argument("--jobs-file", default=str(DEFAULT_JOBS_FILE))
    provider_provision_parser.add_argument("--registry-url")
    provider_provision_parser.add_argument("--heartbeat-url")
    provider_provision_parser.add_argument("--heartbeat-state-file")
    provider_provision_parser.add_argument("--repo-clone-url")
    provider_provision_parser.add_argument("--repo-branch")

    provider_jobs_parser = subparsers.add_parser(
        "providers-jobs",
        help="Inspect provider provisioning jobs from the JSON job store.",
    )
    provider_jobs_parser.add_argument("--jobs-file", default=str(DEFAULT_JOBS_FILE))
    provider_jobs_parser.add_argument("--job-id")
    provider_jobs_parser.add_argument("--status")

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
                ssh_user=args.ssh_user,
                ssh_port=args.ssh_port,
                repo_root=args.repo_root,
                allow_empty=True,
                include_capabilities=args.include_capabilities,
            )
        except (ProbeError, ValueError) as exc:
            print(f"probe-local error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(inventory.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "probe-remote-node":
        try:
            inventory = _probe_remote_node_inventory(
                node_id=args.node_id,
                host=args.host,
                ssh_user=args.ssh_user or DEFAULT_SSH_USER,
                ssh_port=args.ssh_port,
                repo_root=args.repo_root or DEFAULT_REPO_ROOT,
                lease_duration_seconds=args.lease_duration_seconds,
                cached_models=args.cached_model,
                labels=args.label,
                trust_tier=args.trust_tier,
                network_tier=args.network_tier,
            )
        except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "node_id": args.node_id,
                        "host": args.host,
                        "reason": str(exc),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        if args.state_file:
            store = RegistryStateStore(args.state_file)
            registry = store.load()
            registry.register_heartbeat(
                inventory,
                received_at=utc_now(),
                source="ssh-probe",
            )
            store.save(registry)
            print(
                json.dumps(
                    {
                        "status": "registered",
                        "state_file": str(Path(args.state_file)),
                        "inventory": inventory.to_dict(),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
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
        deployment_spec = _build_deployment_spec(args)
        request = deployment_spec.to_agent_request()
        profile = get_runtime_profile(deployment_spec.profile)
        try:
            profile = _apply_launch_profile_overrides(profile, deployment_spec)
        except ValueError as exc:
            print(
                json.dumps(
                    {
                        "deployment": deployment_spec.to_dict(),
                        "launch": {
                            "status": "blocked",
                            "reason": str(exc),
                        }
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        claim_summaries: list[dict[str, object]] = []
        if args.probe_remote_sessions and remote_nodes:
            try:
                remote_nodes, claim_summaries = _filter_nodes_by_remote_session_claims(
                    remote_nodes,
                    request_agent_id=request.agent_id,
                    ssh_user=args.ssh_user,
                    ssh_port=args.ssh_port,
                    repo_root=args.repo_root or DEFAULT_REPO_ROOT,
                    session_dir=args.session_dir,
                )
            except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
                print(
                    json.dumps(
                        {
                            "deployment": deployment_spec.to_dict(),
                            "launch": {
                                "status": "blocked",
                                "reason": f"Remote session probe failed: {exc}",
                            }
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 1
        decision = schedule_agent(local_node, remote_nodes, request)
        if not decision.is_placed:
            reason = decision.reason
            if claim_summaries:
                claim_text = ", ".join(
                    f"{item['node_id']}/gpu{item['gpu_index']} by {item['agent_id']}"
                    for item in claim_summaries
                )
                reason = f"No eligible GPU after excluding active remote sessions: {claim_text}."
            print(
                json.dumps(
                    {
                        "deployment": deployment_spec.to_dict(),
                        "placement": decision.to_dict(),
                        "launch": {
                            "status": "blocked",
                            "reason": reason,
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        try:
            selected_record = registry.get_record(decision.node_id or "")
            selected_node = selected_record.node if selected_record is not None else None
            resolved_ssh_user, resolved_ssh_port, resolved_repo_root = _resolve_node_access(
                selected_node,
                ssh_user=args.ssh_user,
                ssh_port=args.ssh_port,
                repo_root=args.repo_root,
            )
            result = launch_agent(
                request,
                decision,
                profile,
                dry_run=args.dry_run,
                ssh_user=resolved_ssh_user,
                ssh_port=resolved_ssh_port,
                repo_root=resolved_repo_root,
                min_lease_remaining_seconds=args.min_lease_remaining_seconds,
            )
        except ValueError as exc:
            print(
                json.dumps(
                    {
                        "deployment": deployment_spec.to_dict(),
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
                    "deployment": deployment_spec.to_dict(),
                    "placement": decision.to_dict(),
                    "launch": result.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if result.status in {"ready", "launched"} else 1

    if args.command == "list-remote-sessions":
        registry = _load_registry_from_args(args)
        nodes = registry.list_nodes()
        if args.node_id:
            nodes = [node for node in nodes if node.node_id == args.node_id]
        repo_root = args.repo_root or DEFAULT_REPO_ROOT
        results: list[dict[str, object]] = []
        for node in nodes:
            try:
                sessions, claims = _fetch_remote_session_claims_for_node(
                    node,
                    ssh_user=args.ssh_user,
                    ssh_port=args.ssh_port,
                    repo_root=repo_root,
                    session_dir=args.session_dir,
                )
                results.append(
                    {
                        "node_id": node.node_id,
                        "host": node.host,
                        "sessions": sessions,
                        "claims": [claim.to_dict() for claim in claims],
                    }
                )
            except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
                results.append(
                    {
                        "node_id": node.node_id,
                        "host": node.host,
                        "error": str(exc),
                    }
                )
        print(json.dumps({"nodes": results}, indent=2, sort_keys=True))
        return 0

    if args.command == "stop-remote-session":
        registry = _load_registry_from_args(args)
        record = registry.get_record(args.node_id)
        if record is None:
            print(
                json.dumps(
                    {
                        "status": "blocked",
                        "reason": f"Node id {args.node_id!r} is not present in the registry.",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        resolved_ssh_user, resolved_ssh_port, resolved_repo_root = _resolve_node_access(
            record.node,
            ssh_user=args.ssh_user,
            ssh_port=args.ssh_port,
            repo_root=args.repo_root,
        )
        command = build_remote_ssh_argv(
            host=record.node.host,
            remote_command=_build_remote_session_stop_command(args.session_dir, args.agent_id),
            ssh_user=resolved_ssh_user,
            ssh_port=resolved_ssh_port,
            repo_root=resolved_repo_root,
            gpu_index=None,
        )
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            payload = json.loads(completed.stdout)
        except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "node_id": args.node_id,
                        "agent_id": args.agent_id,
                        "reason": str(exc),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "providers-list-offers":
        service = ProviderService()
        try:
            offers = service.list_offers(
                provider=args.provider,
                gpu_name=args.gpu_name,
                min_gpu_count=args.min_gpu_count,
                min_vram_gb=args.min_vram_gb,
                max_price_hourly=args.max_price_hourly,
                region=args.region,
                preemptible_ok=True if args.preemptible_ok else None,
            )
        except ProviderError as exc:
            print(json.dumps({"status": "failed", "reason": str(exc)}, indent=2, sort_keys=True))
            return 1
        print(json.dumps({"offers": [offer.to_dict() for offer in offers]}, indent=2, sort_keys=True))
        return 0

    if args.command == "providers-list-blueprints":
        service = ProviderService()
        blueprints = service.list_blueprints(provider=args.provider, model_id=args.model_id)
        print(
            json.dumps(
                {"blueprints": [blueprint.to_dict() for blueprint in blueprints]},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "providers-provision":
        service = ProviderService(
            jobs_file=args.jobs_file,
            registry_url=args.registry_url,
            heartbeat_url=args.heartbeat_url,
            heartbeat_state_file=args.heartbeat_state_file,
            repo_clone_url=(
                args.repo_clone_url
                or "https://github.com/JohnnyDillinger-hub/Claude-Code-Game-Studios.git"
            ),
            repo_branch=args.repo_branch or "codex/mesh-runtime-tp4",
        )
        try:
            provision_request = _build_provision_request(args)
            job = service.provision(provision_request)
        except (ProviderError, ValueError) as exc:
            print(json.dumps({"status": "failed", "reason": str(exc)}, indent=2, sort_keys=True))
            return 1
        print(json.dumps(job.to_dict(), indent=2, sort_keys=True))
        return 0 if job.status != "failed" else 1

    if args.command == "providers-jobs":
        service = ProviderService(jobs_file=args.jobs_file)
        if args.job_id:
            job = service.get_job(args.job_id)
            if job is None:
                print(
                    json.dumps(
                        {"status": "not_found", "job_id": args.job_id},
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 1
            print(json.dumps(job.to_dict(), indent=2, sort_keys=True))
            return 0
        jobs = service.list_jobs(status=args.status)
        print(json.dumps({"jobs": [job.to_dict() for job in jobs]}, indent=2, sort_keys=True))
        return 0

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
