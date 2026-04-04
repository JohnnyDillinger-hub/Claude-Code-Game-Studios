from __future__ import annotations

from dataclasses import dataclass
import shlex
import subprocess
from typing import Any

from cluster.models import AgentRequest, PlacementDecision, utc_now
from cluster.orchestrator.model_profiles import RuntimeProfile


@dataclass(frozen=True, slots=True)
class LaunchResult:
    status: str
    mode: str
    command: str
    agent_id: str
    node_id: str | None
    reason: str
    executed: bool
    stdout: str | None = None
    stderr: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "mode": self.mode,
            "command": self.command,
            "agent_id": self.agent_id,
            "reason": self.reason,
            "executed": self.executed,
        }
        if self.node_id is not None:
            payload["node_id"] = self.node_id
        if self.stdout:
            payload["stdout"] = self.stdout
        if self.stderr:
            payload["stderr"] = self.stderr
        return payload


def ensure_launchable(
    decision: PlacementDecision,
    *,
    min_lease_remaining_seconds: int = 30,
) -> None:
    if not decision.is_placed:
        raise ValueError("Launch requires a placed PlacementDecision")
    if decision.available_until is None:
        raise ValueError("Launch requires available_until in PlacementDecision")
    remaining = (decision.available_until - utc_now()).total_seconds()
    if remaining <= 0:
        raise ValueError(f"Target node {decision.node_id} lease is already expired")
    if remaining < min_lease_remaining_seconds:
        raise ValueError(
            f"Target node {decision.node_id} lease only has {remaining:.1f}s left, "
            f"which is below the safety threshold of {min_lease_remaining_seconds}s"
        )


def build_worker_command(
    request: AgentRequest,
    decision: PlacementDecision,
    profile: RuntimeProfile,
) -> list[str]:
    if not decision.is_placed:
        raise ValueError("Cannot build a worker command for a rejected placement decision")
    gpu_index = decision.gpu_index if decision.gpu_index is not None else 0
    return [
        "python3",
        "-m",
        profile.launch_metadata["entrypoint_module"],
        "--agent-id",
        request.agent_id,
        "--model",
        profile.model_name,
        "--backend",
        profile.preferred_backend,
        "--runtime-class",
        profile.runtime_class,
        "--gpu-index",
        str(gpu_index),
        "--node-id",
        decision.node_id or "unknown-node",
    ]


def build_remote_ssh_command(
    *,
    host: str,
    remote_command: list[str] | str,
    ssh_user: str | None = None,
    ssh_port: int | None = None,
    repo_root: str,
    gpu_index: int,
) -> str:
    target = f"{ssh_user}@{host}" if ssh_user else host
    command_text = (
        remote_command if isinstance(remote_command, str) else shlex.join(remote_command)
    )
    repo_root_text = repo_root
    if repo_root.startswith("~/"):
        repo_root_text = "$HOME/" + repo_root[2:]
    remote_shell = (
        f"cd {repo_root_text} && "
        f"CUDA_VISIBLE_DEVICES={gpu_index} {command_text}"
    )
    parts = ["ssh"]
    if ssh_port is not None:
        parts.extend(["-p", str(ssh_port)])
    parts.append(target)
    parts.append(shlex.quote(remote_shell))
    return " ".join(parts)


def plan_launch(
    request: AgentRequest,
    decision: PlacementDecision,
    profile: RuntimeProfile,
    *,
    ssh_user: str | None = None,
    ssh_port: int | None = None,
    repo_root: str | None = None,
    min_lease_remaining_seconds: int = 30,
) -> LaunchResult:
    ensure_launchable(decision, min_lease_remaining_seconds=min_lease_remaining_seconds)
    worker_command = build_worker_command(request, decision, profile)
    repo_root = repo_root or str(profile.launch_metadata.get("repo_root_default", "."))

    if decision.source == "local":
        return LaunchResult(
            status="ready",
            mode="local",
            command=shlex.join(worker_command),
            agent_id=request.agent_id,
            node_id=decision.node_id,
            reason="Phase 2 local launch path prepared for a single-agent single-GPU worker.",
            executed=False,
        )

    if decision.host is None:
        raise ValueError("Remote launch planning requires a host in PlacementDecision")
    gpu_index = decision.gpu_index if decision.gpu_index is not None else 0
    ssh_command = build_remote_ssh_command(
        host=decision.host,
        remote_command=worker_command,
        ssh_user=ssh_user,
        ssh_port=ssh_port,
        repo_root=repo_root,
        gpu_index=gpu_index,
    )
    return LaunchResult(
        status="ready",
        mode="remote-ssh",
        command=ssh_command,
        agent_id=request.agent_id,
        node_id=decision.node_id,
        reason="Phase 2 remote SSH launch path prepared for a single-agent single-GPU worker.",
        executed=False,
    )


def launch_agent(
    request: AgentRequest,
    decision: PlacementDecision,
    profile: RuntimeProfile,
    *,
    dry_run: bool = True,
    ssh_user: str | None = None,
    ssh_port: int | None = None,
    repo_root: str | None = None,
    min_lease_remaining_seconds: int = 30,
) -> LaunchResult:
    plan = plan_launch(
        request,
        decision,
        profile,
        ssh_user=ssh_user,
        ssh_port=ssh_port,
        repo_root=repo_root,
        min_lease_remaining_seconds=min_lease_remaining_seconds,
    )
    if dry_run:
        return plan

    try:
        if plan.mode == "local":
            command = shlex.split(plan.command)
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
        else:
            completed = subprocess.run(
                plan.command,
                check=True,
                capture_output=True,
                text=True,
                shell=True,
            )
        return LaunchResult(
            status="launched",
            mode=plan.mode,
            command=plan.command,
            agent_id=request.agent_id,
            node_id=decision.node_id,
            reason="Launch completed.",
            executed=True,
            stdout=completed.stdout.strip() or None,
            stderr=completed.stderr.strip() or None,
        )
    except subprocess.CalledProcessError as exc:
        return LaunchResult(
            status="failed",
            mode=plan.mode,
            command=plan.command,
            agent_id=request.agent_id,
            node_id=decision.node_id,
            reason=f"Launch failed with exit code {exc.returncode}.",
            executed=True,
            stdout=(exc.stdout or "").strip() or None,
            stderr=(exc.stderr or "").strip() or None,
        )
