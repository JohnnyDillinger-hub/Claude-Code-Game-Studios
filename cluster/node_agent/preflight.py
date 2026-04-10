from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import secrets

from cluster.models import NodeInventory, utc_now
from cluster.node_agent.preflight_models import (
    NodePreflightReport,
    PreflightCheck,
    RuntimeRequirement,
    RuntimeRequirementReport,
)
from cluster.orchestrator.model_profiles import RuntimeProfile, get_runtime_profile


_AUTO_REPAIRABLE_RUNTIMES = {"vllm", "sglang", "deepspeed", "tensorrt-llm", "ollama"}


def build_node_preflight_report(
    node: NodeInventory,
    *,
    provider: str | None = None,
    runtime_stack: Iterable[str] = (),
    preferred_launch_profile: str | None = None,
    generated_at: datetime | None = None,
) -> NodePreflightReport:
    observed_at = generated_at or utc_now()
    targets = _resolve_target_runtimes(node, runtime_stack, preferred_launch_profile)
    checks = _build_system_checks(node)
    runtime_reports = tuple(
        _build_runtime_report(
            node,
            runtime_name=runtime_name,
            preferred_launch_profile=preferred_launch_profile,
        )
        for runtime_name in targets
    )
    status = _summarize_report_status(checks, runtime_reports)
    summary = _summarize_report_text(status, checks, runtime_reports)
    timestamp = observed_at.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")
    return NodePreflightReport(
        report_id=f"preflight-{node.node_id}-{timestamp}-{secrets.token_hex(3)}",
        node_id=node.node_id,
        provider=provider,
        generated_at=observed_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        status=status,
        summary=summary,
        inventory_snapshot=node.to_dict(),
        checks=checks,
        runtime_reports=runtime_reports,
    )


def _resolve_target_runtimes(
    node: NodeInventory,
    runtime_stack: Iterable[str],
    preferred_launch_profile: str | None,
) -> tuple[str, ...]:
    targets: list[str] = []
    for runtime_name in runtime_stack:
        normalized = str(runtime_name).strip()
        if normalized and normalized not in targets:
            targets.append(normalized)
    if preferred_launch_profile:
        profile = _safe_get_profile(preferred_launch_profile)
        if profile is not None:
            runtime_name = _runtime_capability_name_for_profile(profile)
            if runtime_name not in targets:
                targets.append(runtime_name)
    if not targets:
        for capability in node.runtime_capabilities:
            if capability.installed and capability.name not in targets:
                targets.append(capability.name)
    return tuple(targets)


def _build_system_checks(node: NodeInventory) -> tuple[PreflightCheck, ...]:
    checks: list[PreflightCheck] = []
    repo_root = node.access.repo_root if node.access is not None else None
    checks.append(
        PreflightCheck(
            check_id="repo_root_known",
            category="workspace",
            status="pass" if repo_root else "warn",
            severity="info" if repo_root else "warning",
            detected_value=repo_root,
            required_value="known repo root",
            auto_repairable=bool(repo_root is None),
            message=(
                f"Repo root recorded as {repo_root}."
                if repo_root
                else "Node access metadata does not yet advertise a repo root."
            ),
        )
    )
    driver_version = node.system_info.driver_version if node.system_info is not None else None
    checks.append(
        PreflightCheck(
            check_id="driver_nvml_ok",
            category="system",
            status="pass" if driver_version else "warn",
            severity="info" if driver_version else "warning",
            detected_value=driver_version or "missing",
            required_value="loaded NVIDIA driver",
            auto_repairable=False,
            message=(
                f"Detected NVIDIA driver {driver_version}."
                if driver_version
                else "No NVIDIA driver version was reported by the node inventory."
            ),
        )
    )
    cuda_version = node.system_info.cuda_version if node.system_info is not None else None
    checks.append(
        PreflightCheck(
            check_id="cuda_runtime_present",
            category="cuda",
            status="pass" if cuda_version else "warn",
            severity="info" if cuda_version else "warning",
            detected_value=cuda_version or "missing",
            required_value="CUDA runtime visible via nvidia-smi",
            auto_repairable=False,
            message=(
                f"Detected CUDA runtime {cuda_version}."
                if cuda_version
                else "CUDA runtime version is missing from the node inventory."
            ),
        )
    )
    python_version = node.system_info.python_version if node.system_info is not None else None
    checks.append(
        PreflightCheck(
            check_id="python_ok",
            category="system",
            status="pass" if python_version else "warn",
            severity="info" if python_version else "warning",
            detected_value=python_version or "missing",
            required_value="python3 available",
            auto_repairable=bool(python_version is None),
            message=(
                f"Detected {python_version}."
                if python_version
                else "Python version was not advertised by the node inventory."
            ),
        )
    )
    checks.append(
        PreflightCheck(
            check_id="gpu_inventory_present",
            category="hardware",
            status="pass" if node.gpus else "warn",
            severity="info" if node.gpus else "warning",
            detected_value=str(len(node.gpus)),
            required_value="at least one GPU",
            auto_repairable=False,
            message=(
                f"Node reports {len(node.gpus)} GPU(s)."
                if node.gpus
                else "Node inventory does not list any GPUs."
            ),
        )
    )
    return tuple(checks)


def _build_runtime_report(
    node: NodeInventory,
    *,
    runtime_name: str,
    preferred_launch_profile: str | None,
) -> RuntimeRequirementReport:
    profile = _resolve_matching_profile(runtime_name, preferred_launch_profile)
    capability = next((item for item in node.runtime_capabilities if item.name == runtime_name), None)
    requirements: list[RuntimeRequirement] = []

    requirements.append(
        RuntimeRequirement(
            key="installed",
            status="pass" if capability is not None and capability.installed else "fail",
            detected_value=(
                capability.version
                or capability.executable
                or "installed"
                if capability is not None and capability.installed
                else "missing"
            ),
            required_value="installed runtime",
            auto_repairable=runtime_name in _AUTO_REPAIRABLE_RUNTIMES,
            message=(
                f"Detected {runtime_name} runtime."
                if capability is not None and capability.installed
                else f"{runtime_name} runtime is missing on this node."
            ),
        )
    )

    if profile is not None:
        required_topology = _required_topology_for_profile(profile)
        requirements.append(
            RuntimeRequirement(
                key="topology",
                status=(
                    "skip"
                    if capability is None or not capability.installed
                    else "pass"
                    if required_topology in capability.supported_topologies
                    else "fail"
                ),
                detected_value=(
                    ",".join(capability.supported_topologies)
                    if capability is not None and capability.supported_topologies
                    else "unknown"
                ),
                required_value=required_topology,
                auto_repairable=capability is None or not capability.installed,
                message=(
                    f"Runtime advertises topology {required_topology}."
                    if capability is not None
                    and capability.installed
                    and required_topology in capability.supported_topologies
                    else f"Topology check for {required_topology} will run after runtime installation."
                    if capability is None or not capability.installed
                    else f"Runtime does not advertise required topology {required_topology}."
                ),
            )
        )
        requirements.append(
            RuntimeRequirement(
                key="gpu_count",
                status="pass" if node.gpu_count >= profile.required_gpu_count else "fail",
                detected_value=str(node.gpu_count),
                required_value=str(profile.required_gpu_count),
                auto_repairable=False,
                message=(
                    f"Node exposes {node.gpu_count} GPU(s), meeting profile requirement."
                    if node.gpu_count >= profile.required_gpu_count
                    else f"Profile requires {profile.required_gpu_count} GPU(s) but node only exposes {node.gpu_count}."
                ),
            )
        )
        selected_gpus = sorted(
            node.gpus,
            key=lambda gpu: gpu.free_memory_mib,
            reverse=True,
        )[: profile.required_gpu_count]
        min_free_vram = min((gpu.free_memory_mib for gpu in selected_gpus), default=0)
        requirements.append(
            RuntimeRequirement(
                key="free_vram_mib",
                status=(
                    "pass"
                    if len(selected_gpus) >= profile.required_gpu_count
                    and min_free_vram >= profile.required_free_vram_mib
                    else "fail"
                ),
                detected_value=str(min_free_vram),
                required_value=str(profile.required_free_vram_mib),
                auto_repairable=False,
                message=(
                    f"Minimum free VRAM across selected GPUs is {min_free_vram} MiB."
                    if len(selected_gpus) >= profile.required_gpu_count
                    else "Not enough GPUs were available to evaluate free VRAM."
                ),
            )
        )

    if runtime_name == "deepspeed" and capability is not None and capability.installed:
        details = capability.details
        packaged_nvcc = bool(
            details.get("packaged_nvcc_present") or details.get("runtime_nvcc_present")
        )
        system_nvcc = bool(details.get("system_nvcc_present"))
        requirements.append(
            RuntimeRequirement(
                key="nvcc",
                status="pass" if packaged_nvcc or system_nvcc else "fail",
                detected_value=(
                    "packaged"
                    if packaged_nvcc
                    else "system"
                    if system_nvcc
                    else "missing"
                ),
                required_value="packaged or system nvcc",
                auto_repairable=True,
                message=(
                    "DeepSpeed has access to nvcc."
                    if packaged_nvcc or system_nvcc
                    else "DeepSpeed currently lacks nvcc; this is the known auto-repair path."
                ),
            )
        )

    failures = [item for item in requirements if item.status == "fail"]
    if not failures:
        status = "ready"
        available = True
        repairable = False
    elif all(item.auto_repairable for item in failures):
        status = "repairable"
        available = False
        repairable = True
    else:
        status = "failed"
        available = False
        repairable = False

    return RuntimeRequirementReport(
        runtime=runtime_name,
        profile=profile.name if profile is not None else None,
        model_id=profile.model_name if profile is not None else None,
        topology=_profile_topology(profile),
        status=status,
        available=available,
        repairable=repairable,
        detected_version=capability.version if capability is not None else None,
        target_version=None,
        requirements=tuple(requirements),
        available_config=dict(profile.runtime_options) if profile is not None and available else {},
        warnings=(),
    )


def _profile_topology(profile: RuntimeProfile | None) -> dict[str, int]:
    if profile is None:
        return {}
    topology: dict[str, int] = {
        "required_gpu_count": profile.required_gpu_count,
    }
    tp_size = profile.runtime_options.get("tensor_parallel_size")
    pp_size = profile.runtime_options.get("pipeline_parallel_size")
    if tp_size is not None:
        topology["tensor_parallel_size"] = int(tp_size)
    elif profile.required_gpu_count > 1 and _required_topology_for_profile(profile) == "tp":
        topology["tensor_parallel_size"] = int(profile.required_gpu_count)
    if pp_size is not None:
        topology["pipeline_parallel_size"] = int(pp_size)
    return topology


def _summarize_report_status(
    checks: tuple[PreflightCheck, ...],
    runtime_reports: tuple[RuntimeRequirementReport, ...],
) -> str:
    check_failures = [item for item in checks if item.status == "fail"]
    runtime_failures = [item for item in runtime_reports if item.status == "failed"]
    runtime_repairs = [item for item in runtime_reports if item.status == "repairable"]
    if check_failures or runtime_failures:
        return "failed"
    if runtime_repairs:
        return "repairable"
    return "ready"


def _summarize_report_text(
    status: str,
    checks: tuple[PreflightCheck, ...],
    runtime_reports: tuple[RuntimeRequirementReport, ...],
) -> str:
    if status == "ready":
        runtimes = ", ".join(report.runtime for report in runtime_reports if report.available)
        if runtimes:
            return f"Node passed preflight and is ready for runtimes: {runtimes}."
        return "Node passed preflight."
    if status == "repairable":
        pending = ", ".join(report.runtime for report in runtime_reports if report.status == "repairable")
        return f"Node joined successfully, but runtime repair is still needed for: {pending}."
    first_failure = next((item for item in checks if item.status == "fail"), None)
    if first_failure is not None:
        return first_failure.message or "Node preflight failed on a system check."
    first_runtime = next((item for item in runtime_reports if item.status == "failed"), None)
    if first_runtime is not None:
        return f"Runtime preflight failed for {first_runtime.runtime}."
    return "Node preflight failed."


def _resolve_matching_profile(
    runtime_name: str,
    preferred_launch_profile: str | None,
) -> RuntimeProfile | None:
    if not preferred_launch_profile:
        return None
    profile = _safe_get_profile(preferred_launch_profile)
    if profile is None:
        return None
    if _runtime_capability_name_for_profile(profile) != runtime_name:
        return None
    return profile


def _safe_get_profile(profile_name: str) -> RuntimeProfile | None:
    try:
        return get_runtime_profile(profile_name)
    except (KeyError, ValueError):
        return None


def _runtime_capability_name_for_profile(profile: RuntimeProfile) -> str:
    mapping = {
        "ollama-server": "ollama",
        "vllm-server": "vllm",
        "sglang-server": "sglang",
        "trtllm-server": "tensorrt-llm",
        "deepspeed-server": "deepspeed",
        "python-hf-probe": "transformers",
    }
    return mapping.get(profile.runtime_adapter, profile.preferred_backend)


def _required_topology_for_profile(profile: RuntimeProfile) -> str:
    if profile.required_gpu_count <= 1:
        return "single-gpu"
    pipeline_parallel_size = int(profile.runtime_options.get("pipeline_parallel_size", 1) or 1)
    if pipeline_parallel_size > 1:
        return "pp"
    return "tp"
