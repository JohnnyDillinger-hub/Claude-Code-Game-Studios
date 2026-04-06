from __future__ import annotations

import argparse
import csv
from datetime import timedelta
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
from typing import Iterable

from cluster.models import (
    GPUInventory,
    LeaseInfo,
    NodeAccess,
    NodeInventory,
    NodeSystemInfo,
    NodeTopology,
    RuntimeCapability,
    parse_datetime,
    utc_now,
)


class ProbeError(RuntimeError):
    """Raised when the local GPU probe cannot be completed."""


NVIDIA_SMI_QUERY = ",".join(
    [
        "index",
        "uuid",
        "memory.total",
        "memory.free",
        "utilization.gpu",
    ]
)

RUNTIME_PROBE_SPECS = (
    {
        "name": "ollama",
        "command": "ollama",
        "version_args": ["--version"],
        "supported_topologies": ("single-gpu",),
    },
    {
        "name": "vllm",
        "distribution": "vllm",
        "python_candidates": (".venv-vllm/bin/python", "python3"),
        "supported_topologies": ("single-gpu", "tp"),
    },
    {
        "name": "sglang",
        "distribution": "sglang",
        "python_candidates": (".venv-sglang/bin/python", "python3"),
        "supported_topologies": ("single-gpu", "tp", "dp"),
    },
    {
        "name": "tensorrt-llm",
        "distribution": "tensorrt_llm",
        "python_candidates": (".venv-trtllm/bin/python", "python3"),
        "supported_topologies": ("single-gpu", "tp", "pp"),
    },
    {
        "name": "deepspeed",
        "distribution": "deepspeed",
        "python_candidates": ("python3",),
        "supported_topologies": ("single-gpu", "tp", "pp"),
    },
    {
        "name": "transformers",
        "distribution": "transformers",
        "python_candidates": ("python3",),
        "supported_topologies": ("single-gpu", "tp"),
    },
)

CUDA_VERSION_PATTERN = re.compile(r"CUDA Version:\s*([0-9.]+)")


def _normalize_optional(value: str) -> str | None:
    normalized = value.strip()
    if normalized in {"", "N/A", "[Not Supported]"}:
        return None
    return normalized


def _normalize_optional_int(value: str) -> int | None:
    normalized = _normalize_optional(value)
    return int(normalized) if normalized is not None else None


def _read_command_text(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    text = (completed.stdout or completed.stderr or "").strip()
    return text or None


def _expand_path_text(value: str | Path) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(str(value))))


def _resolve_python_candidate(candidate: str, repo_root: Path) -> str | None:
    if "/" in candidate:
        resolved = repo_root / candidate
        return str(resolved) if resolved.exists() else None
    executable = shutil.which(candidate)
    return executable or None


def _probe_distribution_version(distribution: str, python_executable: str) -> str | None:
    script = (
        "import importlib.metadata as md; "
        f"print(md.version({distribution!r}))"
    )
    return _read_command_text([python_executable, "-c", script])


def probe_system_info(
    *,
    nvidia_smi_bin: str = "nvidia-smi",
    python_executable: str = "python3",
) -> NodeSystemInfo:
    hostname = socket.gethostname()
    python_version = _read_command_text([python_executable, "--version"])
    driver_version = None
    cuda_version = None
    if shutil.which(nvidia_smi_bin) is not None:
        driver_version = _read_command_text(
            [
                nvidia_smi_bin,
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        if driver_version is not None:
            driver_version = driver_version.splitlines()[0].strip()
        nvidia_smi_text = _read_command_text([nvidia_smi_bin])
        if nvidia_smi_text is not None:
            match = CUDA_VERSION_PATTERN.search(nvidia_smi_text)
            if match:
                cuda_version = match.group(1)
    return NodeSystemInfo(
        hostname=hostname,
        python_version=python_version,
        driver_version=driver_version,
        cuda_version=cuda_version,
    )


def probe_runtime_capabilities(*, repo_root: str | Path | None = None) -> tuple[RuntimeCapability, ...]:
    root_path = _expand_path_text(repo_root) if repo_root is not None else Path.cwd()
    capabilities: list[RuntimeCapability] = []
    for spec in RUNTIME_PROBE_SPECS:
        if "command" in spec:
            executable = shutil.which(str(spec["command"]))
            version = (
                _read_command_text([executable, *spec["version_args"]])
                if executable is not None
                else None
            )
            capabilities.append(
                RuntimeCapability(
                    name=str(spec["name"]),
                    installed=executable is not None,
                    version=version,
                    executable=executable,
                    supported_topologies=tuple(str(item) for item in spec["supported_topologies"]),
                )
            )
            continue

        resolved_python = None
        version = None
        for candidate in spec.get("python_candidates", ("python3",)):
            resolved_python = _resolve_python_candidate(str(candidate), root_path)
            if resolved_python is None:
                continue
            version = _probe_distribution_version(str(spec["distribution"]), resolved_python)
            if version is not None:
                break
        capabilities.append(
            RuntimeCapability(
                name=str(spec["name"]),
                installed=version is not None,
                version=version,
                executable=resolved_python if version is not None else None,
                supported_topologies=tuple(str(item) for item in spec["supported_topologies"]),
            )
        )
    return tuple(capabilities)


def probe_topology(gpus: Iterable[GPUInventory], *, nvidia_smi_bin: str = "nvidia-smi") -> NodeTopology:
    gpu_items = tuple(gpus)
    if len(gpu_items) < 2:
        return NodeTopology(single_node_multi_gpu=False)
    interconnect = "pcie"
    if shutil.which(nvidia_smi_bin) is not None:
        topo_text = _read_command_text([nvidia_smi_bin, "topo", "-m"])
        if topo_text and re.search(r"\bNV\d+\b", topo_text):
            interconnect = "nvlink"
    return NodeTopology(
        single_node_multi_gpu=True,
        interconnect=interconnect,
    )


def probe_gpus(nvidia_smi_bin: str = "nvidia-smi", allow_empty: bool = True) -> list[GPUInventory]:
    if shutil.which(nvidia_smi_bin) is None:
        if allow_empty:
            return []
        raise ProbeError("nvidia-smi is not available on PATH")

    cmd = [
        nvidia_smi_bin,
        f"--query-gpu={NVIDIA_SMI_QUERY}",
        "--format=csv,noheader,nounits",
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        combined = "\n".join(part for part in [stdout, stderr] if part)
        if allow_empty and "No devices were found" in combined:
            return []
        raise ProbeError(f"nvidia-smi probe failed: {combined or exc}") from exc

    rows = [row for row in csv.reader(result.stdout.splitlines()) if row]
    gpus: list[GPUInventory] = []
    for row in rows:
        if len(row) < 5:
            raise ProbeError(f"Unexpected nvidia-smi row: {row!r}")
        gpus.append(
            GPUInventory(
                index=int(row[0].strip()),
                uuid=_normalize_optional(row[1]),
                total_memory_mib=int(row[2].strip()),
                free_memory_mib=int(row[3].strip()),
                utilization_pct=_normalize_optional_int(row[4]),
            )
        )
    return gpus


def build_local_inventory(
    *,
    node_id: str,
    host: str | None = None,
    available_until=None,
    lease_duration_seconds: int | None = None,
    cached_models: Iterable[str] = (),
    labels: dict[str, str] | None = None,
    trust_tier: str | None = None,
    network_tier: str | None = None,
    ssh_user: str | None = None,
    ssh_port: int | None = None,
    repo_root: str | None = None,
    allow_empty: bool = True,
    include_capabilities: bool = False,
) -> NodeInventory:
    if available_until is None:
        duration = lease_duration_seconds if lease_duration_seconds is not None else 3600
        available_until = utc_now() + timedelta(seconds=duration)

    gpus = tuple(probe_gpus(allow_empty=allow_empty))
    runtime_capabilities = probe_runtime_capabilities(repo_root=repo_root) if include_capabilities else ()
    system_info = probe_system_info() if include_capabilities else None
    topology = probe_topology(gpus) if include_capabilities else None
    return NodeInventory(
        node_id=node_id,
        host=host or socket.gethostname(),
        lease=LeaseInfo(
            available_until=available_until,
            lease_duration_seconds=lease_duration_seconds,
            source="node-agent-probe",
        ),
        gpus=gpus,
        cached_models=tuple(sorted(set(cached_models))),
        labels=tuple(sorted((labels or {}).items())),
        trust_tier=trust_tier,
        network_tier=network_tier,
        access=NodeAccess(
            ssh_user=ssh_user,
            ssh_port=ssh_port,
            repo_root=repo_root,
        )
        if any(value is not None for value in (ssh_user, ssh_port, repo_root))
        else None,
        runtime_capabilities=runtime_capabilities,
        system_info=system_info,
        topology=topology,
    )


def parse_label_items(values: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected label in key=value form, got: {item}")
        key, value = item.split("=", 1)
        labels[key.strip()] = value.strip()
    return labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe local GPUs into a NodeInventory.")
    parser.add_argument("--node-id", default="local-node", help="Stable node identifier.")
    parser.add_argument("--host", default=socket.gethostname(), help="Advertised host value.")
    parser.add_argument(
        "--available-until",
        help="Absolute availability timestamp in ISO-8601 with timezone.",
    )
    parser.add_argument(
        "--lease-duration-seconds",
        type=int,
        default=3600,
        help="Relative lease duration when --available-until is omitted.",
    )
    parser.add_argument(
        "--cached-model",
        action="append",
        default=[],
        help="Repeat to describe cached models on the node.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Repeat key=value labels to attach to the node.",
    )
    parser.add_argument("--trust-tier", help="Optional trust classification.")
    parser.add_argument("--network-tier", help="Optional network classification.")
    parser.add_argument("--ssh-user", help="Optional SSH user for control-plane access.")
    parser.add_argument("--ssh-port", type=int, help="Optional SSH port for control-plane access.")
    parser.add_argument("--repo-root", help="Optional repository root on the remote node.")
    parser.add_argument(
        "--no-allow-empty",
        action="store_true",
        help="Fail instead of returning an empty GPU list when no GPU is available.",
    )
    parser.add_argument(
        "--include-capabilities",
        action="store_true",
        help="Include runtime/system/topology capability snapshots in the inventory.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        available_until = None if args.available_until is None else parse_datetime(args.available_until)
        inventory = build_local_inventory(
            node_id=args.node_id,
            host=args.host,
            available_until=available_until,
            lease_duration_seconds=args.lease_duration_seconds,
            cached_models=args.cached_model,
            labels=parse_label_items(args.label),
            trust_tier=args.trust_tier,
            network_tier=args.network_tier,
            ssh_user=args.ssh_user,
            ssh_port=args.ssh_port,
            repo_root=args.repo_root,
            allow_empty=not args.no_allow_empty,
            include_capabilities=args.include_capabilities,
        )
    except (ProbeError, ValueError) as exc:
        print(f"probe error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(inventory.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
