from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any, Iterable


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def parse_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp must include timezone information: {value}")
    return parsed.astimezone(timezone.utc)


def format_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sorted_labels(labels: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    if not labels:
        return ()
    return tuple(sorted((str(key), str(val)) for key, val in labels.items()))


def _labels_dict(labels: Iterable[tuple[str, str]]) -> dict[str, str]:
    return {key: value for key, value in labels}


def _string_tuple(values: Iterable[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(value) for value in values)


CUDA_GRAPH_MODE_VALUES = ("profile-default", "enabled", "disabled")


@dataclass(frozen=True, slots=True)
class NodeAccess:
    ssh_user: str | None = None
    ssh_port: int | None = None
    repo_root: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.ssh_user is not None:
            payload["ssh_user"] = self.ssh_user
        if self.ssh_port is not None:
            payload["ssh_port"] = self.ssh_port
        if self.repo_root is not None:
            payload["repo_root"] = self.repo_root
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NodeAccess":
        return cls(
            ssh_user=str(payload["ssh_user"]) if payload.get("ssh_user") else None,
            ssh_port=int(payload["ssh_port"]) if payload.get("ssh_port") is not None else None,
            repo_root=str(payload["repo_root"]) if payload.get("repo_root") else None,
        )


@dataclass(frozen=True, slots=True)
class RuntimeCapability:
    name: str
    installed: bool
    version: str | None = None
    executable: str | None = None
    supported_topologies: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "installed": self.installed,
        }
        if self.version is not None:
            payload["version"] = self.version
        if self.executable is not None:
            payload["executable"] = self.executable
        if self.supported_topologies:
            payload["supported_topologies"] = list(self.supported_topologies)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeCapability":
        return cls(
            name=str(payload["name"]),
            installed=bool(payload["installed"]),
            version=str(payload["version"]) if payload.get("version") else None,
            executable=str(payload["executable"]) if payload.get("executable") else None,
            supported_topologies=_string_tuple(payload.get("supported_topologies")),
        )


@dataclass(frozen=True, slots=True)
class NodeSystemInfo:
    hostname: str | None = None
    python_version: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.hostname is not None:
            payload["hostname"] = self.hostname
        if self.python_version is not None:
            payload["python_version"] = self.python_version
        if self.driver_version is not None:
            payload["driver_version"] = self.driver_version
        if self.cuda_version is not None:
            payload["cuda_version"] = self.cuda_version
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NodeSystemInfo":
        return cls(
            hostname=str(payload["hostname"]) if payload.get("hostname") else None,
            python_version=(
                str(payload["python_version"]) if payload.get("python_version") else None
            ),
            driver_version=(
                str(payload["driver_version"]) if payload.get("driver_version") else None
            ),
            cuda_version=str(payload["cuda_version"]) if payload.get("cuda_version") else None,
        )


@dataclass(frozen=True, slots=True)
class NodeTopology:
    single_node_multi_gpu: bool
    interconnect: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "single_node_multi_gpu": self.single_node_multi_gpu,
        }
        if self.interconnect is not None:
            payload["interconnect"] = self.interconnect
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NodeTopology":
        return cls(
            single_node_multi_gpu=bool(payload.get("single_node_multi_gpu", False)),
            interconnect=str(payload["interconnect"]) if payload.get("interconnect") else None,
        )


@dataclass(frozen=True, slots=True)
class LeaseInfo:
    available_until: datetime
    lease_duration_seconds: int | None = None
    source: str | None = None

    def is_expired(self, now: datetime | None = None) -> bool:
        return self.available_until <= (now or utc_now())

    def remaining_seconds(self, now: datetime | None = None) -> float:
        delta = self.available_until - (now or utc_now())
        return max(delta.total_seconds(), 0.0)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "available_until": format_datetime(self.available_until),
        }
        if self.lease_duration_seconds is not None:
            payload["lease_duration_seconds"] = self.lease_duration_seconds
        if self.source is not None:
            payload["source"] = self.source
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LeaseInfo":
        if "available_until" not in payload:
            raise ValueError("LeaseInfo requires available_until")
        return cls(
            available_until=parse_datetime(str(payload["available_until"])),
            lease_duration_seconds=(
                int(payload["lease_duration_seconds"])
                if payload.get("lease_duration_seconds") is not None
                else None
            ),
            source=str(payload["source"]) if payload.get("source") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class GPUInventory:
    index: int
    uuid: str | None
    total_memory_mib: int
    free_memory_mib: int
    utilization_pct: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "total_memory_mib": self.total_memory_mib,
            "free_memory_mib": self.free_memory_mib,
        }
        if self.uuid is not None:
            payload["uuid"] = self.uuid
        if self.utilization_pct is not None:
            payload["utilization_pct"] = self.utilization_pct
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GPUInventory":
        return cls(
            index=int(payload["index"]),
            uuid=str(payload["uuid"]) if payload.get("uuid") not in (None, "") else None,
            total_memory_mib=int(payload["total_memory_mib"]),
            free_memory_mib=int(payload["free_memory_mib"]),
            utilization_pct=(
                int(payload["utilization_pct"])
                if payload.get("utilization_pct") not in (None, "")
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class NodeInventory:
    node_id: str
    host: str
    lease: LeaseInfo
    gpus: tuple[GPUInventory, ...] = field(default_factory=tuple)
    cached_models: tuple[str, ...] = field(default_factory=tuple)
    labels: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    trust_tier: str | None = None
    network_tier: str | None = None
    access: NodeAccess | None = None
    runtime_capabilities: tuple[RuntimeCapability, ...] = field(default_factory=tuple)
    system_info: NodeSystemInfo | None = None
    topology: NodeTopology | None = None

    @property
    def available_until(self) -> datetime:
        return self.lease.available_until

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def labels_map(self) -> dict[str, str]:
        return _labels_dict(self.labels)

    def is_expired(self, now: datetime | None = None) -> bool:
        return self.lease.is_expired(now=now)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "node_id": self.node_id,
            "host": self.host,
            "available_until": format_datetime(self.available_until),
            "gpu_count": self.gpu_count,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "cached_models": list(self.cached_models),
        }
        if self.labels:
            payload["labels"] = self.labels_map
        if self.trust_tier is not None:
            payload["trust_tier"] = self.trust_tier
        if self.network_tier is not None:
            payload["network_tier"] = self.network_tier
        if self.access is not None:
            payload["access"] = self.access.to_dict()
        if self.runtime_capabilities:
            payload["runtime_capabilities"] = [
                capability.to_dict() for capability in self.runtime_capabilities
            ]
        if self.system_info is not None:
            payload["system_info"] = self.system_info.to_dict()
        if self.topology is not None:
            payload["topology"] = self.topology.to_dict()
        if self.lease.lease_duration_seconds is not None or self.lease.source is not None:
            payload["lease_info"] = self.lease.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NodeInventory":
        gpus = tuple(GPUInventory.from_dict(item) for item in payload.get("gpus", []))
        declared_gpu_count = payload.get("gpu_count")
        if declared_gpu_count is not None and int(declared_gpu_count) != len(gpus):
            raise ValueError(
                "node inventory gpu_count does not match the number of GPU records"
            )

        lease_payload = payload.get("lease_info")
        if lease_payload is None:
            lease_payload = {"available_until": payload["available_until"]}
        elif "available_until" not in lease_payload and payload.get("available_until"):
            lease_payload = dict(lease_payload)
            lease_payload["available_until"] = payload["available_until"]

        return cls(
            node_id=str(payload["node_id"]),
            host=str(payload["host"]),
            lease=LeaseInfo.from_dict(lease_payload),
            gpus=gpus,
            cached_models=tuple(str(item) for item in payload.get("cached_models", [])),
            labels=_sorted_labels(payload.get("labels")),
            trust_tier=(
                str(payload["trust_tier"])
                if payload.get("trust_tier") is not None
                else None
            ),
            network_tier=(
                str(payload["network_tier"])
                if payload.get("network_tier") is not None
                else None
            ),
            access=(
                NodeAccess.from_dict(payload["access"])
                if isinstance(payload.get("access"), Mapping)
                else None
            ),
            runtime_capabilities=tuple(
                RuntimeCapability.from_dict(item)
                for item in payload.get("runtime_capabilities", [])
            ),
            system_info=(
                NodeSystemInfo.from_dict(payload["system_info"])
                if isinstance(payload.get("system_info"), Mapping)
                else None
            ),
            topology=(
                NodeTopology.from_dict(payload["topology"])
                if isinstance(payload.get("topology"), Mapping)
                else None
            ),
        )

    def with_gpus(self, gpus: Iterable[GPUInventory]) -> "NodeInventory":
        return NodeInventory(
            node_id=self.node_id,
            host=self.host,
            lease=self.lease,
            gpus=tuple(gpus),
            cached_models=self.cached_models,
            labels=self.labels,
            trust_tier=self.trust_tier,
            network_tier=self.network_tier,
            access=self.access,
            runtime_capabilities=self.runtime_capabilities,
            system_info=self.system_info,
            topology=self.topology,
        )


@dataclass(frozen=True, slots=True)
class AgentRequest:
    agent_id: str
    required_vram_mib: int
    required_gpu_count: int = 1
    model_id: str | None = None
    labels: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    trust_tier: str | None = None
    network_tier: str | None = None

    @property
    def labels_map(self) -> dict[str, str]:
        return _labels_dict(self.labels)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "agent_id": self.agent_id,
            "required_vram_mib": self.required_vram_mib,
        }
        if self.required_gpu_count != 1:
            payload["required_gpu_count"] = self.required_gpu_count
        if self.model_id is not None:
            payload["model_id"] = self.model_id
        if self.labels:
            payload["labels"] = self.labels_map
        if self.trust_tier is not None:
            payload["trust_tier"] = self.trust_tier
        if self.network_tier is not None:
            payload["network_tier"] = self.network_tier
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AgentRequest":
        return cls(
            agent_id=str(payload["agent_id"]),
            required_vram_mib=int(payload["required_vram_mib"]),
            required_gpu_count=int(payload.get("required_gpu_count", 1)),
            model_id=str(payload["model_id"]) if payload.get("model_id") else None,
            labels=_sorted_labels(payload.get("labels")),
            trust_tier=(
                str(payload["trust_tier"])
                if payload.get("trust_tier") is not None
                else None
            ),
            network_tier=(
                str(payload["network_tier"])
                if payload.get("network_tier") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimeLaunchPreferences:
    cuda_graph_mode: str | None = None
    cuda_graph_max_bs: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.cuda_graph_mode is not None:
            payload["cuda_graph_mode"] = self.cuda_graph_mode
        if self.cuda_graph_max_bs is not None:
            payload["cuda_graph_max_bs"] = self.cuda_graph_max_bs
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeLaunchPreferences":
        cuda_graph_mode = (
            str(payload["cuda_graph_mode"]) if payload.get("cuda_graph_mode") is not None else None
        )
        if cuda_graph_mode is not None and cuda_graph_mode not in CUDA_GRAPH_MODE_VALUES:
            raise ValueError(
                f"Unsupported cuda_graph_mode {cuda_graph_mode!r}; expected one of {CUDA_GRAPH_MODE_VALUES}"
            )
        return cls(
            cuda_graph_mode=cuda_graph_mode,
            cuda_graph_max_bs=(
                int(payload["cuda_graph_max_bs"])
                if payload.get("cuda_graph_max_bs") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class AgentDeploymentSpec:
    agent_id: str
    profile: str
    required_vram_mib: int
    required_gpu_count: int = 1
    model_id: str | None = None
    labels: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    trust_tier: str | None = None
    network_tier: str | None = None
    launch_preferences: RuntimeLaunchPreferences | None = None

    @property
    def labels_map(self) -> dict[str, str]:
        return _labels_dict(self.labels)

    def to_agent_request(self) -> AgentRequest:
        return AgentRequest(
            agent_id=self.agent_id,
            required_vram_mib=self.required_vram_mib,
            required_gpu_count=self.required_gpu_count,
            model_id=self.model_id,
            labels=self.labels,
            trust_tier=self.trust_tier,
            network_tier=self.network_tier,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "agent_id": self.agent_id,
            "profile": self.profile,
            "required_vram_mib": self.required_vram_mib,
        }
        if self.required_gpu_count != 1:
            payload["required_gpu_count"] = self.required_gpu_count
        if self.model_id is not None:
            payload["model_id"] = self.model_id
        if self.labels:
            payload["labels"] = self.labels_map
        if self.trust_tier is not None:
            payload["trust_tier"] = self.trust_tier
        if self.network_tier is not None:
            payload["network_tier"] = self.network_tier
        if self.launch_preferences is not None:
            payload["launch_preferences"] = self.launch_preferences.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AgentDeploymentSpec":
        return cls(
            agent_id=str(payload["agent_id"]),
            profile=str(payload["profile"]),
            required_vram_mib=int(payload["required_vram_mib"]),
            required_gpu_count=int(payload.get("required_gpu_count", 1)),
            model_id=str(payload["model_id"]) if payload.get("model_id") else None,
            labels=_sorted_labels(payload.get("labels")),
            trust_tier=(
                str(payload["trust_tier"])
                if payload.get("trust_tier") is not None
                else None
            ),
            network_tier=(
                str(payload["network_tier"])
                if payload.get("network_tier") is not None
                else None
            ),
            launch_preferences=(
                RuntimeLaunchPreferences.from_dict(payload["launch_preferences"])
                if payload.get("launch_preferences")
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class PlacementDecision:
    status: str
    reason: str
    agent_id: str
    node_id: str | None = None
    host: str | None = None
    gpu_index: int | None = None
    gpu_indices: tuple[int, ...] = field(default_factory=tuple)
    gpu_uuid: str | None = None
    gpu_uuids: tuple[str, ...] = field(default_factory=tuple)
    source: str | None = None
    model_cached: bool = False
    required_vram_mib: int | None = None
    available_vram_mib: int | None = None
    available_until: datetime | None = None

    @property
    def is_placed(self) -> bool:
        return self.status == "placed"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "reason": self.reason,
            "agent_id": self.agent_id,
        }
        if self.node_id is not None:
            payload["node_id"] = self.node_id
        if self.host is not None:
            payload["host"] = self.host
        if self.gpu_index is not None:
            payload["gpu_index"] = self.gpu_index
        if self.gpu_indices:
            payload["gpu_indices"] = list(self.gpu_indices)
        if self.gpu_uuid is not None:
            payload["gpu_uuid"] = self.gpu_uuid
        if self.gpu_uuids:
            payload["gpu_uuids"] = list(self.gpu_uuids)
        if self.source is not None:
            payload["source"] = self.source
        if self.required_vram_mib is not None:
            payload["required_vram_mib"] = self.required_vram_mib
        if self.available_vram_mib is not None:
            payload["available_vram_mib"] = self.available_vram_mib
        if self.model_cached:
            payload["model_cached"] = True
        if self.available_until is not None:
            payload["available_until"] = format_datetime(self.available_until)
        return payload


def inventories_to_json(node_inventories: Iterable[NodeInventory]) -> str:
    return json.dumps([node.to_dict() for node in node_inventories], indent=2, sort_keys=True)


def load_node_inventory_file(path: str | Path) -> list[NodeInventory]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "nodes" in payload:
        payload = payload["nodes"]
    if not isinstance(payload, list):
        payload = [payload]
    return [NodeInventory.from_dict(item) for item in payload]
