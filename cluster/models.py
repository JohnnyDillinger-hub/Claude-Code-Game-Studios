from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import json
from typing import Any, Iterable, Mapping


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def parse_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp must include timezone information: {value}")
    return parsed.astimezone(UTC)


def format_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _sorted_labels(labels: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    if not labels:
        return ()
    return tuple(sorted((str(key), str(val)) for key, val in labels.items()))


def _labels_dict(labels: Iterable[tuple[str, str]]) -> dict[str, str]:
    return {key: value for key, value in labels}


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
        )


@dataclass(frozen=True, slots=True)
class AgentRequest:
    agent_id: str
    required_vram_mib: int
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
class PlacementDecision:
    status: str
    reason: str
    agent_id: str
    node_id: str | None = None
    host: str | None = None
    gpu_index: int | None = None
    gpu_uuid: str | None = None
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
        if self.gpu_uuid is not None:
            payload["gpu_uuid"] = self.gpu_uuid
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
