from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Iterable


def _string_tuple(values: Iterable[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(value) for value in values)


def _dict_copy(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    return {} if payload is None else dict(payload)


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    check_id: str
    category: str
    status: str
    severity: str
    detected_value: str | None = None
    required_value: str | None = None
    auto_repairable: bool = False
    repair_action_ids: tuple[str, ...] = field(default_factory=tuple)
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "check_id": self.check_id,
            "category": self.category,
            "status": self.status,
            "severity": self.severity,
            "auto_repairable": self.auto_repairable,
        }
        if self.detected_value is not None:
            payload["detected_value"] = self.detected_value
        if self.required_value is not None:
            payload["required_value"] = self.required_value
        if self.repair_action_ids:
            payload["repair_action_ids"] = list(self.repair_action_ids)
        if self.message is not None:
            payload["message"] = self.message
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PreflightCheck":
        return cls(
            check_id=str(payload["check_id"]),
            category=str(payload["category"]),
            status=str(payload["status"]),
            severity=str(payload["severity"]),
            detected_value=(
                str(payload["detected_value"])
                if payload.get("detected_value") is not None
                else None
            ),
            required_value=(
                str(payload["required_value"])
                if payload.get("required_value") is not None
                else None
            ),
            auto_repairable=bool(payload.get("auto_repairable", False)),
            repair_action_ids=_string_tuple(payload.get("repair_action_ids")),
            message=str(payload["message"]) if payload.get("message") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class RuntimeRequirement:
    key: str
    status: str
    detected_value: str | None = None
    required_value: str | None = None
    auto_repairable: bool = False
    repair_action_id: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "key": self.key,
            "status": self.status,
            "auto_repairable": self.auto_repairable,
        }
        if self.detected_value is not None:
            payload["detected_value"] = self.detected_value
        if self.required_value is not None:
            payload["required_value"] = self.required_value
        if self.repair_action_id is not None:
            payload["repair_action_id"] = self.repair_action_id
        if self.message is not None:
            payload["message"] = self.message
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeRequirement":
        return cls(
            key=str(payload["key"]),
            status=str(payload["status"]),
            detected_value=(
                str(payload["detected_value"])
                if payload.get("detected_value") is not None
                else None
            ),
            required_value=(
                str(payload["required_value"])
                if payload.get("required_value") is not None
                else None
            ),
            auto_repairable=bool(payload.get("auto_repairable", False)),
            repair_action_id=(
                str(payload["repair_action_id"])
                if payload.get("repair_action_id") is not None
                else None
            ),
            message=str(payload["message"]) if payload.get("message") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class RuntimeRequirementReport:
    runtime: str
    profile: str | None
    model_id: str | None
    topology: dict[str, int]
    status: str
    available: bool
    repairable: bool
    detected_version: str | None = None
    target_version: str | None = None
    requirements: tuple[RuntimeRequirement, ...] = field(default_factory=tuple)
    available_config: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "runtime": self.runtime,
            "status": self.status,
            "available": self.available,
            "repairable": self.repairable,
            "topology": dict(self.topology),
            "requirements": [item.to_dict() for item in self.requirements],
            "available_config": dict(self.available_config),
            "warnings": list(self.warnings),
        }
        if self.profile is not None:
            payload["profile"] = self.profile
        if self.model_id is not None:
            payload["model_id"] = self.model_id
        if self.detected_version is not None:
            payload["detected_version"] = self.detected_version
        if self.target_version is not None:
            payload["target_version"] = self.target_version
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeRequirementReport":
        return cls(
            runtime=str(payload["runtime"]),
            profile=str(payload["profile"]) if payload.get("profile") is not None else None,
            model_id=str(payload["model_id"]) if payload.get("model_id") is not None else None,
            topology={str(key): int(value) for key, value in dict(payload.get("topology") or {}).items()},
            status=str(payload["status"]),
            available=bool(payload["available"]),
            repairable=bool(payload["repairable"]),
            detected_version=(
                str(payload["detected_version"])
                if payload.get("detected_version") is not None
                else None
            ),
            target_version=(
                str(payload["target_version"])
                if payload.get("target_version") is not None
                else None
            ),
            requirements=tuple(
                RuntimeRequirement.from_dict(item)
                for item in payload.get("requirements", [])
                if isinstance(item, Mapping)
            ),
            available_config=_dict_copy(payload.get("available_config")),
            warnings=_string_tuple(payload.get("warnings")),
        )


@dataclass(frozen=True, slots=True)
class NodePreflightReport:
    report_id: str
    node_id: str
    provider: str | None
    generated_at: str
    status: str
    summary: str
    inventory_snapshot: dict[str, Any]
    checks: tuple[PreflightCheck, ...] = field(default_factory=tuple)
    runtime_reports: tuple[RuntimeRequirementReport, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "report_id": self.report_id,
            "node_id": self.node_id,
            "generated_at": self.generated_at,
            "status": self.status,
            "summary": self.summary,
            "inventory_snapshot": dict(self.inventory_snapshot),
            "checks": [item.to_dict() for item in self.checks],
            "runtime_reports": [item.to_dict() for item in self.runtime_reports],
        }
        if self.provider is not None:
            payload["provider"] = self.provider
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NodePreflightReport":
        return cls(
            report_id=str(payload["report_id"]),
            node_id=str(payload["node_id"]),
            provider=str(payload["provider"]) if payload.get("provider") is not None else None,
            generated_at=str(payload["generated_at"]),
            status=str(payload["status"]),
            summary=str(payload["summary"]),
            inventory_snapshot=_dict_copy(payload.get("inventory_snapshot")),
            checks=tuple(
                PreflightCheck.from_dict(item)
                for item in payload.get("checks", [])
                if isinstance(item, Mapping)
            ),
            runtime_reports=tuple(
                RuntimeRequirementReport.from_dict(item)
                for item in payload.get("runtime_reports", [])
                if isinstance(item, Mapping)
            ),
        )
