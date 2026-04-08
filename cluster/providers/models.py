from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

from cluster.models import format_datetime, parse_datetime, utc_now


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


def _dict_copy(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    return {} if payload is None else dict(payload)


@dataclass(frozen=True, slots=True)
class ProviderOffer:
    provider: str
    offer_id: str
    resource_kind: str
    region: str | None = None
    datacenter: str | None = None
    gpu_name: str | None = None
    gpu_count: int | None = None
    vram_gb: float | None = None
    cpu_count: int | None = None
    ram_gb: float | None = None
    price_hourly: float | None = None
    currency: str | None = None
    availability_mode: str | None = None
    preemptible: bool = False
    supports_template: bool = False
    supports_cloud_init: bool = False
    supports_public_ip: bool = False
    supports_volume: bool = False
    raw_provider_payload: dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": self.provider,
            "offer_id": self.offer_id,
            "resource_kind": self.resource_kind,
            "preemptible": self.preemptible,
            "supports_template": self.supports_template,
            "supports_cloud_init": self.supports_cloud_init,
            "supports_public_ip": self.supports_public_ip,
            "supports_volume": self.supports_volume,
            "discovered_at": format_datetime(self.discovered_at),
            "raw_provider_payload": self.raw_provider_payload,
        }
        if self.region is not None:
            payload["region"] = self.region
        if self.datacenter is not None:
            payload["datacenter"] = self.datacenter
        if self.gpu_name is not None:
            payload["gpu_name"] = self.gpu_name
        if self.gpu_count is not None:
            payload["gpu_count"] = self.gpu_count
        if self.vram_gb is not None:
            payload["vram_gb"] = self.vram_gb
        if self.cpu_count is not None:
            payload["cpu_count"] = self.cpu_count
        if self.ram_gb is not None:
            payload["ram_gb"] = self.ram_gb
        if self.price_hourly is not None:
            payload["price_hourly"] = self.price_hourly
        if self.currency is not None:
            payload["currency"] = self.currency
        if self.availability_mode is not None:
            payload["availability_mode"] = self.availability_mode
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderOffer":
        return cls(
            provider=str(payload["provider"]),
            offer_id=str(payload["offer_id"]),
            resource_kind=str(payload["resource_kind"]),
            region=str(payload["region"]) if payload.get("region") else None,
            datacenter=str(payload["datacenter"]) if payload.get("datacenter") else None,
            gpu_name=str(payload["gpu_name"]) if payload.get("gpu_name") else None,
            gpu_count=int(payload["gpu_count"]) if payload.get("gpu_count") is not None else None,
            vram_gb=float(payload["vram_gb"]) if payload.get("vram_gb") is not None else None,
            cpu_count=int(payload["cpu_count"]) if payload.get("cpu_count") is not None else None,
            ram_gb=float(payload["ram_gb"]) if payload.get("ram_gb") is not None else None,
            price_hourly=(
                float(payload["price_hourly"]) if payload.get("price_hourly") is not None else None
            ),
            currency=str(payload["currency"]) if payload.get("currency") else None,
            availability_mode=(
                str(payload["availability_mode"]) if payload.get("availability_mode") else None
            ),
            preemptible=bool(payload.get("preemptible", False)),
            supports_template=bool(payload.get("supports_template", False)),
            supports_cloud_init=bool(payload.get("supports_cloud_init", False)),
            supports_public_ip=bool(payload.get("supports_public_ip", False)),
            supports_volume=bool(payload.get("supports_volume", False)),
            raw_provider_payload=_dict_copy(payload.get("raw_provider_payload")),
            discovered_at=(
                parse_datetime(str(payload["discovered_at"]))
                if payload.get("discovered_at")
                else utc_now()
            ),
        )


@dataclass(frozen=True, slots=True)
class ProviderBlueprint:
    blueprint_id: str
    name: str
    description: str
    provider: str
    runtime_family: str
    model_id: str | None = None
    cached_models: tuple[str, ...] = field(default_factory=tuple)
    labels: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    trust_tier: str | None = None
    network_tier: str | None = None
    default_gpu_count: int | None = None
    default_region: str | None = None
    provider_config_template: dict[str, Any] = field(default_factory=dict)

    @property
    def labels_map(self) -> dict[str, str]:
        return _labels_dict(self.labels)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "blueprint_id": self.blueprint_id,
            "name": self.name,
            "description": self.description,
            "provider": self.provider,
            "runtime_family": self.runtime_family,
            "cached_models": list(self.cached_models),
            "provider_config_template": self.provider_config_template,
        }
        if self.labels:
            payload["labels"] = self.labels_map
        if self.model_id is not None:
            payload["model_id"] = self.model_id
        if self.trust_tier is not None:
            payload["trust_tier"] = self.trust_tier
        if self.network_tier is not None:
            payload["network_tier"] = self.network_tier
        if self.default_gpu_count is not None:
            payload["default_gpu_count"] = self.default_gpu_count
        if self.default_region is not None:
            payload["default_region"] = self.default_region
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderBlueprint":
        return cls(
            blueprint_id=str(payload["blueprint_id"]),
            name=str(payload["name"]),
            description=str(payload["description"]),
            provider=str(payload["provider"]),
            runtime_family=str(payload["runtime_family"]),
            model_id=str(payload["model_id"]) if payload.get("model_id") else None,
            cached_models=_string_tuple(payload.get("cached_models")),
            labels=_sorted_labels(payload.get("labels")),
            trust_tier=str(payload["trust_tier"]) if payload.get("trust_tier") else None,
            network_tier=str(payload["network_tier"]) if payload.get("network_tier") else None,
            default_gpu_count=(
                int(payload["default_gpu_count"])
                if payload.get("default_gpu_count") is not None
                else None
            ),
            default_region=str(payload["default_region"]) if payload.get("default_region") else None,
            provider_config_template=_dict_copy(payload.get("provider_config_template")),
        )


@dataclass(frozen=True, slots=True)
class ProvisionRequest:
    provider: str
    blueprint_id: str | None = None
    offer_id: str | None = None
    region: str | None = None
    gpu_count: int | None = None
    public_ip: bool | None = None
    volume_gb: int | None = None
    preemptible_ok: bool = False
    cached_models: tuple[str, ...] = field(default_factory=tuple)
    labels: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    trust_tier: str | None = None
    network_tier: str | None = None
    dry_run: bool = False
    provider_options: dict[str, Any] = field(default_factory=dict)

    @property
    def labels_map(self) -> dict[str, str]:
        return _labels_dict(self.labels)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": self.provider,
            "preemptible_ok": self.preemptible_ok,
            "cached_models": list(self.cached_models),
            "dry_run": self.dry_run,
            "provider_options": self.provider_options,
        }
        if self.blueprint_id is not None:
            payload["blueprint_id"] = self.blueprint_id
        if self.offer_id is not None:
            payload["offer_id"] = self.offer_id
        if self.region is not None:
            payload["region"] = self.region
        if self.gpu_count is not None:
            payload["gpu_count"] = self.gpu_count
        if self.public_ip is not None:
            payload["public_ip"] = self.public_ip
        if self.volume_gb is not None:
            payload["volume_gb"] = self.volume_gb
        if self.labels:
            payload["labels"] = self.labels_map
        if self.trust_tier is not None:
            payload["trust_tier"] = self.trust_tier
        if self.network_tier is not None:
            payload["network_tier"] = self.network_tier
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProvisionRequest":
        return cls(
            provider=str(payload["provider"]),
            blueprint_id=str(payload["blueprint_id"]) if payload.get("blueprint_id") else None,
            offer_id=str(payload["offer_id"]) if payload.get("offer_id") else None,
            region=str(payload["region"]) if payload.get("region") else None,
            gpu_count=int(payload["gpu_count"]) if payload.get("gpu_count") is not None else None,
            public_ip=bool(payload["public_ip"]) if payload.get("public_ip") is not None else None,
            volume_gb=int(payload["volume_gb"]) if payload.get("volume_gb") is not None else None,
            preemptible_ok=bool(payload.get("preemptible_ok", False)),
            cached_models=_string_tuple(payload.get("cached_models")),
            labels=_sorted_labels(payload.get("labels")),
            trust_tier=str(payload["trust_tier"]) if payload.get("trust_tier") else None,
            network_tier=str(payload["network_tier"]) if payload.get("network_tier") else None,
            dry_run=bool(payload.get("dry_run", False)),
            provider_options=_dict_copy(payload.get("provider_options")),
        )


@dataclass(frozen=True, slots=True)
class BootstrapBundle:
    node_id: str
    registry_url: str | None = None
    heartbeat_url: str | None = None
    heartbeat_state_file: str | None = None
    join_token_placeholder: str | None = None
    lease_duration_seconds: int | None = None
    cached_models: tuple[str, ...] = field(default_factory=tuple)
    labels: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    trust_tier: str | None = None
    network_tier: str | None = None
    runtime_labels: tuple[str, ...] = field(default_factory=tuple)
    node_agent_config: dict[str, Any] = field(default_factory=dict)
    cloud_init_user_data: str | None = None
    onstart_command: str | None = None

    @property
    def labels_map(self) -> dict[str, str]:
        return _labels_dict(self.labels)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "node_id": self.node_id,
            "cached_models": list(self.cached_models),
            "runtime_labels": list(self.runtime_labels),
            "node_agent_config": self.node_agent_config,
        }
        if self.labels:
            payload["labels"] = self.labels_map
        if self.registry_url is not None:
            payload["registry_url"] = self.registry_url
        if self.heartbeat_url is not None:
            payload["heartbeat_url"] = self.heartbeat_url
        if self.heartbeat_state_file is not None:
            payload["heartbeat_state_file"] = self.heartbeat_state_file
        if self.join_token_placeholder is not None:
            payload["join_token_placeholder"] = self.join_token_placeholder
        if self.lease_duration_seconds is not None:
            payload["lease_duration_seconds"] = self.lease_duration_seconds
        if self.trust_tier is not None:
            payload["trust_tier"] = self.trust_tier
        if self.network_tier is not None:
            payload["network_tier"] = self.network_tier
        if self.cloud_init_user_data is not None:
            payload["cloud_init_user_data"] = self.cloud_init_user_data
        if self.onstart_command is not None:
            payload["onstart_command"] = self.onstart_command
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BootstrapBundle":
        return cls(
            node_id=str(payload["node_id"]),
            registry_url=str(payload["registry_url"]) if payload.get("registry_url") else None,
            heartbeat_url=str(payload["heartbeat_url"]) if payload.get("heartbeat_url") else None,
            heartbeat_state_file=(
                str(payload["heartbeat_state_file"]) if payload.get("heartbeat_state_file") else None
            ),
            join_token_placeholder=(
                str(payload["join_token_placeholder"])
                if payload.get("join_token_placeholder")
                else None
            ),
            lease_duration_seconds=(
                int(payload["lease_duration_seconds"])
                if payload.get("lease_duration_seconds") is not None
                else None
            ),
            cached_models=_string_tuple(payload.get("cached_models")),
            labels=_sorted_labels(payload.get("labels")),
            trust_tier=str(payload["trust_tier"]) if payload.get("trust_tier") else None,
            network_tier=str(payload["network_tier"]) if payload.get("network_tier") else None,
            runtime_labels=_string_tuple(payload.get("runtime_labels")),
            node_agent_config=_dict_copy(payload.get("node_agent_config")),
            cloud_init_user_data=(
                str(payload["cloud_init_user_data"]) if payload.get("cloud_init_user_data") else None
            ),
            onstart_command=str(payload["onstart_command"]) if payload.get("onstart_command") else None,
        )


@dataclass(frozen=True, slots=True)
class ProvisionedResource:
    provider: str
    resource_id: str
    resource_kind: str
    display_name: str
    region: str | None = None
    host: str | None = None
    public_ip: str | None = None
    ssh_user: str | None = None
    ssh_port: int | None = None
    status: str = "created"
    offer_id: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    raw_provider_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": self.provider,
            "resource_id": self.resource_id,
            "resource_kind": self.resource_kind,
            "display_name": self.display_name,
            "status": self.status,
            "created_at": format_datetime(self.created_at),
            "raw_provider_payload": self.raw_provider_payload,
        }
        if self.region is not None:
            payload["region"] = self.region
        if self.host is not None:
            payload["host"] = self.host
        if self.public_ip is not None:
            payload["public_ip"] = self.public_ip
        if self.ssh_user is not None:
            payload["ssh_user"] = self.ssh_user
        if self.ssh_port is not None:
            payload["ssh_port"] = self.ssh_port
        if self.offer_id is not None:
            payload["offer_id"] = self.offer_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProvisionedResource":
        return cls(
            provider=str(payload["provider"]),
            resource_id=str(payload["resource_id"]),
            resource_kind=str(payload["resource_kind"]),
            display_name=str(payload["display_name"]),
            region=str(payload["region"]) if payload.get("region") else None,
            host=str(payload["host"]) if payload.get("host") else None,
            public_ip=str(payload["public_ip"]) if payload.get("public_ip") else None,
            ssh_user=str(payload["ssh_user"]) if payload.get("ssh_user") else None,
            ssh_port=int(payload["ssh_port"]) if payload.get("ssh_port") is not None else None,
            status=str(payload.get("status", "created")),
            offer_id=str(payload["offer_id"]) if payload.get("offer_id") else None,
            created_at=(
                parse_datetime(str(payload["created_at"]))
                if payload.get("created_at")
                else utc_now()
            ),
            raw_provider_payload=_dict_copy(payload.get("raw_provider_payload")),
        )


@dataclass(frozen=True, slots=True)
class ProvisionJob:
    job_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    request: ProvisionRequest
    selected_offer: ProviderOffer | None = None
    bootstrap_bundle: BootstrapBundle | None = None
    provisioned_resource: ProvisionedResource | None = None
    joined_node_id: str | None = None
    joined_at: datetime | None = None
    joined_node_snapshot: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": format_datetime(self.created_at),
            "updated_at": format_datetime(self.updated_at),
            "request": self.request.to_dict(),
        }
        if self.selected_offer is not None:
            payload["selected_offer"] = self.selected_offer.to_dict()
        if self.bootstrap_bundle is not None:
            payload["bootstrap_bundle"] = self.bootstrap_bundle.to_dict()
        if self.provisioned_resource is not None:
            payload["provisioned_resource"] = self.provisioned_resource.to_dict()
        if self.joined_node_id is not None:
            payload["joined_node_id"] = self.joined_node_id
        if self.joined_at is not None:
            payload["joined_at"] = format_datetime(self.joined_at)
        if self.joined_node_snapshot:
            payload["joined_node_snapshot"] = self.joined_node_snapshot
        if self.error_code is not None:
            payload["error_code"] = self.error_code
        if self.error_message is not None:
            payload["error_message"] = self.error_message
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProvisionJob":
        return cls(
            job_id=str(payload["job_id"]),
            status=str(payload["status"]),
            created_at=parse_datetime(str(payload["created_at"])),
            updated_at=parse_datetime(str(payload["updated_at"])),
            request=ProvisionRequest.from_dict(payload["request"]),
            selected_offer=(
                ProviderOffer.from_dict(payload["selected_offer"])
                if isinstance(payload.get("selected_offer"), Mapping)
                else None
            ),
            bootstrap_bundle=(
                BootstrapBundle.from_dict(payload["bootstrap_bundle"])
                if isinstance(payload.get("bootstrap_bundle"), Mapping)
                else None
            ),
            provisioned_resource=(
                ProvisionedResource.from_dict(payload["provisioned_resource"])
                if isinstance(payload.get("provisioned_resource"), Mapping)
                else None
            ),
            joined_node_id=(
                str(payload["joined_node_id"]) if payload.get("joined_node_id") is not None else None
            ),
            joined_at=(
                parse_datetime(str(payload["joined_at"]))
                if payload.get("joined_at") is not None
                else None
            ),
            joined_node_snapshot=_dict_copy(payload.get("joined_node_snapshot")),
            error_code=str(payload["error_code"]) if payload.get("error_code") else None,
            error_message=str(payload["error_message"]) if payload.get("error_message") else None,
        )
