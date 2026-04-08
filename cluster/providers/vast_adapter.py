from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from urllib import request as urlrequest

from cluster.providers.base import ProviderAdapter, ProviderError
from cluster.providers.models import BootstrapBundle, ProviderBlueprint, ProviderOffer, ProvisionRequest, ProvisionedResource


VAST_API_BASE = "https://console.vast.ai/api/v0"


class VastAdapter(ProviderAdapter):
    name = "vast"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("VAST_API_KEY")

    def list_offers(self) -> list[ProviderOffer]:
        if not self.api_key:
            return self._demo_offers()
        payload = {
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "limit": 10,
        }
        raw = self._post_json(f"{VAST_API_BASE}/bundles/", payload)
        offers = raw.get("offers", [])
        return [self._normalize_offer(item) for item in offers]

    def validate_request(
        self,
        request: ProvisionRequest,
        blueprint: ProviderBlueprint | None = None,
    ) -> None:
        if request.provider != self.name:
            raise ProviderError(f"Provider request expected {self.name!r}, got {request.provider!r}")
        if request.offer_id is None and blueprint is None:
            raise ProviderError("Vast provisioning requires either --offer-id or a provider blueprint")

    def create_resource(
        self,
        request: ProvisionRequest,
        bundle: BootstrapBundle,
        blueprint: ProviderBlueprint | None = None,
        selected_offer: ProviderOffer | None = None,
    ) -> ProvisionedResource:
        display_name = bundle.node_id
        if request.dry_run or not self.api_key:
            return ProvisionedResource(
                provider=self.name,
                resource_id=f"dry-run:{request.offer_id or bundle.node_id}",
                resource_kind="instance",
                display_name=display_name,
                region=request.region or (selected_offer.region if selected_offer is not None else None),
                status="dry-run",
                offer_id=request.offer_id,
                raw_provider_payload={
                    "mode": "dry-run",
                    "blueprint_id": request.blueprint_id,
                    "provider_config_template": (
                        blueprint.provider_config_template if blueprint is not None else {}
                    ),
                },
            )
        raise ProviderError("Real Vast instance creation is not enabled in this phase; use --dry-run")

    def _post_json(self, url: str, payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        http_request = urlrequest.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlrequest.urlopen(http_request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def _normalize_offer(self, payload: dict[str, object]) -> ProviderOffer:
        gpu_count = payload.get("num_gpus")
        vram_gb = payload.get("gpu_ram")
        discovered_at = datetime.now(tz=timezone.utc)
        return ProviderOffer(
            provider=self.name,
            offer_id=str(payload["id"]),
            resource_kind="instance",
            region=str(payload.get("geolocation")) if payload.get("geolocation") else None,
            datacenter=str(payload.get("datacenter")) if payload.get("datacenter") else None,
            gpu_name=str(payload.get("gpu_name")) if payload.get("gpu_name") else None,
            gpu_count=int(gpu_count) if gpu_count is not None else None,
            vram_gb=float(vram_gb) if vram_gb is not None else None,
            cpu_count=int(payload["cpu_cores"]) if payload.get("cpu_cores") is not None else None,
            ram_gb=float(payload["cpu_ram"]) if payload.get("cpu_ram") is not None else None,
            price_hourly=(
                float(payload["dph_total"]) if payload.get("dph_total") is not None else None
            ),
            currency="USD",
            availability_mode=str(payload.get("type")) if payload.get("type") else "ondemand",
            preemptible=str(payload.get("type")) == "interruptible",
            supports_template=True,
            supports_cloud_init=False,
            supports_public_ip=True,
            supports_volume=True,
            raw_provider_payload=dict(payload),
            discovered_at=discovered_at,
        )

    def _demo_offers(self) -> list[ProviderOffer]:
        return [
            ProviderOffer(
                provider=self.name,
                offer_id="vast-demo-rtx5090x2",
                resource_kind="instance",
                region="EU",
                datacenter="vast-demo-eu-1",
                gpu_name="NVIDIA GeForce RTX 5090",
                gpu_count=2,
                vram_gb=32.0,
                cpu_count=32,
                ram_gb=128.0,
                price_hourly=1.89,
                currency="USD",
                availability_mode="ondemand",
                preemptible=False,
                supports_template=True,
                supports_public_ip=True,
                supports_volume=True,
                raw_provider_payload={"demo": True},
            ),
            ProviderOffer(
                provider=self.name,
                offer_id="vast-demo-rtx5090x4",
                resource_kind="instance",
                region="US",
                datacenter="vast-demo-us-2",
                gpu_name="NVIDIA GeForce RTX 5090",
                gpu_count=4,
                vram_gb=32.0,
                cpu_count=64,
                ram_gb=256.0,
                price_hourly=3.95,
                currency="USD",
                availability_mode="interruptible",
                preemptible=True,
                supports_template=True,
                supports_public_ip=True,
                supports_volume=True,
                raw_provider_payload={"demo": True},
            ),
        ]
