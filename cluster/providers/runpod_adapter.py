from __future__ import annotations

import json
import os
from urllib import request as urlrequest

from cluster.providers.base import ProviderAdapter, ProviderError
from cluster.providers.models import BootstrapBundle, ProviderBlueprint, ProviderOffer, ProvisionRequest, ProvisionedResource


RUNPOD_API_BASE = "https://rest.runpod.io/v1"


class RunpodAdapter(ProviderAdapter):
    name = "runpod"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")

    def list_offers(self) -> list[ProviderOffer]:
        if not self.api_key:
            return self._demo_offers()
        gpu_types = self._get_json(f"{RUNPOD_API_BASE}/gpu-types")
        items = gpu_types.get("gpuTypes", [])
        offers: list[ProviderOffer] = []
        for item in items:
            gpu_name = str(item.get("displayName") or item.get("id") or "")
            secure_price = item.get("securePrice")
            community_price = item.get("communityPrice")
            memory_gb = item.get("memoryInGb")
            if secure_price is not None:
                offers.append(
                    ProviderOffer(
                        provider=self.name,
                        offer_id=f"{item.get('id')}:SECURE",
                        resource_kind="pod",
                        region=None,
                        datacenter=None,
                        gpu_name=gpu_name or None,
                        gpu_count=1,
                        vram_gb=float(memory_gb) if memory_gb is not None else None,
                        price_hourly=float(secure_price),
                        currency="USD",
                        availability_mode="SECURE",
                        preemptible=False,
                        supports_template=True,
                        supports_cloud_init=False,
                        supports_public_ip=True,
                        supports_volume=True,
                        raw_provider_payload=dict(item),
                    )
                )
            if community_price is not None:
                offers.append(
                    ProviderOffer(
                        provider=self.name,
                        offer_id=f"{item.get('id')}:COMMUNITY",
                        resource_kind="pod",
                        region=None,
                        datacenter=None,
                        gpu_name=gpu_name or None,
                        gpu_count=1,
                        vram_gb=float(memory_gb) if memory_gb is not None else None,
                        price_hourly=float(community_price),
                        currency="USD",
                        availability_mode="COMMUNITY",
                        preemptible=True,
                        supports_template=True,
                        supports_cloud_init=False,
                        supports_public_ip=True,
                        supports_volume=True,
                        raw_provider_payload=dict(item),
                    )
                )
        return offers

    def validate_request(
        self,
        request: ProvisionRequest,
        blueprint: ProviderBlueprint | None = None,
    ) -> None:
        if request.provider != self.name:
            raise ProviderError(f"Provider request expected {self.name!r}, got {request.provider!r}")
        if request.offer_id is None and blueprint is None:
            raise ProviderError("Runpod provisioning requires either --offer-id or a provider blueprint")

    def create_resource(
        self,
        request: ProvisionRequest,
        bundle: BootstrapBundle,
        blueprint: ProviderBlueprint | None = None,
        selected_offer: ProviderOffer | None = None,
    ) -> ProvisionedResource:
        if request.dry_run or not self.api_key:
            return ProvisionedResource(
                provider=self.name,
                resource_id=f"dry-run:{request.offer_id or bundle.node_id}",
                resource_kind="pod",
                display_name=bundle.node_id,
                region=request.region,
                status="dry-run",
                offer_id=request.offer_id,
                raw_provider_payload={
                    "mode": "dry-run",
                    "templateId": request.provider_options.get("templateId")
                    if request.provider_options
                    else None,
                    "networkVolumeId": request.provider_options.get("networkVolumeId")
                    if request.provider_options
                    else None,
                    "blueprint_id": request.blueprint_id,
                    "selected_offer": selected_offer.to_dict() if selected_offer is not None else None,
                },
            )
        raise ProviderError("Real Runpod pod creation is not enabled in this phase; use --dry-run")

    def _get_json(self, url: str) -> dict[str, object]:
        http_request = urlrequest.Request(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        with urlrequest.urlopen(http_request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def _demo_offers(self) -> list[ProviderOffer]:
        return [
            ProviderOffer(
                provider=self.name,
                offer_id="NVIDIA GeForce RTX 5090:SECURE",
                resource_kind="pod",
                region="EU-RO-1",
                datacenter="EU-RO-1",
                gpu_name="NVIDIA GeForce RTX 5090",
                gpu_count=2,
                vram_gb=32.0,
                price_hourly=2.49,
                currency="USD",
                availability_mode="SECURE",
                preemptible=False,
                supports_template=True,
                supports_public_ip=True,
                supports_volume=True,
                raw_provider_payload={"demo": True},
            ),
            ProviderOffer(
                provider=self.name,
                offer_id="NVIDIA GeForce RTX 5090:COMMUNITY",
                resource_kind="pod",
                region="US-KS-2",
                datacenter="US-KS-2",
                gpu_name="NVIDIA GeForce RTX 5090",
                gpu_count=4,
                vram_gb=32.0,
                price_hourly=3.29,
                currency="USD",
                availability_mode="COMMUNITY",
                preemptible=True,
                supports_template=True,
                supports_public_ip=True,
                supports_volume=True,
                raw_provider_payload={"demo": True},
            ),
        ]
