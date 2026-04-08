from __future__ import annotations

import os

from cluster.providers.base import ProviderAdapter, ProviderError
from cluster.providers.models import BootstrapBundle, ProviderBlueprint, ProviderOffer, ProvisionRequest, ProvisionedResource


class NebiusAdapter(ProviderAdapter):
    name = "nebius"

    def __init__(self, api_key: str | None = None, project_id: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("NEBIUS_IAM_TOKEN")
        self.project_id = project_id or os.environ.get("NEBIUS_PROJECT_ID")

    def list_offers(self) -> list[ProviderOffer]:
        return self._demo_offers()

    def validate_request(
        self,
        request: ProvisionRequest,
        blueprint: ProviderBlueprint | None = None,
    ) -> None:
        if request.provider != self.name:
            raise ProviderError(f"Provider request expected {self.name!r}, got {request.provider!r}")
        if request.offer_id is None and blueprint is None:
            raise ProviderError("Nebius provisioning requires either --offer-id or a provider blueprint")

    def create_resource(
        self,
        request: ProvisionRequest,
        bundle: BootstrapBundle,
        blueprint: ProviderBlueprint | None = None,
        selected_offer: ProviderOffer | None = None,
    ) -> ProvisionedResource:
        if request.dry_run or not self.api_key or not self.project_id:
            return ProvisionedResource(
                provider=self.name,
                resource_id=f"dry-run:{request.offer_id or bundle.node_id}",
                resource_kind="vm",
                display_name=bundle.node_id,
                region=request.region or (selected_offer.region if selected_offer is not None else None),
                status="dry-run",
                offer_id=request.offer_id,
                raw_provider_payload={
                    "mode": "dry-run",
                    "project_id_present": bool(self.project_id),
                    "blueprint_id": request.blueprint_id,
                    "provider_options": request.provider_options,
                },
            )
        raise ProviderError("Real Nebius VM creation is not enabled in this phase; use --dry-run")

    def _demo_offers(self) -> list[ProviderOffer]:
        return [
            ProviderOffer(
                provider=self.name,
                offer_id="gpu-h100-sxm:gpu1",
                resource_kind="vm",
                region="eu-north1",
                datacenter="eu-north1-a",
                gpu_name="NVIDIA H100",
                gpu_count=1,
                vram_gb=80.0,
                price_hourly=None,
                currency="USD",
                availability_mode="regular",
                preemptible=False,
                supports_template=False,
                supports_cloud_init=True,
                supports_public_ip=True,
                supports_volume=True,
                raw_provider_payload={"demo": True, "platform": "gpu-h100-sxm", "preset": "gpu1"},
            ),
            ProviderOffer(
                provider=self.name,
                offer_id="gpu-h100-sxm:gpu8-preemptible",
                resource_kind="vm",
                region="eu-north1",
                datacenter="eu-north1-b",
                gpu_name="NVIDIA H100",
                gpu_count=8,
                vram_gb=80.0,
                price_hourly=None,
                currency="USD",
                availability_mode="preemptible",
                preemptible=True,
                supports_template=False,
                supports_cloud_init=True,
                supports_public_ip=True,
                supports_volume=True,
                raw_provider_payload={
                    "demo": True,
                    "platform": "gpu-h100-sxm",
                    "preset": "gpu8",
                    "preemptible": True,
                },
            ),
        ]
