from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import os
import shlex
import time
from urllib import error as urlerror
from urllib import request as urlrequest

from cluster.providers.base import ProviderAdapter, ProviderError
from cluster.providers.models import (
    BootstrapBundle,
    ProviderBlueprint,
    ProviderOffer,
    ProvisionRequest,
    ProvisionedResource,
)


VAST_API_BASE = "https://console.vast.ai/api/v0"


class VastAdapter(ProviderAdapter):
    name = "vast"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("VAST_API_KEY")

    def list_offers(self) -> list[ProviderOffer]:
        if not self.api_key:
            return self._demo_offers()
        payload = {
            "type": "ondemand",
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "limit": 100,
        }
        raw = self._post_json(f"{VAST_API_BASE}/bundles/", payload)
        offers = raw.get("offers", [])
        if not isinstance(offers, list):
            raise ProviderError(f"Unexpected Vast offer payload: {raw}")
        return [self._normalize_offer(item) for item in offers if isinstance(item, dict)]

    def validate_request(
        self,
        request: ProvisionRequest,
        blueprint: ProviderBlueprint | None = None,
    ) -> None:
        if request.provider != self.name:
            raise ProviderError(
                f"Provider request expected {self.name!r}, got {request.provider!r}"
            )
        if request.offer_id is None and blueprint is None:
            raise ProviderError(
                "Vast provisioning requires either --offer-id or a provider blueprint"
            )

    def create_resource(
        self,
        request: ProvisionRequest,
        bundle: BootstrapBundle,
        blueprint: ProviderBlueprint | None = None,
        selected_offer: ProviderOffer | None = None,
    ) -> ProvisionedResource:
        display_name = bundle.node_id
        if request.dry_run:
            return ProvisionedResource(
                provider=self.name,
                resource_id=f"dry-run:{request.offer_id or bundle.node_id}",
                resource_kind="instance",
                display_name=display_name,
                region=(
                    request.region
                    or (selected_offer.region if selected_offer is not None else None)
                ),
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
        if not self.api_key:
            raise ProviderError(
                "VAST_API_KEY is not configured; use --dry-run or set the provider key"
            )

        offer_id = request.offer_id or (
            selected_offer.offer_id if selected_offer is not None else None
        )
        if offer_id is None:
            raise ProviderError("Vast instance creation requires a concrete offer id")

        create_payload = self._build_create_payload(
            request,
            bundle,
            blueprint=blueprint,
            selected_offer=selected_offer,
        )
        create_response = self._put_json(f"{VAST_API_BASE}/asks/{offer_id}/", create_payload)
        contract_id = create_response.get("new_contract")
        if contract_id is None:
            raise ProviderError(
                f"Vast create response did not contain new_contract: {create_response}"
            )
        instance = self._wait_for_instance(int(contract_id))
        resource_status = str(
            instance.get("actual_status") or instance.get("cur_state") or "created"
        )
        return ProvisionedResource(
            provider=self.name,
            resource_id=str(contract_id),
            resource_kind="instance",
            display_name=str(instance.get("label") or display_name),
            region=request.region or (selected_offer.region if selected_offer is not None else None),
            host=(
                str(instance.get("ssh_host"))
                if instance.get("ssh_host")
                else (
                    str(instance.get("public_ipaddr"))
                    if instance.get("public_ipaddr")
                    else None
                )
            ),
            public_ip=(
                str(instance.get("public_ipaddr"))
                if instance.get("public_ipaddr")
                else (str(instance.get("public_ip")) if instance.get("public_ip") else None)
            ),
            ssh_user="root",
            ssh_port=(
                int(instance["ssh_port"])
                if instance.get("ssh_port") is not None
                else None
            ),
            status=resource_status,
            offer_id=str(offer_id),
            raw_provider_payload={
                "create_payload": create_payload,
                "create_response": create_response,
                "instance": instance,
            },
        )

    def _post_json(self, url: str, payload: dict[str, object]) -> dict[str, object]:
        return self._request_json(url, method="POST", payload=payload)

    def _put_json(self, url: str, payload: dict[str, object]) -> dict[str, object]:
        return self._request_json(url, method="PUT", payload=payload)

    def _get_json(self, url: str) -> dict[str, object]:
        return self._request_json(url, method="GET")

    def _request_json(
        self,
        url: str,
        *,
        method: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        http_request = urlrequest.Request(
            url,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlrequest.urlopen(http_request, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"Vast API {method} {url} failed: {exc.code} {detail}") from exc
        except urlerror.URLError as exc:
            raise ProviderError(f"Vast API {method} {url} failed: {exc}") from exc
        reply = json.loads(raw)
        if not isinstance(reply, dict):
            raise ProviderError(f"Expected JSON object from Vast API {method} {url}")
        return reply

    def _build_create_payload(
        self,
        request: ProvisionRequest,
        bundle: BootstrapBundle,
        *,
        blueprint: ProviderBlueprint | None,
        selected_offer: ProviderOffer | None,
    ) -> dict[str, object]:
        provider_options = dict(request.provider_options)
        payload: dict[str, object] = {}

        template_hash_id = provider_options.pop(
            "template_hash_id",
            provider_options.pop("template_hash", None),
        )
        if template_hash_id is not None:
            payload["template_hash_id"] = str(template_hash_id)

        image = provider_options.pop(
            "image",
            (blueprint.provider_config_template.get("image") if blueprint is not None else None),
        )
        if image is not None:
            payload["image"] = str(image)

        payload["label"] = str(provider_options.pop("label", bundle.node_id))

        disk = provider_options.pop("disk", request.volume_gb)
        if disk is not None:
            payload["disk"] = int(disk)

        runtype = provider_options.pop(
            "runtype",
            (
                blueprint.provider_config_template.get("runtype")
                if blueprint is not None
                else "ssh_direct"
            ),
        )
        if runtype is not None:
            payload["runtype"] = str(runtype)

        if selected_offer is not None and selected_offer.preemptible:
            bid_price = provider_options.pop("price", selected_offer.price_hourly)
            if bid_price is not None:
                payload["price"] = float(bid_price)
        elif "price" in provider_options:
            payload["price"] = float(provider_options.pop("price"))

        onstart = provider_options.pop("onstart", None) or self._build_onstart_command(bundle)
        if onstart:
            payload["onstart"] = onstart

        env = provider_options.pop("env", None)
        if isinstance(env, dict):
            payload["env"] = {str(key): str(value) for key, value in env.items()}

        image_login = provider_options.pop("image_login", None)
        if image_login is not None:
            payload["image_login"] = str(image_login)

        target_state = provider_options.pop("target_state", None)
        if target_state is not None:
            payload["target_state"] = str(target_state)

        vm = provider_options.pop("vm", None)
        if vm is not None:
            payload["vm"] = bool(vm)

        volume_info = provider_options.pop("volume_info", None)
        if isinstance(volume_info, dict):
            payload["volume_info"] = volume_info

        cancel_unavail = provider_options.pop("cancel_unavail", None)
        if cancel_unavail is not None:
            payload["cancel_unavail"] = bool(cancel_unavail)

        for key, value in provider_options.items():
            payload[str(key)] = value
        return payload

    def _build_onstart_command(self, bundle: BootstrapBundle) -> str:
        config_json = json.dumps(bundle.node_agent_config, indent=2, sort_keys=True) + "\n"
        config_b64 = base64.b64encode(config_json.encode("utf-8")).decode("ascii")
        write_config_script = (
            "import base64, pathlib; "
            "pathlib.Path('/opt/cluster-node').mkdir(parents=True, exist_ok=True); "
            f"pathlib.Path('/opt/cluster-node/config.json').write_bytes(base64.b64decode('{config_b64}'))"
        )
        prefix = (
            "mkdir -p /opt/cluster-node /var/log && "
            f"python3 -c {shlex.quote(write_config_script)}"
        )
        if bundle.onstart_command:
            return prefix + " && " + bundle.onstart_command
        return prefix

    def _wait_for_instance(
        self,
        contract_id: int,
        attempts: int = 6,
        delay_seconds: float = 2.0,
    ) -> dict[str, object]:
        last_instance: dict[str, object] | None = None
        for _ in range(attempts):
            payload = self._get_json(f"{VAST_API_BASE}/instances/{contract_id}/")
            raw_instance = payload.get("instances")
            if isinstance(raw_instance, dict):
                last_instance = raw_instance
                status = str(
                    raw_instance.get("actual_status") or raw_instance.get("cur_state") or ""
                ).lower()
                if status in {"running", "created", "loading", "pending"}:
                    return raw_instance
            time.sleep(delay_seconds)
        if last_instance is not None:
            return last_instance
        raise ProviderError(
            f"Vast instance {contract_id} was created but details were not available"
        )

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
