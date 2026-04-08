from __future__ import annotations

from dataclasses import replace
from datetime import timezone
import json
from pathlib import Path
import secrets
import time
from typing import Iterable

from cluster.models import utc_now
from cluster.orchestrator.registry import NodeRegistry
from cluster.orchestrator.state_store import RegistryStateStore
from cluster.providers.base import ProviderAdapter, ProviderError
from cluster.providers.blueprints import load_builtin_blueprints
from cluster.providers.job_store import ProvisionJobStore
from cluster.providers.models import BootstrapBundle, ProviderBlueprint, ProviderOffer, ProvisionJob, ProvisionRequest
from cluster.providers.nebius_adapter import NebiusAdapter
from cluster.providers.runpod_adapter import RunpodAdapter
from cluster.providers.vast_adapter import VastAdapter


DEFAULT_JOBS_FILE = Path("production/session-state/provider-jobs.json")


class ProviderService:
    def __init__(
        self,
        adapters: Iterable[ProviderAdapter] | None = None,
        *,
        jobs_file: str | Path = DEFAULT_JOBS_FILE,
        registry_url: str | None = None,
        heartbeat_url: str | None = None,
        heartbeat_state_file: str | None = None,
        repo_clone_url: str = "https://github.com/JohnnyDillinger-hub/Claude-Code-Game-Studios.git",
        repo_branch: str = "codex/mesh-runtime-tp4",
    ) -> None:
        self.adapters: dict[str, ProviderAdapter] = {
            adapter.name: adapter
            for adapter in (adapters or (VastAdapter(), RunpodAdapter(), NebiusAdapter()))
        }
        self.jobs = ProvisionJobStore(jobs_file)
        self.registry_url = registry_url
        self.heartbeat_url = heartbeat_url
        self.heartbeat_state_file = heartbeat_state_file
        self.repo_clone_url = repo_clone_url
        self.repo_branch = repo_branch

    def list_offers(
        self,
        *,
        provider: str | None = None,
        gpu_name: str | None = None,
        min_gpu_count: int | None = None,
        min_vram_gb: float | None = None,
        max_price_hourly: float | None = None,
        region: str | None = None,
        preemptible_ok: bool | None = None,
    ) -> list[ProviderOffer]:
        offers: list[ProviderOffer] = []
        adapters = (
            [self._require_adapter(provider)]
            if provider
            else [self.adapters[name] for name in sorted(self.adapters)]
        )
        for adapter in adapters:
            offers.extend(adapter.list_offers())
        filtered: list[ProviderOffer] = []
        for offer in offers:
            if gpu_name and (offer.gpu_name or "").lower() != gpu_name.lower():
                continue
            if min_gpu_count is not None and (offer.gpu_count or 0) < min_gpu_count:
                continue
            if min_vram_gb is not None and (offer.vram_gb or 0.0) < min_vram_gb:
                continue
            if max_price_hourly is not None and offer.price_hourly is not None and offer.price_hourly > max_price_hourly:
                continue
            if region and offer.region is not None and offer.region != region and offer.datacenter != region:
                continue
            if preemptible_ok is False and offer.preemptible:
                continue
            filtered.append(offer)
        return sorted(
            filtered,
            key=lambda item: (
                item.provider,
                item.price_hourly if item.price_hourly is not None else float("inf"),
                -(item.gpu_count or 0),
                item.offer_id,
            ),
        )

    def list_blueprints(
        self,
        *,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> list[ProviderBlueprint]:
        blueprints = list(load_builtin_blueprints())
        filtered: list[ProviderBlueprint] = []
        for blueprint in blueprints:
            if provider and blueprint.provider not in {provider, "any"}:
                continue
            if model_id and blueprint.model_id not in {None, model_id}:
                continue
            filtered.append(blueprint)
        return filtered

    def build_bootstrap_bundle(
        self,
        request: ProvisionRequest,
        *,
        blueprint: ProviderBlueprint | None = None,
    ) -> BootstrapBundle:
        node_id = self._generate_node_id(request.provider)
        cached_models = request.cached_models or (blueprint.cached_models if blueprint is not None else ())
        labels = request.labels
        if blueprint is not None:
            merged_labels = dict(blueprint.labels_map)
            merged_labels.update(request.labels_map)
            labels = tuple(sorted(merged_labels.items()))
        trust_tier = request.trust_tier or (blueprint.trust_tier if blueprint is not None else None)
        network_tier = request.network_tier or (
            blueprint.network_tier if blueprint is not None else None
        )
        node_agent_config = {
            "node_id": node_id,
            "host": node_id,
            "lease_duration_seconds": 3600,
            "cached_models": list(cached_models),
            "labels": dict(labels),
            "trust_tier": trust_tier,
            "network_tier": network_tier,
            "heartbeat_url": self.heartbeat_url,
            "heartbeat_state_file": self.heartbeat_state_file,
        }
        config_json = json.dumps(node_agent_config, indent=2, sort_keys=True)
        cloud_init_user_data = "\n".join(
            [
                "#cloud-config",
                "package_update: true",
                "packages:",
                "  - git",
                "  - python3-venv",
                "write_files:",
                "  - path: /opt/cluster-node/config.json",
                "    permissions: '0644'",
                "    content: |",
                *[f"      {line}" for line in config_json.splitlines()],
                "runcmd:",
                f"  - git clone {self.repo_clone_url} /opt/Claude-Code-Game-Studios || true",
                "  - cd /opt/Claude-Code-Game-Studios && git fetch origin && git checkout "
                f"{self.repo_branch} && git pull --ff-only origin {self.repo_branch}",
                "  - nohup python3 -m cluster.node_agent.daemon --config /opt/cluster-node/config.json "
                "> /var/log/cluster-node-agent.log 2>&1 &",
            ]
        )
        onstart_command = (
            "cd /opt/Claude-Code-Game-Studios && "
            f"git fetch origin && git checkout {self.repo_branch} && "
            f"git pull --ff-only origin {self.repo_branch} && "
            "nohup python3 -m cluster.node_agent.daemon --config /opt/cluster-node/config.json "
            "> /var/log/cluster-node-agent.log 2>&1 &"
        )
        return BootstrapBundle(
            node_id=node_id,
            registry_url=self.registry_url,
            heartbeat_url=self.heartbeat_url,
            heartbeat_state_file=self.heartbeat_state_file,
            join_token_placeholder="mesh-join-token-placeholder",
            lease_duration_seconds=3600,
            cached_models=cached_models,
            labels=labels,
            trust_tier=trust_tier,
            network_tier=network_tier,
            runtime_labels=(blueprint.runtime_family,) if blueprint is not None else (),
            node_agent_config=node_agent_config,
            cloud_init_user_data=cloud_init_user_data,
            onstart_command=onstart_command,
        )

    def provision(self, request: ProvisionRequest) -> ProvisionJob:
        return self.provision_with_join_wait(request)

    def provision_with_join_wait(
        self,
        request: ProvisionRequest,
        *,
        wait_for_join: bool = False,
        join_state_file: str | Path | None = None,
        join_timeout_seconds: float = 120.0,
        join_poll_interval_seconds: float = 5.0,
    ) -> ProvisionJob:
        blueprint = self._resolve_blueprint(request)
        effective_request = self._apply_blueprint_defaults(request, blueprint)
        adapter = self._require_adapter(effective_request.provider)
        adapter.validate_request(effective_request, blueprint)
        selected_offer = self._select_offer(effective_request, blueprint)
        bundle = self.build_bootstrap_bundle(effective_request, blueprint=blueprint)
        now = utc_now()
        job_id = self._generate_job_id(effective_request.provider)
        try:
            resource = adapter.create_resource(
                effective_request,
                bundle,
                blueprint=blueprint,
                selected_offer=selected_offer,
            )
            status = self._initial_job_status(resource)
            job = ProvisionJob(
                job_id=job_id,
                status=status,
                created_at=now,
                updated_at=now,
                request=effective_request,
                selected_offer=selected_offer,
                bootstrap_bundle=bundle,
                provisioned_resource=resource,
            )
        except ProviderError as exc:
            job = ProvisionJob(
                job_id=job_id,
                status="failed",
                created_at=now,
                updated_at=now,
                request=effective_request,
                selected_offer=selected_offer,
                bootstrap_bundle=bundle,
                error_code="provider_error",
                error_message=str(exc),
            )
        self.jobs.upsert(job)
        if wait_for_join and job.status in {"bootstrapping", "provisioning"}:
            if join_state_file is None:
                raise ValueError("join_state_file is required when wait_for_join is enabled")
            return self.wait_for_job_join(
                job.job_id,
                state_file=join_state_file,
                timeout_seconds=join_timeout_seconds,
                poll_interval_seconds=join_poll_interval_seconds,
            )
        return job

    def list_jobs(self, *, status: str | None = None) -> list[ProvisionJob]:
        jobs = self.jobs.load()
        if status is None:
            return jobs
        return [job for job in jobs if job.status == status]

    def get_job(self, job_id: str) -> ProvisionJob | None:
        for job in self.jobs.load():
            if job.job_id == job_id:
                return job
        return None

    def reconcile_jobs(self, registry: NodeRegistry) -> list[ProvisionJob]:
        reconciled = [self._reconcile_job(job, registry) for job in self.jobs.load()]
        self.jobs.save(reconciled)
        return reconciled

    def reconcile_jobs_from_state_file(self, state_file: str | Path) -> list[ProvisionJob]:
        registry = RegistryStateStore(state_file).load()
        return self.reconcile_jobs(registry)

    def wait_for_job_join(
        self,
        job_id: str,
        *,
        state_file: str | Path,
        timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 5.0,
    ) -> ProvisionJob:
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        deadline = time.monotonic() + timeout_seconds
        last_job = self.get_job(job_id)
        if last_job is None:
            raise ProviderError(f"Unknown provisioning job id: {job_id}")
        while True:
            jobs = self.reconcile_jobs_from_state_file(state_file)
            for job in jobs:
                if job.job_id == job_id:
                    last_job = job
                    break
            if last_job.status in {"joined", "failed"}:
                return last_job
            if time.monotonic() >= deadline:
                return last_job
            time.sleep(poll_interval_seconds)

    def _apply_blueprint_defaults(
        self,
        request: ProvisionRequest,
        blueprint: ProviderBlueprint | None,
    ) -> ProvisionRequest:
        if blueprint is None:
            return request
        provider = request.provider
        if provider == "any":
            provider = blueprint.provider
        gpu_count = request.gpu_count if request.gpu_count is not None else blueprint.default_gpu_count
        region = request.region or blueprint.default_region
        cached_models = request.cached_models or blueprint.cached_models
        if request.labels:
            merged_labels = dict(blueprint.labels_map)
            merged_labels.update(request.labels_map)
            labels = tuple(sorted(merged_labels.items()))
        else:
            labels = blueprint.labels
        provider_options = dict(blueprint.provider_config_template)
        provider_options.update(request.provider_options)
        return replace(
            request,
            provider=provider,
            gpu_count=gpu_count,
            region=region,
            cached_models=cached_models,
            labels=labels,
            trust_tier=request.trust_tier or blueprint.trust_tier,
            network_tier=request.network_tier or blueprint.network_tier,
            provider_options=provider_options,
        )

    def _reconcile_job(self, job: ProvisionJob, registry: NodeRegistry) -> ProvisionJob:
        if job.status == "failed" or job.bootstrap_bundle is None:
            return job
        node_id = job.bootstrap_bundle.node_id
        record = registry.get_record(node_id)
        if record is None:
            return job
        now = utc_now()
        if record.node.is_expired(now=now):
            return job

        resource = job.provisioned_resource
        if resource is not None and resource.status != "joined":
            resource = replace(resource, status="joined")
        joined_at = job.joined_at or record.last_heartbeat_at or now
        return replace(
            job,
            status="joined",
            updated_at=now,
            provisioned_resource=resource,
            joined_node_id=node_id,
            joined_at=joined_at,
            joined_node_snapshot=record.node.to_dict(),
        )

    def _initial_job_status(self, resource) -> str:
        normalized = str(resource.status).lower()
        if normalized in {"failed", "error"}:
            return "failed"
        if normalized in {
            "dry-run",
            "created",
            "running",
            "loading",
            "pending",
            "starting",
            "booting",
            "bootstrapping",
        }:
            return "bootstrapping"
        return "provisioning"

    def _resolve_blueprint(self, request: ProvisionRequest) -> ProviderBlueprint | None:
        if request.blueprint_id is None:
            return None
        for blueprint in load_builtin_blueprints():
            if blueprint.blueprint_id == request.blueprint_id:
                if blueprint.provider not in {request.provider, "any"} and request.provider != "any":
                    raise ProviderError(
                        f"Blueprint {request.blueprint_id!r} is for provider {blueprint.provider!r}"
                    )
                return blueprint
        raise ProviderError(f"Unknown blueprint id: {request.blueprint_id}")

    def _select_offer(
        self,
        request: ProvisionRequest,
        blueprint: ProviderBlueprint | None,
    ) -> ProviderOffer | None:
        offers = self.list_offers(
            provider=request.provider,
            min_gpu_count=request.gpu_count,
            region=request.region,
            preemptible_ok=request.preemptible_ok,
        )
        if request.offer_id is not None:
            for offer in offers:
                if offer.offer_id == request.offer_id:
                    return offer
            return ProviderOffer(
                provider=request.provider,
                offer_id=request.offer_id,
                resource_kind="instance",
                region=request.region,
            )
        if blueprint is None:
            return None
        if not offers:
            return None
        return offers[0]

    def _require_adapter(self, provider: str) -> ProviderAdapter:
        if provider == "any":
            raise ProviderError("A concrete provider is required for this operation")
        adapter = self.adapters.get(provider)
        if adapter is None:
            raise ProviderError(f"Unknown provider: {provider}")
        return adapter

    def _generate_job_id(self, provider: str) -> str:
        return f"{provider}-{secrets.token_hex(6)}"

    def _generate_node_id(self, provider: str) -> str:
        timestamp = utc_now().astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{provider}-node-{timestamp}-{secrets.token_hex(3)}"
