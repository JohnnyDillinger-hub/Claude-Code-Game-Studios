from __future__ import annotations

from dataclasses import replace
from datetime import timezone
import json
from pathlib import Path
import secrets
import shlex
import subprocess
import time
from typing import Iterable

from cluster.models import NodeInventory, utc_now
from cluster.node_agent.preflight import build_node_preflight_report
from cluster.node_agent.preflight_models import RuntimeRequirementReport
from cluster.orchestrator.launcher import build_remote_ssh_argv
from cluster.orchestrator.registry import NodeRegistry
from cluster.orchestrator.state_store import RegistryStateStore
from cluster.providers.base import ProviderAdapter, ProviderError
from cluster.providers.blueprints import load_builtin_blueprints
from cluster.providers.job_store import ProvisionJobStore
from cluster.providers.models import (
    BootstrapBundle,
    ProviderBlueprint,
    ProviderOffer,
    ProvisionJob,
    ProvisionRequest,
    RepairAction,
    RuntimeInstallStatus,
)
from cluster.providers.nebius_adapter import NebiusAdapter
from cluster.providers.runpod_adapter import RunpodAdapter
from cluster.providers.vast_adapter import VastAdapter


DEFAULT_JOBS_FILE = Path("production/session-state/provider-jobs.json")
PROVIDER_REPO_ROOT = "/opt/Claude-Code-Game-Studios"
PROVIDER_NODE_CONFIG_PATH = "/opt/cluster-node/config.json"
PROVIDER_RUNTIME_BOOTSTRAP_LOG = "/var/log/cluster-runtime-bootstrap.log"
PROVIDER_RUNTIME_REPAIR_LOG = "/var/log/cluster-runtime-repair.log"
PROVIDER_RUNTIME_BOOTSTRAP_STABILIZATION_SECONDS = 60
PROVIDER_RUNTIME_BOOTSTRAP_MIN_FREE_DISK_GIB = 10
PROVIDER_RUNTIME_BOOTSTRAP_RETRIES = 2
PROVIDER_RUNTIME_BOOTSTRAP_TIMEOUT_SECONDS = 3600
PROVIDER_RUNTIME_BOOTSTRAP_PAUSE_SECONDS = 5
SUPPORTED_RUNTIME_STACKS = {"vllm", "sglang", "deepspeed", "ollama"}


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
        runtime_stack = self._resolve_runtime_stack(request, blueprint)
        preferred_launch_profile = self._resolve_preferred_launch_profile(request, blueprint)
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
            "bootstrap_mode": "staged",
            "runtime_stack": list(runtime_stack),
            "preferred_launch_profile": preferred_launch_profile,
            "heartbeat_url": self.heartbeat_url,
            "heartbeat_state_file": self.heartbeat_state_file,
            "runtime_bootstrap_min_free_disk_gib": PROVIDER_RUNTIME_BOOTSTRAP_MIN_FREE_DISK_GIB,
            "runtime_bootstrap_retry_count": PROVIDER_RUNTIME_BOOTSTRAP_RETRIES,
            "runtime_bootstrap_timeout_seconds": PROVIDER_RUNTIME_BOOTSTRAP_TIMEOUT_SECONDS,
            "runtime_bootstrap_pause_seconds": PROVIDER_RUNTIME_BOOTSTRAP_PAUSE_SECONDS,
            "runtime_bootstrap_stabilization_seconds": PROVIDER_RUNTIME_BOOTSTRAP_STABILIZATION_SECONDS,
        }
        config_json = json.dumps(node_agent_config, indent=2, sort_keys=True)
        runtime_bootstrap_command = self._build_runtime_bootstrap_command(
            runtime_stack,
            cached_models=cached_models,
            preferred_launch_profile=preferred_launch_profile,
        )
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
                "  - mkdir -p /opt",
                "  - if [ ! -d /opt/Claude-Code-Game-Studios/.git ]; then "
                f"git clone {self.repo_clone_url} /opt/Claude-Code-Game-Studios; fi",
                "  - cd /opt/Claude-Code-Game-Studios && git fetch origin && git checkout "
                f"{self.repo_branch} && git pull --ff-only origin {self.repo_branch}",
                "  - nohup python3 -m cluster.node_agent.daemon --config /opt/cluster-node/config.json "
                "> /var/log/cluster-node-agent.log 2>&1 &",
            ]
        )
        onstart_command = (
            "mkdir -p /opt && "
            f"if [ ! -d {PROVIDER_REPO_ROOT}/.git ]; then git clone {self.repo_clone_url} {PROVIDER_REPO_ROOT}; fi && "
            f"cd {PROVIDER_REPO_ROOT} && "
            f"git fetch origin && git checkout {self.repo_branch} && "
            f"git pull --ff-only origin {self.repo_branch} && "
            f"nohup python3 -m cluster.node_agent.daemon --config {PROVIDER_NODE_CONFIG_PATH} "
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
            runtime_labels=runtime_stack or ((blueprint.runtime_family,) if blueprint is not None else ()),
            runtime_stack=runtime_stack,
            preferred_launch_profile=preferred_launch_profile,
            node_agent_config=node_agent_config,
            cloud_init_user_data=cloud_init_user_data,
            onstart_command=onstart_command,
            runtime_bootstrap_command=runtime_bootstrap_command,
            runtime_bootstrap_log_path=PROVIDER_RUNTIME_BOOTSTRAP_LOG,
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
            runtime_bootstrap_status = "pending" if bundle.runtime_stack else "skipped"
            runtime_bootstrap_note = (
                None if bundle.runtime_stack else "No runtime bootstrap was requested for this blueprint."
            )
            job = ProvisionJob(
                job_id=job_id,
                status=status,
                created_at=now,
                updated_at=now,
                request=effective_request,
                selected_offer=selected_offer,
                bootstrap_bundle=bundle,
                provisioned_resource=resource,
                runtime_bootstrap_status=runtime_bootstrap_status,
                runtime_bootstrap_command=bundle.runtime_bootstrap_command,
                runtime_bootstrap_note=runtime_bootstrap_note,
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

    def list_jobs(
        self,
        *,
        status: str | None = None,
        provider: str | None = None,
        resource_id: str | None = None,
        resource_status: str | None = None,
    ) -> list[ProvisionJob]:
        jobs = self.jobs.load()
        return [
            job
            for job in jobs
            if self._job_matches_filters(
                job,
                status=status,
                provider=provider,
                resource_id=resource_id,
                resource_status=resource_status,
            )
        ]

    def get_job(self, job_id: str) -> ProvisionJob | None:
        for job in self.jobs.load():
            if job.job_id == job_id:
                return job
        return None

    def reconcile_jobs(
        self,
        registry: NodeRegistry,
        *,
        job_id: str | None = None,
        provider: str | None = None,
        resource_id: str | None = None,
    ) -> list[ProvisionJob]:
        reconciled: list[ProvisionJob] = []
        for job in self.jobs.load():
            if not self._job_matches_filters(
                job,
                job_id=job_id,
                provider=provider,
                resource_id=resource_id,
            ):
                reconciled.append(job)
                continue
            reconciled.append(self._reconcile_job(job, registry))
        self.jobs.save(reconciled)
        return [
            job
            for job in reconciled
            if self._job_matches_filters(
                job,
                job_id=job_id,
                provider=provider,
                resource_id=resource_id,
            )
        ]

    def reconcile_jobs_from_state_file(
        self,
        state_file: str | Path,
        *,
        job_id: str | None = None,
        provider: str | None = None,
        resource_id: str | None = None,
    ) -> list[ProvisionJob]:
        store = RegistryStateStore(state_file)
        registry = store.load()
        jobs = self.reconcile_jobs(
            registry,
            job_id=job_id,
            provider=provider,
            resource_id=resource_id,
        )
        store.save(registry)
        return jobs

    def preview_destroy_job(
        self,
        *,
        job_id: str | None = None,
        resource_id: str | None = None,
        provider: str | None = None,
    ) -> dict[str, object]:
        job = self._resolve_destroy_job(
            job_id=job_id,
            resource_id=resource_id,
            provider=provider,
        )
        resource = job.provisioned_resource
        if resource is None:
            raise ProviderError(f"Job {job.job_id!r} does not have a provisioned resource")
        return {
            "status": "dry-run",
            "job_id": job.job_id,
            "provider": job.request.provider,
            "job_status": job.status,
            "resource": resource.to_dict(),
            "message": "Re-run with --confirm to destroy the provider resource.",
        }

    def destroy_job(
        self,
        *,
        job_id: str | None = None,
        resource_id: str | None = None,
        provider: str | None = None,
        force: bool = False,
    ) -> ProvisionJob:
        job = self._resolve_destroy_job(
            job_id=job_id,
            resource_id=resource_id,
            provider=provider,
        )
        resource = job.provisioned_resource
        if resource is None:
            raise ProviderError(f"Job {job.job_id!r} does not have a provisioned resource")
        if job.status == "destroyed" or resource.status == "destroyed":
            return job

        now = utc_now()
        if resource.status == "dry-run":
            destroyed_resource = replace(
                resource,
                status="destroyed",
                raw_provider_payload={
                    **resource.raw_provider_payload,
                    "destroy_mode": "dry-run-local",
                },
            )
        else:
            adapter = self._require_adapter(job.request.provider)
            try:
                destroyed_resource = adapter.destroy_resource(resource, dry_run=False)
            except ProviderError:
                if not force:
                    raise
                destroyed_resource = replace(
                    resource,
                    status="destroyed",
                    raw_provider_payload={
                        **resource.raw_provider_payload,
                        "destroy_mode": "forced-local",
                    },
                )

        destroyed_job = replace(
            job,
            status="destroyed",
            updated_at=now,
            provisioned_resource=destroyed_resource,
        )
        self.jobs.upsert(destroyed_job)
        return destroyed_job

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

    def _resolve_runtime_stack(
        self,
        request: ProvisionRequest,
        blueprint: ProviderBlueprint | None,
    ) -> tuple[str, ...]:
        candidate = request.provider_options.get("runtime_stack")
        if candidate is None and blueprint is not None and blueprint.runtime_stack:
            return blueprint.runtime_stack
        if candidate is None and blueprint is not None and blueprint.runtime_family in SUPPORTED_RUNTIME_STACKS:
            return (blueprint.runtime_family,)
        if candidate is None:
            return ()
        if isinstance(candidate, str):
            values = [item.strip() for item in candidate.split(",") if item.strip()]
        elif isinstance(candidate, Iterable):
            values = [str(item).strip() for item in candidate if str(item).strip()]
        else:
            raise ProviderError("runtime_stack provider option must be a string or list")
        unsupported = [item for item in values if item not in SUPPORTED_RUNTIME_STACKS]
        if unsupported:
            raise ProviderError(f"Unsupported runtime(s) for provider bootstrap: {', '.join(sorted(unsupported))}")
        return tuple(values)

    def _resolve_preferred_launch_profile(
        self,
        request: ProvisionRequest,
        blueprint: ProviderBlueprint | None,
    ) -> str | None:
        profile = request.provider_options.get("launch_profile")
        if profile:
            return str(profile)
        if blueprint is not None:
            return blueprint.preferred_launch_profile
        return None

    def _build_runtime_bootstrap_command(
        self,
        runtime_stack: tuple[str, ...],
        *,
        cached_models: tuple[str, ...],
        preferred_launch_profile: str | None,
        repo_root: str = PROVIDER_REPO_ROOT,
    ) -> str | None:
        if not runtime_stack:
            return None
        runtime_args = " ".join(shlex.quote(runtime) for runtime in runtime_stack)
        env_parts = [f"REPO_ROOT={shlex.quote(repo_root)}"]
        if cached_models:
            env_parts.append(
                "PROVIDER_CACHED_MODELS_JSON="
                + shlex.quote(json.dumps(list(cached_models), separators=(",", ":")))
            )
        if preferred_launch_profile is not None:
            env_parts.append(
                f"PROVIDER_LAUNCH_PROFILE={shlex.quote(preferred_launch_profile)}"
            )
        env_parts.extend(
            [
                f"PROVIDER_BOOTSTRAP_MIN_FREE_DISK_GIB={PROVIDER_RUNTIME_BOOTSTRAP_MIN_FREE_DISK_GIB}",
                f"PROVIDER_BOOTSTRAP_RETRIES={PROVIDER_RUNTIME_BOOTSTRAP_RETRIES}",
                f"PROVIDER_BOOTSTRAP_TIMEOUT_SECONDS={PROVIDER_RUNTIME_BOOTSTRAP_TIMEOUT_SECONDS}",
                f"PROVIDER_BOOTSTRAP_PAUSE_SECONDS={PROVIDER_RUNTIME_BOOTSTRAP_PAUSE_SECONDS}",
            ]
        )
        env_expr = " ".join(env_parts)
        return (
            f"mkdir -p /opt && "
            f"if [ ! -d {shlex.quote(repo_root)}/.git ]; then git clone "
            f"{shlex.quote(self.repo_clone_url)} {shlex.quote(repo_root)}; fi && "
            f"cd {shlex.quote(repo_root)} && "
            f"git fetch origin && git checkout {shlex.quote(self.repo_branch)} && "
            f"git pull --ff-only origin {shlex.quote(self.repo_branch)} && "
            f"nohup env {env_expr} bash scripts/runtime/bootstrap_provider_node.sh {runtime_args} "
            f">{shlex.quote(PROVIDER_RUNTIME_BOOTSTRAP_LOG)} 2>&1 < /dev/null &"
        )

    def _reconcile_job(self, job: ProvisionJob, registry: NodeRegistry) -> ProvisionJob:
        if job.status in {"failed", "destroyed"} or job.bootstrap_bundle is None:
            return job
        if job.provisioned_resource is not None and job.provisioned_resource.status == "destroyed":
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
        reconciled = replace(
            job,
            status="joined",
            updated_at=now,
            provisioned_resource=resource,
            joined_node_id=node_id,
            joined_at=joined_at,
            joined_node_snapshot=record.node.to_dict(),
        )
        reconciled = self._reconcile_preflight(reconciled, record.node)
        reconciled = self._reconcile_runtime_bootstrap(reconciled, record.node)
        return self._reconcile_repairs(reconciled, record.node, registry)

    def _reconcile_runtime_bootstrap(
        self,
        job: ProvisionJob,
        node,
    ) -> ProvisionJob:
        bundle = job.bootstrap_bundle
        if bundle is None or not bundle.runtime_stack:
            if job.runtime_bootstrap_status in {None, "pending"}:
                return replace(
                    job,
                    runtime_bootstrap_status="skipped",
                    runtime_bootstrap_note="No runtime bootstrap was requested for this job.",
                )
            return job

        now = utc_now()
        if self._node_has_runtime_stack(node, bundle.runtime_stack):
            return replace(
                job,
                runtime_bootstrap_status="ready",
                runtime_bootstrap_finished_at=job.runtime_bootstrap_finished_at or now,
                runtime_bootstrap_note=(
                    f"Detected installed runtime stack on joined node: {', '.join(bundle.runtime_stack)}."
                ),
            )

        if job.runtime_bootstrap_status in {None, "pending", "stabilizing"}:
            if self._runtime_bootstrap_should_stabilize(job, now):
                remaining = self._runtime_bootstrap_stabilization_remaining_seconds(job, now)
                return replace(
                    job,
                    updated_at=now,
                    runtime_bootstrap_status="stabilizing",
                    runtime_bootstrap_note=(
                        "Joined node is stabilizing before runtime bootstrap starts."
                        if remaining <= 0
                        else (
                            f"Joined node is stabilizing for another {int(remaining)}s "
                            "before runtime bootstrap starts."
                        )
                    ),
                )
            try:
                self._start_runtime_bootstrap(job)
            except ProviderError as exc:
                return replace(
                    job,
                    updated_at=now,
                    runtime_bootstrap_status="failed",
                    runtime_bootstrap_finished_at=now,
                    runtime_bootstrap_note=str(exc),
                )
            return replace(
                job,
                updated_at=now,
                runtime_bootstrap_status="starting",
                runtime_bootstrap_started_at=job.runtime_bootstrap_started_at or now,
                runtime_bootstrap_command=job.runtime_bootstrap_command or bundle.runtime_bootstrap_command,
                runtime_bootstrap_note=(
                    f"Started provider runtime bootstrap for {', '.join(bundle.runtime_stack)}."
                ),
            )

        if job.runtime_bootstrap_status == "starting":
            return replace(
                job,
                updated_at=now,
                runtime_bootstrap_note=(
                    f"Awaiting provider runtime bootstrap completion for {', '.join(bundle.runtime_stack)}."
                ),
            )
        return job

    def _reconcile_preflight(
        self,
        job: ProvisionJob,
        node,
    ) -> ProvisionJob:
        bundle = job.bootstrap_bundle
        runtime_stack = bundle.runtime_stack if bundle is not None else ()
        preferred_launch_profile = (
            bundle.preferred_launch_profile if bundle is not None else None
        )
        report = build_node_preflight_report(
            node,
            provider=job.request.provider,
            runtime_stack=runtime_stack,
            preferred_launch_profile=preferred_launch_profile,
        )
        return replace(
            job,
            updated_at=utc_now(),
            preflight_status=report.status,
            preflight_report=report,
        )

    def _reconcile_repairs(
        self,
        job: ProvisionJob,
        node,
        registry: NodeRegistry,
    ) -> ProvisionJob:
        report = job.preflight_report
        if report is None:
            return job
        now = utc_now()
        actions_by_runtime: dict[str, RepairAction] = {}
        action_list: list[RepairAction] = []
        for action in job.repair_actions:
            if action.runtime is not None:
                actions_by_runtime[action.runtime] = action
            action_list.append(action)

        statuses: list[RuntimeInstallStatus] = []
        updated = False

        for runtime_report in report.runtime_reports:
            existing_action = actions_by_runtime.get(runtime_report.runtime)

            if (
                runtime_report.status == "repairable"
                and existing_action is None
                and self._should_start_runtime_repair(job, runtime_report)
            ):
                try:
                    started_action = self._start_runtime_repair(job, runtime_report)
                except ProviderError as exc:
                    started_action = RepairAction(
                        action_id=f"repair-{runtime_report.runtime}-{secrets.token_hex(4)}",
                        node_id=node.node_id,
                        runtime=runtime_report.runtime,
                        kind=self._repair_kind_for_runtime_report(runtime_report) or "repair",
                        status="failed",
                        started_at=now,
                        finished_at=now,
                        auto_retryable=True,
                        requires_reboot=False,
                        user_visible_label=self._repair_label_for_runtime(runtime_report),
                        detail=str(exc),
                        log_path=PROVIDER_RUNTIME_REPAIR_LOG,
                    )
                    runtime_status = RuntimeInstallStatus(
                        runtime=runtime_report.runtime,
                        status="failed",
                        detected_version=runtime_report.detected_version,
                        last_report_id=report.report_id,
                        active_action_id=started_action.action_id,
                        last_error=str(exc),
                        note=str(exc),
                    )
                else:
                    runtime_status = RuntimeInstallStatus(
                        runtime=runtime_report.runtime,
                        status="repairing",
                        detected_version=runtime_report.detected_version,
                        last_report_id=report.report_id,
                        active_action_id=started_action.action_id,
                        note=started_action.user_visible_label,
                    )
                action_list.append(started_action)
                actions_by_runtime[runtime_report.runtime] = started_action
                updated = True

        active_repairs = {
            runtime_name: action
            for runtime_name, action in actions_by_runtime.items()
            if action.status == "running"
        }

        refreshed_node = None
        refreshed_report = report
        if active_repairs:
            refreshed_node, refreshed_report, action_list, updated = self._refresh_after_runtime_repairs(
                job,
                registry=registry,
                current_report=report,
                active_repairs=active_repairs,
                action_list=action_list,
                updated=updated,
            )

        final_report = refreshed_report
        for runtime_report in final_report.runtime_reports:
            active_action = next(
                (
                    action
                    for action in action_list
                    if action.runtime == runtime_report.runtime
                ),
                None,
            )
            runtime_status = self._build_runtime_install_status(
                job,
                report_id=final_report.report_id,
                runtime_report=runtime_report,
                active_action=active_action,
            )
            statuses.append(runtime_status)

        if (
            not updated
            and tuple(statuses) == job.runtime_install_statuses
            and final_report.to_dict() == report.to_dict()
        ):
            return job

        return replace(
            job,
            updated_at=now,
            joined_node_snapshot=(
                refreshed_node.to_dict() if refreshed_node is not None else job.joined_node_snapshot
            ),
            preflight_status=final_report.status,
            preflight_report=final_report,
            repair_actions=tuple(action_list),
            runtime_install_statuses=tuple(statuses),
        )

    def _job_matches_filters(
        self,
        job: ProvisionJob,
        *,
        job_id: str | None = None,
        provider: str | None = None,
        resource_id: str | None = None,
        resource_status: str | None = None,
        status: str | None = None,
    ) -> bool:
        if job_id is not None and job.job_id != job_id:
            return False
        if provider is not None and not self._job_matches_provider(job, provider):
            return False
        if resource_id is not None and not self._job_matches_resource_id(job, resource_id):
            return False
        if resource_status is not None:
            resource = job.provisioned_resource
            if resource is None or resource.status != resource_status:
                return False
        if status is not None and job.status != status:
            return False
        return True

    def _job_matches_provider(self, job: ProvisionJob, provider: str) -> bool:
        if job.request.provider == provider:
            return True
        if job.selected_offer is not None and job.selected_offer.provider == provider:
            return True
        if job.provisioned_resource is not None and job.provisioned_resource.provider == provider:
            return True
        return False

    def _job_matches_resource_id(self, job: ProvisionJob, resource_id: str) -> bool:
        resource = job.provisioned_resource
        return resource is not None and resource.resource_id == resource_id

    def _resolve_destroy_job(
        self,
        *,
        job_id: str | None = None,
        resource_id: str | None = None,
        provider: str | None = None,
    ) -> ProvisionJob:
        if job_id is not None:
            job = self.get_job(job_id)
            if job is None:
                raise ProviderError(f"Unknown provisioning job id: {job_id}")
            if provider is not None and not self._job_matches_provider(job, provider):
                raise ProviderError(
                    f"Provisioning job {job_id!r} does not belong to provider {provider!r}"
                )
            if resource_id is not None and not self._job_matches_resource_id(job, resource_id):
                raise ProviderError(
                    f"Provisioning job {job_id!r} does not match resource id {resource_id!r}"
                )
            return job

        jobs = self.list_jobs(provider=provider, resource_id=resource_id)
        if not jobs:
            selector_bits = []
            if provider is not None:
                selector_bits.append(f"provider={provider!r}")
            if resource_id is not None:
                selector_bits.append(f"resource_id={resource_id!r}")
            selector = ", ".join(selector_bits) or "unspecified selector"
            raise ProviderError(f"No provisioning job matched {selector}")
        if len(jobs) > 1:
            raise ProviderError(
                "Multiple provisioning jobs matched the destroy selector; use --job-id to disambiguate"
            )
        return jobs[0]

    def _refresh_after_runtime_repairs(
        self,
        job: ProvisionJob,
        *,
        registry: NodeRegistry,
        current_report,
        active_repairs: dict[str, RepairAction],
        action_list: list[RepairAction],
        updated: bool,
    ):
        now = utc_now()
        try:
            refreshed_node = self._probe_joined_node_inventory(job)
        except ProviderError as exc:
            refreshed_report = current_report
            refreshed_actions = [
                replace(
                    action,
                    detail=(
                        f"{action.detail} Inventory refresh pending: {exc}"
                        if action.action_id in {item.action_id for item in active_repairs.values()}
                        else action.detail
                    ),
                )
                for action in action_list
            ]
            return None, refreshed_report, refreshed_actions, True

        registry.register_heartbeat(
            refreshed_node,
            received_at=now,
            source="provider-repair-probe",
        )
        bundle = job.bootstrap_bundle
        refreshed_report = build_node_preflight_report(
            refreshed_node,
            provider=job.request.provider,
            runtime_stack=bundle.runtime_stack if bundle is not None else (),
            preferred_launch_profile=(
                bundle.preferred_launch_profile if bundle is not None else None
            ),
        )

        refreshed_actions: list[RepairAction] = []
        refreshed_by_runtime = {
            report.runtime: report for report in refreshed_report.runtime_reports
        }
        for action in action_list:
            if action.runtime is None or action.action_id not in {item.action_id for item in active_repairs.values()}:
                refreshed_actions.append(action)
                continue
            runtime_report = refreshed_by_runtime.get(action.runtime)
            if runtime_report is not None and runtime_report.status == "ready":
                refreshed_actions.append(
                    replace(
                        action,
                        status="completed",
                        finished_at=now,
                        detail="Repair completed after follow-up preflight.",
                    )
                )
                updated = True
            elif runtime_report is not None and runtime_report.status == "repairable":
                refreshed_actions.append(
                    replace(
                        action,
                        status="failed",
                        finished_at=now,
                        detail="Repair command completed, but follow-up preflight still reports an auto-repairable gap.",
                    )
                )
                updated = True
            else:
                refreshed_actions.append(action)
        return (
            refreshed_node,
            refreshed_report,
            refreshed_actions,
            updated or refreshed_report.to_dict() != current_report.to_dict(),
        )

    def _build_runtime_install_status(
        self,
        job: ProvisionJob,
        *,
        report_id: str,
        runtime_report: RuntimeRequirementReport,
        active_action: RepairAction | None,
    ) -> RuntimeInstallStatus:
        if runtime_report.status == "ready":
            return RuntimeInstallStatus(
                runtime=runtime_report.runtime,
                status="ready",
                detected_version=runtime_report.detected_version,
                last_report_id=report_id,
                active_action_id=active_action.action_id if active_action is not None else None,
                note="Runtime passed preflight.",
            )
        if job.runtime_bootstrap_status == "stabilizing" and self._runtime_is_waiting_for_install(runtime_report):
            return RuntimeInstallStatus(
                runtime=runtime_report.runtime,
                status="stabilizing",
                detected_version=runtime_report.detected_version,
                last_report_id=report_id,
                note="Joined node is stabilizing before runtime bootstrap begins.",
            )
        if (
            job.runtime_bootstrap_status in {"pending", "starting"}
            and self._runtime_is_waiting_for_install(runtime_report)
        ):
            return RuntimeInstallStatus(
                runtime=runtime_report.runtime,
                status="installing",
                detected_version=runtime_report.detected_version,
                last_report_id=report_id,
                note="Provider runtime bootstrap is still in progress.",
            )
        if active_action is not None and active_action.status == "running":
            return RuntimeInstallStatus(
                runtime=runtime_report.runtime,
                status="repairing",
                detected_version=runtime_report.detected_version,
                last_report_id=report_id,
                active_action_id=active_action.action_id,
                note=active_action.user_visible_label or "Runtime repair is in progress.",
            )
        if active_action is not None and active_action.status == "failed":
            return RuntimeInstallStatus(
                runtime=runtime_report.runtime,
                status="failed",
                detected_version=runtime_report.detected_version,
                last_report_id=report_id,
                active_action_id=active_action.action_id,
                last_error=active_action.detail,
                note=active_action.detail or "Runtime repair failed.",
            )
        if runtime_report.status == "repairable":
            return RuntimeInstallStatus(
                runtime=runtime_report.runtime,
                status="repairable",
                detected_version=runtime_report.detected_version,
                last_report_id=report_id,
                active_action_id=active_action.action_id if active_action is not None else None,
                note="Runtime has auto-repairable preflight gaps.",
            )
        return RuntimeInstallStatus(
            runtime=runtime_report.runtime,
            status="failed",
            detected_version=runtime_report.detected_version,
            last_report_id=report_id,
            active_action_id=active_action.action_id if active_action is not None else None,
            note="Runtime preflight failed.",
        )

    def _runtime_is_waiting_for_install(self, runtime_report: RuntimeRequirementReport) -> bool:
        for requirement in runtime_report.requirements:
            if requirement.key == "installed" and requirement.status == "fail":
                return True
        return False

    def _repair_kind_for_runtime_report(
        self,
        runtime_report: RuntimeRequirementReport,
    ) -> str | None:
        if runtime_report.runtime == "deepspeed":
            for requirement in runtime_report.requirements:
                if requirement.key == "nvcc" and requirement.status == "fail":
                    return "repair_deepspeed_nvcc"
        return None

    def _repair_label_for_runtime(
        self,
        runtime_report: RuntimeRequirementReport,
    ) -> str:
        kind = self._repair_kind_for_runtime_report(runtime_report)
        if kind == "repair_deepspeed_nvcc":
            return "Repairing DeepSpeed CUDA toolkit layout"
        return f"Repairing runtime {runtime_report.runtime}"

    def _should_start_runtime_repair(
        self,
        job: ProvisionJob,
        runtime_report: RuntimeRequirementReport,
    ) -> bool:
        kind = self._repair_kind_for_runtime_report(runtime_report)
        if kind is None:
            return False
        if job.runtime_bootstrap_status in {None, "pending", "starting", "stabilizing"}:
            return False
        resource = job.provisioned_resource
        if resource is None or not resource.host or not resource.ssh_user or not resource.ssh_port:
            return False
        return True

    def _build_runtime_repair_command(
        self,
        runtime: str,
        *,
        repo_root: str = PROVIDER_REPO_ROOT,
    ) -> str:
        return (
            f"cd {shlex.quote(repo_root)} && "
            f"env REPO_ROOT={shlex.quote(repo_root)} "
            f"bash scripts/runtime/repair_provider_node.sh {shlex.quote(runtime)} "
            f">> {shlex.quote(PROVIDER_RUNTIME_REPAIR_LOG)} 2>&1"
        )

    def _start_runtime_repair(
        self,
        job: ProvisionJob,
        runtime_report: RuntimeRequirementReport,
    ) -> RepairAction:
        resource = job.provisioned_resource
        bundle = job.bootstrap_bundle
        if resource is None or bundle is None:
            raise ProviderError("Provisioned resource is not available for runtime repair")
        if not resource.host or not resource.ssh_user or not resource.ssh_port:
            raise ProviderError("Provisioned resource is missing SSH connection details for runtime repair")
        command = self._build_runtime_repair_command(runtime_report.runtime)
        ssh_command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-p",
            str(resource.ssh_port),
            f"{resource.ssh_user}@{resource.host}",
            command,
        ]
        result = subprocess.run(
            ssh_command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            detail = stderr or stdout or "unknown ssh error"
            raise ProviderError(f"Provider runtime repair SSH command failed: {detail}")
        now = utc_now()
        return RepairAction(
            action_id=f"repair-{runtime_report.runtime}-{secrets.token_hex(4)}",
            node_id=bundle.node_id,
            runtime=runtime_report.runtime,
            kind=self._repair_kind_for_runtime_report(runtime_report) or "repair",
            status="running",
            started_at=now,
            auto_retryable=True,
            requires_reboot=False,
            user_visible_label=self._repair_label_for_runtime(runtime_report),
            detail=f"Started runtime repair for {runtime_report.runtime}.",
            log_path=PROVIDER_RUNTIME_REPAIR_LOG,
        )

    def _build_remote_inventory_probe_command(
        self,
        job: ProvisionJob,
    ) -> list[str]:
        bundle = job.bootstrap_bundle
        resource = job.provisioned_resource
        if bundle is None or resource is None:
            raise ProviderError("Cannot build remote inventory probe command without bootstrap bundle and resource")
        labels = [f"{key}={value}" for key, value in bundle.labels]
        command = [
            "python3",
            "-m",
            "cluster.orchestrator.clusterctl",
            "probe-local",
            "--node-id",
            bundle.node_id,
            "--host",
            resource.host or bundle.node_id,
            "--lease-duration-seconds",
            str(bundle.lease_duration_seconds or 3600),
            "--include-capabilities",
            "--ssh-user",
            resource.ssh_user or "root",
            "--repo-root",
            PROVIDER_REPO_ROOT,
        ]
        if resource.ssh_port is not None:
            command.extend(["--ssh-port", str(resource.ssh_port)])
        if bundle.trust_tier is not None:
            command.extend(["--trust-tier", bundle.trust_tier])
        if bundle.network_tier is not None:
            command.extend(["--network-tier", bundle.network_tier])
        for value in bundle.cached_models:
            command.extend(["--cached-model", value])
        for value in labels:
            command.extend(["--label", value])
        return command

    def _probe_joined_node_inventory(
        self,
        job: ProvisionJob,
    ):
        bundle = job.bootstrap_bundle
        resource = job.provisioned_resource
        if bundle is None or resource is None:
            raise ProviderError("Provisioned job is missing bootstrap bundle or resource metadata")
        if not resource.host or not resource.ssh_user or not resource.ssh_port:
            raise ProviderError("Provisioned resource is missing SSH connection details for inventory probe")
        command = build_remote_ssh_argv(
            host=resource.host,
            remote_command=self._build_remote_inventory_probe_command(job),
            ssh_user=resource.ssh_user,
            ssh_port=resource.ssh_port,
            repo_root=PROVIDER_REPO_ROOT,
            gpu_index=None,
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            detail = stderr or stdout or "unknown ssh error"
            raise ProviderError(f"Provider inventory refresh SSH command failed: {detail}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Provider inventory refresh returned invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ProviderError("Provider inventory refresh did not return a JSON object")
        return NodeInventory.from_dict(payload)

    def _node_has_runtime_stack(self, node, runtime_stack: tuple[str, ...]) -> bool:
        available = {
            capability.name
            for capability in getattr(node, "runtime_capabilities", ())
            if getattr(capability, "installed", False)
        }
        return all(runtime in available for runtime in runtime_stack)

    def _start_runtime_bootstrap(self, job: ProvisionJob) -> None:
        bundle = job.bootstrap_bundle
        resource = job.provisioned_resource
        if bundle is None or resource is None or bundle.runtime_bootstrap_command is None:
            raise ProviderError("Runtime bootstrap command is not available for this job")
        if not resource.host or not resource.ssh_user or not resource.ssh_port:
            raise ProviderError("Provisioned resource is missing SSH connection details for runtime bootstrap")
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-p",
            str(resource.ssh_port),
            f"{resource.ssh_user}@{resource.host}",
            bundle.runtime_bootstrap_command,
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            detail = stderr or stdout or "unknown ssh error"
            raise ProviderError(f"Provider runtime bootstrap SSH command failed: {detail}")

    def _runtime_bootstrap_should_stabilize(self, job: ProvisionJob, now) -> bool:
        if job.runtime_bootstrap_status not in {None, "pending", "stabilizing"}:
            return False
        if job.joined_at is None:
            return False
        return self._runtime_bootstrap_stabilization_remaining_seconds(job, now) > 0

    def _runtime_bootstrap_stabilization_remaining_seconds(self, job: ProvisionJob, now) -> float:
        if job.joined_at is None:
            return 0.0
        elapsed = max((now - job.joined_at).total_seconds(), 0.0)
        return max(PROVIDER_RUNTIME_BOOTSTRAP_STABILIZATION_SECONDS - elapsed, 0.0)

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
