from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from datetime import timedelta
from unittest.mock import patch

from cluster.models import GPUInventory, LeaseInfo, NodeInventory, NodeSystemInfo, RuntimeCapability, parse_datetime, utc_now
from cluster.node_agent.preflight import build_node_preflight_report
from cluster.node_agent.preflight_models import NodePreflightReport
from cluster.orchestrator.clusterctl import main as clusterctl_main
from cluster.orchestrator.registry import NodeRegistry
from cluster.orchestrator.state_store import RegistryStateStore
from cluster.providers.blueprints import load_builtin_blueprints
from cluster.providers.job_store import ProvisionJobStore
from cluster.providers.models import ProviderOffer, ProvisionRequest, ProvisionedResource, RepairAction
from cluster.providers.vast_adapter import VastAdapter
from cluster.providers.service import ProviderService


class ClusterPhase4ProviderTests(unittest.TestCase):
    def test_node_preflight_report_marks_missing_deepspeed_nvcc_as_repairable(self) -> None:
        node = NodeInventory(
            node_id="node-preflight-1",
            host="203.0.113.99",
            lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
            gpus=(
                GPUInventory(index=0, uuid="gpu-0", total_memory_mib=32607, free_memory_mib=32000),
                GPUInventory(index=1, uuid="gpu-1", total_memory_mib=32607, free_memory_mib=32000),
            ),
            system_info=NodeSystemInfo(
                hostname="node-preflight-1",
                python_version="Python 3.10.12",
                driver_version="580.95.05",
                cuda_version="13.0",
            ),
            runtime_capabilities=(
                RuntimeCapability(
                    name="deepspeed",
                    installed=True,
                    version="0.18.9",
                    executable="/opt/Claude-Code-Game-Studios/.venv-deepspeed/bin/deepspeed",
                    supported_topologies=("single-gpu", "tp", "pp"),
                    details={
                        "python_executable": "/opt/Claude-Code-Game-Studios/.venv-deepspeed/bin/python",
                        "system_nvcc_present": False,
                        "packaged_nvcc_present": False,
                        "runtime_nvcc_present": False,
                    },
                ),
            ),
        )

        report = build_node_preflight_report(
            node,
            provider="vast",
            runtime_stack=("deepspeed",),
            preferred_launch_profile="qwen-coder-30b-deepspeed-tp2",
        )

        self.assertEqual(report.status, "repairable")
        runtime_report = report.runtime_reports[0]
        self.assertEqual(runtime_report.runtime, "deepspeed")
        self.assertEqual(runtime_report.status, "repairable")
        nvcc_requirement = next(
            requirement for requirement in runtime_report.requirements if requirement.key == "nvcc"
        )
        self.assertEqual(nvcc_requirement.status, "fail")
        self.assertTrue(nvcc_requirement.auto_repairable)

    def test_node_preflight_report_roundtrip_is_stable(self) -> None:
        node = NodeInventory(
            node_id="node-preflight-2",
            host="203.0.113.100",
            lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
            gpus=(
                GPUInventory(index=0, uuid="gpu-0", total_memory_mib=32607, free_memory_mib=32000),
                GPUInventory(index=1, uuid="gpu-1", total_memory_mib=32607, free_memory_mib=32000),
            ),
            system_info=NodeSystemInfo(
                hostname="node-preflight-2",
                python_version="Python 3.10.12",
                driver_version="580.95.05",
                cuda_version="13.0",
            ),
            runtime_capabilities=(
                RuntimeCapability(
                    name="vllm",
                    installed=True,
                    version="0.19.0",
                    supported_topologies=("single-gpu", "tp"),
                    details={"python_executable": "/opt/Claude-Code-Game-Studios/.venv-vllm/bin/python"},
                ),
            ),
        )

        report = build_node_preflight_report(
            node,
            provider="vast",
            runtime_stack=("vllm",),
            preferred_launch_profile="qwen-coder-30b-vllm-tp2",
        )

        payload_a = json.dumps(report.to_dict(), sort_keys=True)
        payload_b = json.dumps(NodePreflightReport.from_dict(report.to_dict()).to_dict(), sort_keys=True)

        self.assertEqual(payload_a, payload_b)

    def test_provider_offer_roundtrip_is_stable(self) -> None:
        offer = ProviderOffer(
            provider="vast",
            offer_id="offer-123",
            resource_kind="instance",
            region="EU",
            datacenter="eu-demo-1",
            gpu_name="NVIDIA GeForce RTX 5090",
            gpu_count=2,
            vram_gb=32.0,
            price_hourly=1.75,
            currency="USD",
            availability_mode="ondemand",
            preemptible=False,
            supports_template=True,
            supports_public_ip=True,
            supports_volume=True,
            raw_provider_payload={"id": "offer-123", "demo": True},
        )

        payload_a = json.dumps(offer.to_dict(), sort_keys=True)
        payload_b = json.dumps(ProviderOffer.from_dict(offer.to_dict()).to_dict(), sort_keys=True)

        self.assertEqual(payload_a, payload_b)

    def test_builtin_blueprints_cover_expected_presets(self) -> None:
        blueprints = {item.blueprint_id: item for item in load_builtin_blueprints()}

        self.assertIn("qwen-coder-node-vast", blueprints)
        self.assertIn("gemma-review-node-runpod", blueprints)
        self.assertIn("trusted-nebius-worker", blueprints)
        self.assertIn("custom-gpu-node", blueprints)
        self.assertEqual(blueprints["qwen-coder-node-vast"].provider, "vast")
        self.assertEqual(blueprints["gemma-review-node-runpod"].provider, "runpod")
        self.assertEqual(blueprints["trusted-nebius-worker"].trust_tier, "trusted")
        self.assertEqual(blueprints["qwen-coder-node-vast"].runtime_stack, ("vllm",))
        self.assertEqual(
            blueprints["qwen-coder-node-vast"].preferred_launch_profile,
            "qwen-coder-30b-vllm-tp2",
        )

    def test_bootstrap_bundle_includes_runtime_bootstrap_plan(self) -> None:
        service = ProviderService()

        bundle = service.build_bootstrap_bundle(
            ProvisionRequest(
                provider="vast",
                blueprint_id="qwen-coder-node-vast",
                dry_run=True,
            ),
            blueprint={
                item.blueprint_id: item for item in load_builtin_blueprints()
            }["qwen-coder-node-vast"],
        )

        self.assertEqual(bundle.runtime_stack, ("vllm",))
        self.assertEqual(bundle.preferred_launch_profile, "qwen-coder-30b-vllm-tp2")
        self.assertIsNotNone(bundle.runtime_bootstrap_command)
        assert bundle.runtime_bootstrap_command is not None
        self.assertIn("bootstrap_provider_node.sh vllm", bundle.runtime_bootstrap_command)
        self.assertIn("PROVIDER_LAUNCH_PROFILE", bundle.runtime_bootstrap_command)
        self.assertIn("PROVIDER_BOOTSTRAP_MIN_FREE_DISK_GIB", bundle.runtime_bootstrap_command)
        self.assertEqual(
            bundle.node_agent_config["preferred_launch_profile"],
            "qwen-coder-30b-vllm-tp2",
        )
        self.assertEqual(bundle.node_agent_config["bootstrap_mode"], "staged")
        self.assertEqual(bundle.node_agent_config["runtime_bootstrap_min_free_disk_gib"], 10)

    def test_provider_service_lists_normalized_demo_offers(self) -> None:
        service = ProviderService()

        offers = service.list_offers()

        self.assertGreaterEqual(len(offers), 6)
        providers = {offer.provider for offer in offers}
        self.assertEqual(providers, {"vast", "runpod", "nebius"})

    def test_provider_service_filters_offers(self) -> None:
        service = ProviderService()

        offers = service.list_offers(provider="runpod", min_gpu_count=4)

        self.assertEqual(len(offers), 1)
        self.assertEqual(offers[0].provider, "runpod")
        self.assertEqual(offers[0].gpu_count, 4)

    def test_dry_run_provision_creates_bootstrapping_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_file = Path(tmpdir) / "jobs.json"
            service = ProviderService(jobs_file=jobs_file)

            job = service.provision(
                ProvisionRequest(
                    provider="vast",
                    blueprint_id="qwen-coder-node-vast",
                    dry_run=True,
                )
            )

            self.assertEqual(job.status, "bootstrapping")
            assert job.selected_offer is not None
            assert job.bootstrap_bundle is not None
            assert job.provisioned_resource is not None
            self.assertEqual(job.selected_offer.provider, "vast")
            self.assertEqual(job.provisioned_resource.status, "dry-run")
            self.assertIn("node_id", job.bootstrap_bundle.node_agent_config)
            self.assertEqual(job.runtime_bootstrap_status, "pending")
            self.assertIn("bootstrap_provider_node.sh", job.runtime_bootstrap_command or "")
            reloaded_jobs = ProvisionJobStore(jobs_file).load()
            self.assertEqual(len(reloaded_jobs), 1)
            self.assertEqual(reloaded_jobs[0].job_id, job.job_id)

    def test_custom_blueprint_accepts_concrete_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = ProviderService(jobs_file=Path(tmpdir) / "jobs.json")

            job = service.provision(
                ProvisionRequest(
                    provider="runpod",
                    blueprint_id="custom-gpu-node",
                    offer_id="NVIDIA GeForce RTX 5090:SECURE",
                    dry_run=True,
                )
            )

            self.assertEqual(job.request.provider, "runpod")
            self.assertEqual(job.status, "bootstrapping")

    def test_registry_and_provider_jobs_remain_separate(self) -> None:
        registry = NodeRegistry()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = ProviderService(jobs_file=Path(tmpdir) / "jobs.json")
            _ = service.provision(
                ProvisionRequest(
                    provider="nebius",
                    blueprint_id="trusted-nebius-worker",
                    dry_run=True,
                )
            )

            self.assertEqual(registry.list_nodes(), [])

    def test_clusterctl_lists_provider_blueprints(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = clusterctl_main(["providers-list-blueprints", "--provider", "vast"])
        payload = json.loads(buffer.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["blueprints"][0]["provider"], "vast")

    def test_clusterctl_lists_provider_offers(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = clusterctl_main(
                [
                    "providers-list-offers",
                    "--provider",
                    "runpod",
                    "--min-gpu-count",
                    "2",
                ]
            )
        payload = json.loads(buffer.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["offers"])
        self.assertEqual(payload["offers"][0]["provider"], "runpod")

    def test_clusterctl_creates_and_reads_provider_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_file = Path(tmpdir) / "provider-jobs.json"

            create_buffer = io.StringIO()
            with redirect_stdout(create_buffer):
                exit_code = clusterctl_main(
                    [
                        "providers-provision",
                        "--provider",
                        "vast",
                        "--blueprint",
                        "qwen-coder-node-vast",
                        "--dry-run",
                        "--jobs-file",
                        str(jobs_file),
                    ]
                )
            created = json.loads(create_buffer.getvalue())

            read_buffer = io.StringIO()
            with redirect_stdout(read_buffer):
                read_exit_code = clusterctl_main(
                    [
                        "providers-jobs",
                        "--jobs-file",
                        str(jobs_file),
                        "--job-id",
                        created["job_id"],
                    ]
                )
            loaded = json.loads(read_buffer.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(read_exit_code, 0)
        self.assertEqual(created["job_id"], loaded["job_id"])
        self.assertEqual(loaded["status"], "bootstrapping")

    def test_provider_service_preserves_explicit_offer_id_when_not_in_listing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = ProviderService(jobs_file=Path(tmpdir) / "jobs.json")

            job = service.provision(
                ProvisionRequest(
                    provider="vast",
                    blueprint_id="custom-gpu-node",
                    offer_id="offer-manual-123",
                    dry_run=True,
                )
            )

        assert job.selected_offer is not None
        self.assertEqual(job.selected_offer.offer_id, "offer-manual-123")
        self.assertEqual(job.provisioned_resource.offer_id, "offer-manual-123")

    def test_provider_service_reconciles_job_when_node_joins_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = ProviderService(jobs_file=Path(tmpdir) / "jobs.json")
            job = service.provision(
                ProvisionRequest(
                    provider="vast",
                    blueprint_id="qwen-coder-node-vast",
                    dry_run=True,
                )
            )
            assert job.bootstrap_bundle is not None
            registry = NodeRegistry()
            registry.register(
                NodeInventory(
                    node_id=job.bootstrap_bundle.node_id,
                    host="198.51.100.25",
                    lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
                    gpus=(
                        GPUInventory(index=0, uuid="gpu-0", total_memory_mib=32607, free_memory_mib=32000),
                        GPUInventory(index=1, uuid="gpu-1", total_memory_mib=32607, free_memory_mib=32000),
                    ),
                    system_info=NodeSystemInfo(
                        hostname="joined-node",
                        python_version="Python 3.10.12",
                        driver_version="580.95.05",
                        cuda_version="13.0",
                    ),
                    runtime_capabilities=(
                        RuntimeCapability(
                            name="vllm",
                            installed=True,
                            version="0.19.0",
                            supported_topologies=("single-gpu", "tp"),
                        ),
                    ),
                )
            )

            reconciled_jobs = service.reconcile_jobs(registry)

        self.assertEqual(len(reconciled_jobs), 1)
        updated = reconciled_jobs[0]
        self.assertEqual(updated.status, "joined")
        self.assertEqual(updated.joined_node_id, job.bootstrap_bundle.node_id)
        self.assertIsNotNone(updated.joined_at)
        self.assertEqual(updated.joined_node_snapshot["node_id"], job.bootstrap_bundle.node_id)
        assert updated.provisioned_resource is not None
        self.assertEqual(updated.provisioned_resource.status, "joined")
        self.assertEqual(updated.runtime_bootstrap_status, "ready")
        self.assertEqual(updated.preflight_status, "ready")
        assert updated.preflight_report is not None
        self.assertEqual(updated.preflight_report.node_id, job.bootstrap_bundle.node_id)
        self.assertTrue(updated.preflight_report.runtime_reports)

    def test_provider_service_starts_runtime_bootstrap_after_join(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_file = Path(tmpdir) / "jobs.json"
            service = ProviderService(jobs_file=jobs_file)
            joined_at = utc_now() - timedelta(seconds=300)
            job = service.provision(
                ProvisionRequest(
                    provider="vast",
                    blueprint_id="qwen-coder-node-vast",
                    dry_run=True,
                )
            )
            assert job.bootstrap_bundle is not None
            job = replace(
                job,
                provisioned_resource=ProvisionedResource(
                    provider="vast",
                    resource_id="resource-1",
                    resource_kind="instance",
                    display_name=job.bootstrap_bundle.node_id,
                    host="203.0.113.25",
                    ssh_user="root",
                    ssh_port=40222,
                    status="running",
                    ),
                )
            job = replace(job, joined_at=joined_at)
            service.jobs.upsert(job)
            registry = NodeRegistry()
            registry.register(
                NodeInventory(
                    node_id=job.bootstrap_bundle.node_id,
                    host="203.0.113.25",
                    lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
                    gpus=(
                        GPUInventory(index=0, uuid="gpu-0", total_memory_mib=32607, free_memory_mib=32000),
                        GPUInventory(index=1, uuid="gpu-1", total_memory_mib=32607, free_memory_mib=32000),
                    ),
                    system_info=NodeSystemInfo(
                        hostname="joining-node",
                        python_version="Python 3.10.12",
                        driver_version="580.95.05",
                        cuda_version="13.0",
                    ),
                )
            )

            with patch("cluster.providers.service.subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = ""
                run.return_value.stderr = ""
                reconciled_jobs = service.reconcile_jobs(registry)

        updated = reconciled_jobs[0]
        self.assertEqual(updated.status, "joined")
        self.assertEqual(updated.runtime_bootstrap_status, "starting")
        self.assertEqual(updated.preflight_status, "repairable")
        self.assertEqual(updated.joined_node_id, job.bootstrap_bundle.node_id)
        assert updated.preflight_report is not None
        self.assertEqual(updated.preflight_report.runtime_reports[0].runtime, "vllm")
        assert updated.runtime_bootstrap_command is not None
        self.assertIn("bootstrap_provider_node.sh vllm", updated.runtime_bootstrap_command)
        run.assert_called_once()
        ssh_args = run.call_args.args[0]
        self.assertEqual(ssh_args[:6], ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "-p"])
        self.assertIn("root@203.0.113.25", ssh_args)

    def test_provider_service_records_runtime_bootstrap_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_file = Path(tmpdir) / "jobs.json"
            service = ProviderService(jobs_file=jobs_file)
            joined_at = utc_now() - timedelta(seconds=300)
            job = service.provision(
                ProvisionRequest(
                    provider="vast",
                    blueprint_id="qwen-coder-node-vast",
                    dry_run=True,
                )
            )
            assert job.bootstrap_bundle is not None
            job = replace(
                job,
                provisioned_resource=ProvisionedResource(
                    provider="vast",
                    resource_id="resource-1",
                    resource_kind="instance",
                    display_name=job.bootstrap_bundle.node_id,
                    host="203.0.113.26",
                    ssh_user="root",
                    ssh_port=40222,
                    status="running",
                    ),
                )
            job = replace(job, joined_at=joined_at)
            service.jobs.upsert(job)
            registry = NodeRegistry()
            registry.register(
                NodeInventory(
                    node_id=job.bootstrap_bundle.node_id,
                    host="203.0.113.26",
                    lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
                    gpus=(
                        GPUInventory(index=0, uuid="gpu-0", total_memory_mib=32607, free_memory_mib=32000),
                        GPUInventory(index=1, uuid="gpu-1", total_memory_mib=32607, free_memory_mib=32000),
                    ),
                    system_info=NodeSystemInfo(
                        hostname="joining-node-fail",
                        python_version="Python 3.10.12",
                        driver_version="580.95.05",
                        cuda_version="13.0",
                    ),
                )
            )

            with patch("cluster.providers.service.subprocess.run") as run:
                run.return_value.returncode = 255
                run.return_value.stdout = ""
                run.return_value.stderr = "permission denied"
                reconciled_jobs = service.reconcile_jobs(registry)

        updated = reconciled_jobs[0]
        self.assertEqual(updated.runtime_bootstrap_status, "failed")
        self.assertIn("permission denied", updated.runtime_bootstrap_note or "")
        self.assertEqual(updated.preflight_status, "repairable")
        assert updated.preflight_report is not None
        self.assertTrue(updated.preflight_report.runtime_reports)

    def test_provider_service_defers_runtime_bootstrap_for_recent_join(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_file = Path(tmpdir) / "jobs.json"
            service = ProviderService(jobs_file=jobs_file)
            job = service.provision(
                ProvisionRequest(
                    provider="vast",
                    blueprint_id="qwen-coder-node-vast",
                    dry_run=True,
                )
            )
            assert job.bootstrap_bundle is not None
            job = replace(
                job,
                status="joined",
                joined_node_id=job.bootstrap_bundle.node_id,
                joined_at=utc_now(),
                provisioned_resource=ProvisionedResource(
                    provider="vast",
                    resource_id="resource-2",
                    resource_kind="instance",
                    display_name=job.bootstrap_bundle.node_id,
                    host="203.0.113.29",
                    ssh_user="root",
                    ssh_port=40222,
                    status="joined",
                ),
            )
            service.jobs.upsert(job)
            registry = NodeRegistry()
            registry.register(
                NodeInventory(
                    node_id=job.bootstrap_bundle.node_id,
                    host="203.0.113.29",
                    lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
                    gpus=(
                        GPUInventory(index=0, uuid="gpu-0", total_memory_mib=32607, free_memory_mib=32000),
                        GPUInventory(index=1, uuid="gpu-1", total_memory_mib=32607, free_memory_mib=32000),
                    ),
                    system_info=NodeSystemInfo(
                        hostname="recent-join",
                        python_version="Python 3.10.12",
                        driver_version="580.95.05",
                        cuda_version="13.0",
                    ),
                )
            )

            with patch("cluster.providers.service.subprocess.run") as run:
                reconciled_jobs = service.reconcile_jobs(registry)

        updated = reconciled_jobs[0]
        self.assertEqual(updated.runtime_bootstrap_status, "stabilizing")
        self.assertEqual(updated.preflight_status, "repairable")
        assert updated.preflight_report is not None
        run.assert_not_called()

    def test_provider_service_starts_deepspeed_nvcc_repair_after_joined_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_file = Path(tmpdir) / "jobs.json"
            service = ProviderService(jobs_file=jobs_file)
            job = service.provision(
                ProvisionRequest(
                    provider="vast",
                    blueprint_id="custom-gpu-node",
                    dry_run=True,
                    provider_options={
                        "runtime_stack": ["deepspeed"],
                        "launch_profile": "qwen-coder-30b-deepspeed-tp2",
                    },
                )
            )
            assert job.bootstrap_bundle is not None
            job = replace(
                job,
                provisioned_resource=ProvisionedResource(
                    provider="vast",
                    resource_id="resource-deepspeed-1",
                    resource_kind="instance",
                    display_name=job.bootstrap_bundle.node_id,
                    host="203.0.113.27",
                    ssh_user="root",
                    ssh_port=40222,
                    status="running",
                ),
            )
            service.jobs.upsert(job)
            registry = NodeRegistry()
            registry.register(
                NodeInventory(
                    node_id=job.bootstrap_bundle.node_id,
                    host="203.0.113.27",
                    lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
                    gpus=(
                        GPUInventory(index=0, uuid="gpu-0", total_memory_mib=32607, free_memory_mib=32000),
                        GPUInventory(index=1, uuid="gpu-1", total_memory_mib=32607, free_memory_mib=32000),
                    ),
                    system_info=NodeSystemInfo(
                        hostname="repair-node",
                        python_version="Python 3.10.12",
                        driver_version="580.95.05",
                        cuda_version="13.0",
                    ),
                    runtime_capabilities=(
                        RuntimeCapability(
                            name="deepspeed",
                            installed=True,
                            version="0.18.9",
                            executable="/opt/Claude-Code-Game-Studios/.venv-deepspeed/bin/deepspeed",
                            supported_topologies=("single-gpu", "tp", "pp"),
                            details={
                                "python_executable": "/opt/Claude-Code-Game-Studios/.venv-deepspeed/bin/python",
                                "system_nvcc_present": False,
                                "packaged_nvcc_present": False,
                                "runtime_nvcc_present": False,
                            },
                        ),
                    ),
                )
            )

            with patch("cluster.providers.service.subprocess.run") as run:
                refreshed_inventory = NodeInventory(
                    node_id=job.bootstrap_bundle.node_id,
                    host="203.0.113.27",
                    lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
                    gpus=(
                        GPUInventory(index=0, uuid="gpu-0", total_memory_mib=32607, free_memory_mib=32000),
                        GPUInventory(index=1, uuid="gpu-1", total_memory_mib=32607, free_memory_mib=32000),
                    ),
                    system_info=NodeSystemInfo(
                        hostname="repair-node",
                        python_version="Python 3.10.12",
                        driver_version="580.95.05",
                        cuda_version="13.0",
                    ),
                    runtime_capabilities=(
                        RuntimeCapability(
                            name="deepspeed",
                            installed=True,
                            version="0.18.9",
                            executable="/opt/Claude-Code-Game-Studios/.venv-deepspeed/bin/deepspeed",
                            supported_topologies=("single-gpu", "tp", "pp"),
                            details={
                                "python_executable": "/opt/Claude-Code-Game-Studios/.venv-deepspeed/bin/python",
                                "system_nvcc_present": False,
                                "packaged_nvcc_present": False,
                                "runtime_nvcc_present": True,
                            },
                        ),
                    ),
                )
                run.side_effect = [
                    unittest.mock.Mock(returncode=0, stdout="", stderr=""),
                    unittest.mock.Mock(
                        returncode=0,
                        stdout=json.dumps(refreshed_inventory.to_dict()),
                        stderr="",
                    ),
                ]
                reconciled_jobs = service.reconcile_jobs(registry)

        updated = reconciled_jobs[0]
        self.assertEqual(updated.runtime_bootstrap_status, "ready")
        self.assertEqual(updated.preflight_status, "ready")
        self.assertTrue(updated.repair_actions)
        self.assertEqual(updated.repair_actions[0].kind, "repair_deepspeed_nvcc")
        self.assertEqual(updated.repair_actions[0].status, "completed")
        self.assertTrue(updated.runtime_install_statuses)
        self.assertEqual(updated.runtime_install_statuses[0].runtime, "deepspeed")
        self.assertEqual(updated.runtime_install_statuses[0].status, "ready")
        self.assertEqual(run.call_count, 2)
        repair_args = run.call_args_list[0].args[0]
        probe_args = run.call_args_list[1].args[0]
        self.assertIn("root@203.0.113.27", repair_args)
        self.assertIn("repair_provider_node.sh deepspeed", repair_args[-1])
        self.assertIn("root@203.0.113.27", probe_args)
        self.assertIn("probe-local", probe_args[-1])

    def test_provider_service_marks_repair_completed_after_followup_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_file = Path(tmpdir) / "jobs.json"
            service = ProviderService(jobs_file=jobs_file)
            now = parse_datetime("2035-01-01T00:00:00Z")
            job = service.provision(
                ProvisionRequest(
                    provider="vast",
                    blueprint_id="custom-gpu-node",
                    dry_run=True,
                    provider_options={
                        "runtime_stack": ["deepspeed"],
                        "launch_profile": "qwen-coder-30b-deepspeed-tp2",
                    },
                )
            )
            assert job.bootstrap_bundle is not None
            initial_report = build_node_preflight_report(
                NodeInventory(
                    node_id=job.bootstrap_bundle.node_id,
                    host="203.0.113.28",
                    lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
                    gpus=(
                        GPUInventory(index=0, uuid="gpu-0", total_memory_mib=32607, free_memory_mib=32000),
                        GPUInventory(index=1, uuid="gpu-1", total_memory_mib=32607, free_memory_mib=32000),
                    ),
                    system_info=NodeSystemInfo(
                        hostname="repair-node-2",
                        python_version="Python 3.10.12",
                        driver_version="580.95.05",
                        cuda_version="13.0",
                    ),
                    runtime_capabilities=(
                        RuntimeCapability(
                            name="deepspeed",
                            installed=True,
                            version="0.18.9",
                            executable="/opt/Claude-Code-Game-Studios/.venv-deepspeed/bin/deepspeed",
                            supported_topologies=("single-gpu", "tp", "pp"),
                            details={
                                "python_executable": "/opt/Claude-Code-Game-Studios/.venv-deepspeed/bin/python",
                                "system_nvcc_present": False,
                                "packaged_nvcc_present": False,
                                "runtime_nvcc_present": False,
                            },
                        ),
                    ),
                ),
                provider="vast",
                runtime_stack=("deepspeed",),
                preferred_launch_profile="qwen-coder-30b-deepspeed-tp2",
            )
            job = replace(
                job,
                status="joined",
                joined_node_id=job.bootstrap_bundle.node_id,
                joined_at=now,
                provisioned_resource=ProvisionedResource(
                    provider="vast",
                    resource_id="resource-deepspeed-2",
                    resource_kind="instance",
                    display_name=job.bootstrap_bundle.node_id,
                    host="203.0.113.28",
                    ssh_user="root",
                    ssh_port=40222,
                    status="joined",
                ),
                runtime_bootstrap_status="ready",
                preflight_status="repairable",
                preflight_report=initial_report,
                repair_actions=(
                    RepairAction(
                        action_id="repair-deepspeed-test",
                        node_id=job.bootstrap_bundle.node_id,
                        runtime="deepspeed",
                        kind="repair_deepspeed_nvcc",
                        status="running",
                        started_at=now,
                        user_visible_label="Repairing DeepSpeed CUDA toolkit layout",
                        detail="Started runtime repair for deepspeed.",
                    ),
                ),
            )
            service.jobs.upsert(job)
            registry = NodeRegistry()
            registry.register(
                NodeInventory(
                    node_id=job.bootstrap_bundle.node_id,
                    host="203.0.113.28",
                    lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
                    gpus=(
                        GPUInventory(index=0, uuid="gpu-0", total_memory_mib=32607, free_memory_mib=32000),
                        GPUInventory(index=1, uuid="gpu-1", total_memory_mib=32607, free_memory_mib=32000),
                    ),
                    system_info=NodeSystemInfo(
                        hostname="repair-node-2",
                        python_version="Python 3.10.12",
                        driver_version="580.95.05",
                        cuda_version="13.0",
                    ),
                    runtime_capabilities=(
                        RuntimeCapability(
                            name="deepspeed",
                            installed=True,
                            version="0.18.9",
                            executable="/opt/Claude-Code-Game-Studios/.venv-deepspeed/bin/deepspeed",
                            supported_topologies=("single-gpu", "tp", "pp"),
                            details={
                                "python_executable": "/opt/Claude-Code-Game-Studios/.venv-deepspeed/bin/python",
                                "system_nvcc_present": False,
                                "packaged_nvcc_present": False,
                                "runtime_nvcc_present": True,
                            },
                        ),
                    ),
                )
            )

            refreshed_inventory = NodeInventory(
                node_id=job.bootstrap_bundle.node_id,
                host="203.0.113.28",
                lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
                gpus=(
                    GPUInventory(index=0, uuid="gpu-0", total_memory_mib=32607, free_memory_mib=32000),
                    GPUInventory(index=1, uuid="gpu-1", total_memory_mib=32607, free_memory_mib=32000),
                ),
                system_info=NodeSystemInfo(
                    hostname="repair-node-2",
                    python_version="Python 3.10.12",
                    driver_version="580.95.05",
                    cuda_version="13.0",
                ),
                runtime_capabilities=(
                    RuntimeCapability(
                        name="deepspeed",
                        installed=True,
                        version="0.18.9",
                        executable="/opt/Claude-Code-Game-Studios/.venv-deepspeed/bin/deepspeed",
                        supported_topologies=("single-gpu", "tp", "pp"),
                        details={
                            "python_executable": "/opt/Claude-Code-Game-Studios/.venv-deepspeed/bin/python",
                            "system_nvcc_present": False,
                            "packaged_nvcc_present": False,
                            "runtime_nvcc_present": True,
                        },
                    ),
                ),
            )
            with patch("cluster.providers.service.subprocess.run") as run:
                run.return_value = unittest.mock.Mock(
                    returncode=0,
                    stdout=json.dumps(refreshed_inventory.to_dict()),
                    stderr="",
                )
                reconciled_jobs = service.reconcile_jobs(registry)
                run.assert_called_once()

        updated = reconciled_jobs[0]
        self.assertEqual(updated.preflight_status, "ready")
        self.assertEqual(updated.runtime_install_statuses[0].status, "ready")
        self.assertEqual(updated.repair_actions[0].status, "completed")

    def test_provider_service_leaves_job_bootstrapping_when_node_not_joined(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = ProviderService(jobs_file=Path(tmpdir) / "jobs.json")
            job = service.provision(
                ProvisionRequest(
                    provider="nebius",
                    blueprint_id="trusted-nebius-worker",
                    dry_run=True,
                )
            )

            reconciled_jobs = service.reconcile_jobs(NodeRegistry())

        self.assertEqual(len(reconciled_jobs), 1)
        self.assertEqual(reconciled_jobs[0].job_id, job.job_id)
        self.assertEqual(reconciled_jobs[0].status, "bootstrapping")
        self.assertIsNone(reconciled_jobs[0].joined_node_id)

    def test_provider_service_waits_for_join_during_provision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_file = Path(tmpdir) / "jobs.json"
            state_file = Path(tmpdir) / "cluster-registry.json"
            RegistryStateStore(state_file).save(
                NodeRegistry(
                    [
                        NodeInventory(
                            node_id="vast-node-known",
                            host="198.51.100.44",
                            lease=LeaseInfo(
                                available_until=parse_datetime("2035-01-01T00:00:00Z")
                            ),
                            gpus=(
                                GPUInventory(index=0, uuid="gpu-0", total_memory_mib=32607, free_memory_mib=32000),
                                GPUInventory(index=1, uuid="gpu-1", total_memory_mib=32607, free_memory_mib=32000),
                            ),
                            system_info=NodeSystemInfo(
                                hostname="vast-node-known",
                                python_version="Python 3.10.12",
                                driver_version="580.95.05",
                                cuda_version="13.0",
                            ),
                        )
                    ]
                )
            )
            service = ProviderService(jobs_file=jobs_file)

            with patch.object(service, "_generate_node_id", return_value="vast-node-known"):
                job = service.provision_with_join_wait(
                    ProvisionRequest(
                        provider="vast",
                        blueprint_id="qwen-coder-node-vast",
                        dry_run=True,
                    ),
                    wait_for_join=True,
                    join_state_file=state_file,
                    join_timeout_seconds=0.0,
                    join_poll_interval_seconds=0.1,
                )

        self.assertEqual(job.status, "joined")
        self.assertEqual(job.joined_node_id, "vast-node-known")
        assert job.provisioned_resource is not None
        self.assertEqual(job.provisioned_resource.status, "joined")

    def test_vast_adapter_real_create_normalizes_created_instance(self) -> None:
        adapter = VastAdapter(api_key="test-token")
        service = ProviderService(adapters=(adapter,))
        blueprint = {
            item.blueprint_id: item for item in load_builtin_blueprints()
        }["qwen-coder-node-vast"]
        request = ProvisionRequest(
            provider="vast",
            blueprint_id="qwen-coder-node-vast",
            offer_id="12345678",
            dry_run=False,
        )
        effective_request = service._apply_blueprint_defaults(request, blueprint)
        bundle = service.build_bootstrap_bundle(effective_request, blueprint=blueprint)
        selected_offer = ProviderOffer(
            provider="vast",
            offer_id="12345678",
            resource_kind="instance",
            region="EU",
            gpu_name="NVIDIA GeForce RTX 5090",
            gpu_count=2,
            preemptible=False,
        )

        with patch.object(
            adapter,
            "_put_json",
            return_value={"success": True, "new_contract": 99123},
        ) as put_json, patch.object(
            adapter,
            "_get_json",
            return_value={
                "instances": {
                    "id": 99123,
                    "actual_status": "running",
                    "label": bundle.node_id,
                    "ssh_host": "203.0.113.10",
                    "ssh_port": 40222,
                    "public_ipaddr": "203.0.113.10",
                }
            },
        ) as get_json:
            resource = adapter.create_resource(
                effective_request,
                bundle,
                blueprint=blueprint,
                selected_offer=selected_offer,
            )

        self.assertEqual(resource.resource_id, "99123")
        self.assertEqual(resource.status, "running")
        self.assertEqual(resource.host, "203.0.113.10")
        self.assertEqual(resource.ssh_port, 40222)
        self.assertEqual(resource.ssh_user, "root")
        self.assertEqual(resource.offer_id, "12345678")
        put_payload = put_json.call_args.args[1]
        self.assertEqual(put_json.call_args.args[0], "https://console.vast.ai/api/v0/asks/12345678/")
        self.assertEqual(put_payload["runtype"], "ssh_direct")
        self.assertIn("cluster.node_agent.daemon", put_payload["onstart"])
        self.assertEqual(get_json.call_args.args[0], "https://console.vast.ai/api/v0/instances/99123/")

    def test_clusterctl_reconciles_provider_jobs_against_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_file = Path(tmpdir) / "provider-jobs.json"
            state_file = Path(tmpdir) / "cluster-registry.json"

            create_buffer = io.StringIO()
            with redirect_stdout(create_buffer):
                create_exit_code = clusterctl_main(
                    [
                        "providers-provision",
                        "--provider",
                        "vast",
                        "--blueprint",
                        "qwen-coder-node-vast",
                        "--dry-run",
                        "--jobs-file",
                        str(jobs_file),
                    ]
                )
            created = json.loads(create_buffer.getvalue())
            registry = NodeRegistry()
            registry.register(
                NodeInventory(
                    node_id=created["bootstrap_bundle"]["node_id"],
                    host="203.0.113.55",
                    lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
                )
            )
            RegistryStateStore(state_file).save(registry)

            reconcile_buffer = io.StringIO()
            with redirect_stdout(reconcile_buffer):
                reconcile_exit_code = clusterctl_main(
                    [
                        "providers-reconcile-jobs",
                        "--jobs-file",
                        str(jobs_file),
                        "--state-file",
                        str(state_file),
                        "--job-id",
                        created["job_id"],
                    ]
                )
            reconciled = json.loads(reconcile_buffer.getvalue())
            persisted = ProvisionJobStore(jobs_file).load()[0]

        self.assertEqual(create_exit_code, 0)
        self.assertEqual(reconcile_exit_code, 0)
        self.assertEqual(reconciled["status"], "joined")
        self.assertEqual(reconciled["joined_node_id"], created["bootstrap_bundle"]["node_id"])
        self.assertEqual(persisted.status, "joined")

    def test_clusterctl_provision_wait_for_join_returns_joined_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_file = Path(tmpdir) / "provider-jobs.json"
            state_file = Path(tmpdir) / "cluster-registry.json"
            RegistryStateStore(state_file).save(
                NodeRegistry(
                    [
                        NodeInventory(
                            node_id="vast-node-cli",
                            host="203.0.113.77",
                            lease=LeaseInfo(
                                available_until=parse_datetime("2035-01-01T00:00:00Z")
                            ),
                        )
                    ]
                )
            )

            buffer = io.StringIO()
            with patch.object(ProviderService, "_generate_node_id", return_value="vast-node-cli"):
                with redirect_stdout(buffer):
                    exit_code = clusterctl_main(
                        [
                            "providers-provision",
                            "--provider",
                            "vast",
                            "--blueprint",
                            "qwen-coder-node-vast",
                            "--dry-run",
                            "--jobs-file",
                            str(jobs_file),
                            "--wait-for-join",
                            "--join-state-file",
                            str(state_file),
                            "--join-timeout-seconds",
                            "0",
                            "--join-poll-interval-seconds",
                            "0.1",
                        ]
                    )
            payload = json.loads(buffer.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "joined")
        self.assertEqual(payload["joined_node_id"], "vast-node-cli")
