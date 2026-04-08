from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from cluster.orchestrator.clusterctl import main as clusterctl_main
from cluster.orchestrator.registry import NodeRegistry
from cluster.providers.blueprints import load_builtin_blueprints
from cluster.providers.job_store import ProvisionJobStore
from cluster.providers.models import ProviderOffer, ProvisionRequest
from cluster.providers.service import ProviderService


class ClusterPhase4ProviderTests(unittest.TestCase):
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
