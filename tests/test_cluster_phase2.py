from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cluster.models import (
    AgentRequest,
    GPUInventory,
    LeaseInfo,
    NodeAccess,
    NodeInventory,
    NodeSystemInfo,
    NodeTopology,
    RuntimeCapability,
    parse_datetime,
)
from cluster.node_agent.heartbeat import HeartbeatPayload, build_heartbeat_payload
from cluster.node_agent.probe_gpu import probe_runtime_capabilities
from cluster.orchestrator.clusterctl import main as clusterctl_main
from cluster.orchestrator.launcher import LaunchResult, build_remote_ssh_command, launch_agent
from cluster.orchestrator.model_profiles import get_runtime_profile, load_runtime_profiles
from cluster.orchestrator.registry import NodeRegistry
from cluster.orchestrator.state_store import RegistryStateStore


def make_gpu(index: int, free_mib: int, total_mib: int = 49152) -> GPUInventory:
    return GPUInventory(
        index=index,
        uuid=f"GPU-{index}",
        total_memory_mib=total_mib,
        free_memory_mib=free_mib,
        utilization_pct=0,
    )


def make_node(
    node_id: str,
    *,
    host: str | None = None,
    available_until: str = "2035-01-01T00:00:00Z",
    gpus: list[GPUInventory] | None = None,
    cached_models: tuple[str, ...] = (),
    access: NodeAccess | None = None,
    runtime_capabilities: tuple[RuntimeCapability, ...] = (),
    system_info: NodeSystemInfo | None = None,
    topology: NodeTopology | None = None,
) -> NodeInventory:
    return NodeInventory(
        node_id=node_id,
        host=host or f"{node_id}.example",
        lease=LeaseInfo(available_until=parse_datetime(available_until)),
        gpus=tuple(gpus or ()),
        cached_models=cached_models,
        access=access,
        runtime_capabilities=runtime_capabilities,
        system_info=system_info,
        topology=topology,
    )


class ClusterPhase2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = parse_datetime("2026-04-04T00:00:00Z")

    def test_heartbeat_updates_refresh_registry(self) -> None:
        registry = NodeRegistry()
        initial = make_node(
            "node-a",
            gpus=[make_gpu(0, 12000)],
            available_until="2026-04-04T00:10:00Z",
        )
        updated = make_node(
            "node-a",
            gpus=[make_gpu(0, 24000)],
            available_until="2026-04-04T01:00:00Z",
        )

        registry.register(initial)
        registry.register_heartbeat(
            updated,
            received_at=parse_datetime("2026-04-04T00:05:00Z"),
            heartbeat_interval_seconds=30,
        )

        record = registry.get_record("node-a")
        assert record is not None
        self.assertEqual(record.node.gpus[0].free_memory_mib, 24000)
        self.assertEqual(record.last_heartbeat_at, parse_datetime("2026-04-04T00:05:00Z"))

    def test_persistent_state_roundtrip(self) -> None:
        registry = NodeRegistry(
            [make_node("node-a", gpus=[make_gpu(0, 16000)]), make_node("node-b", gpus=[])]
        )
        registry.register_heartbeat(
            make_node("node-a", gpus=[make_gpu(0, 18000)]),
            received_at=parse_datetime("2026-04-04T00:03:00Z"),
            heartbeat_interval_seconds=20,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RegistryStateStore(Path(tmpdir) / "registry.json")
            store.save(registry)

            reloaded = store.load()

        self.assertEqual([node.node_id for node in reloaded.list_nodes()], ["node-a", "node-b"])
        record = reloaded.get_record("node-a")
        assert record is not None
        self.assertEqual(record.heartbeat_interval_seconds, 20)
        self.assertEqual(record.node.gpus[0].free_memory_mib, 18000)

    def test_node_inventory_roundtrip_preserves_access_and_capabilities(self) -> None:
        node = make_node(
            "node-a",
            gpus=[make_gpu(0, 16000), make_gpu(1, 15000)],
            access=NodeAccess(
                ssh_user="root",
                ssh_port=11866,
                repo_root="$HOME/Claude-Code-Game-Studios",
            ),
            runtime_capabilities=(
                RuntimeCapability(
                    name="vllm",
                    installed=True,
                    version="0.19.0",
                    executable="/root/Claude-Code-Game-Studios/.venv-vllm/bin/python",
                    supported_topologies=("single-gpu", "tp"),
                ),
            ),
            system_info=NodeSystemInfo(
                hostname="ubuntu",
                python_version="Python 3.10.12",
                driver_version="570.133.20",
                cuda_version="12.8",
            ),
            topology=NodeTopology(
                single_node_multi_gpu=True,
                interconnect="pcie",
            ),
        )

        reloaded = NodeInventory.from_dict(node.to_dict())

        assert reloaded.access is not None
        assert reloaded.system_info is not None
        assert reloaded.topology is not None
        self.assertEqual(reloaded.access.ssh_port, 11866)
        self.assertEqual(reloaded.runtime_capabilities[0].name, "vllm")
        self.assertEqual(reloaded.runtime_capabilities[0].version, "0.19.0")
        self.assertEqual(reloaded.system_info.cuda_version, "12.8")
        self.assertTrue(reloaded.topology.single_node_multi_gpu)

    def test_expired_lease_prevents_launch(self) -> None:
        profile = get_runtime_profile("qwen-coder-30b")
        request = AgentRequest(
            agent_id="blocked-agent",
            model_id=profile.model_name,
            required_vram_mib=profile.required_free_vram_mib,
        )
        decision = make_node(
            "expired-node",
            host="10.0.0.9",
            available_until="2000-01-01T00:00:00Z",
            gpus=[make_gpu(0, 64000)],
        )
        placement = {
            "status": "placed",
            "reason": "test",
            "agent_id": request.agent_id,
            "node_id": decision.node_id,
            "host": decision.host,
            "gpu_index": 0,
            "available_until": decision.available_until,
            "source": "remote",
            "available_vram_mib": 64000,
            "required_vram_mib": request.required_vram_mib,
        }
        from cluster.models import PlacementDecision

        with self.assertRaises(ValueError):
            launch_agent(
                request,
                PlacementDecision(**placement),
                profile,
                dry_run=True,
            )

    def test_ssh_launcher_builds_deterministic_command(self) -> None:
        command_a = build_remote_ssh_command(
            host="10.0.0.58",
            ssh_user="root",
            ssh_port=2222,
            repo_root="$HOME/Claude-Code-Game-Studios",
            gpu_index=1,
            remote_command=["python3", "-m", "cluster.orchestrator.remote_worker", "--agent-id", "a1"],
        )
        command_b = build_remote_ssh_command(
            host="10.0.0.58",
            ssh_user="root",
            ssh_port=2222,
            repo_root="$HOME/Claude-Code-Game-Studios",
            gpu_index=1,
            remote_command=["python3", "-m", "cluster.orchestrator.remote_worker", "--agent-id", "a1"],
        )
        self.assertEqual(command_a, command_b)
        self.assertIn("ssh -p 2222 root@10.0.0.58", command_a)
        self.assertIn("cd $HOME/Claude-Code-Game-Studios", command_a)
        self.assertIn("CUDA_VISIBLE_DEVICES=1", command_a)
        self.assertIn("cluster.orchestrator.remote_worker --agent-id a1", command_a)

    def test_profile_lookup_and_validation(self) -> None:
        profiles = load_runtime_profiles()
        self.assertIn("qwen-coder-30b", profiles)
        self.assertIn("gemma3-12b", profiles)
        self.assertIn("gemma-3-12b-pt", profiles)
        self.assertEqual(profiles["qwen-coder-30b"].preferred_backend, "ollama")

    def test_launch_agent_dry_run_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "registry.json"
            registry = NodeRegistry(
                [
                    make_node("local-node", host="127.0.0.1", gpus=[make_gpu(0, 8000)]),
                    make_node(
                        "remote-node",
                        host="10.0.0.58",
                        gpus=[make_gpu(0, 70000)],
                        cached_models=("qwen3-coder:30b",),
                    ),
                ]
            )
            RegistryStateStore(state_file).save(registry)

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = clusterctl_main(
                    [
                        "launch-agent",
                        "--state-file",
                        str(state_file),
                        "--local-node-id",
                        "local-node",
                        "--agent-id",
                        "dry-run-agent",
                        "--profile",
                        "qwen-coder-30b",
                        "--dry-run",
                    ]
                )
            payload = json.loads(buffer.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["placement"]["node_id"], "remote-node")
        self.assertEqual(payload["launch"]["status"], "ready")
        self.assertEqual(payload["launch"]["mode"], "remote-ssh")

    def test_launch_agent_uses_node_access_defaults_from_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "registry.json"
            registry = NodeRegistry(
                [
                    make_node("local-node", host="127.0.0.1", gpus=[make_gpu(0, 8000)]),
                    make_node(
                        "remote-node",
                        host="209.50.14.20",
                        gpus=[make_gpu(0, 70000)],
                        cached_models=("qwen3-coder:30b",),
                        access=NodeAccess(
                            ssh_user="root",
                            ssh_port=11866,
                            repo_root="$HOME/Claude-Code-Game-Studios",
                        ),
                    ),
                ]
            )
            RegistryStateStore(state_file).save(registry)

            with patch("cluster.orchestrator.clusterctl.launch_agent") as launch_mock:
                launch_mock.return_value = LaunchResult(
                    status="ready",
                    mode="remote-ssh",
                    command="ssh ...",
                    agent_id="dry-run-agent",
                    node_id="remote-node",
                    reason="dry run",
                    executed=False,
                )
                exit_code = clusterctl_main(
                    [
                        "launch-agent",
                        "--state-file",
                        str(state_file),
                        "--local-node-id",
                        "local-node",
                        "--agent-id",
                        "dry-run-agent",
                        "--profile",
                        "qwen-coder-30b",
                        "--dry-run",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(launch_mock.call_args.kwargs["ssh_user"], "root")
        self.assertEqual(launch_mock.call_args.kwargs["ssh_port"], 11866)
        self.assertEqual(
            launch_mock.call_args.kwargs["repo_root"],
            "$HOME/Claude-Code-Game-Studios",
        )

    def test_probe_remote_node_can_register_inventory_into_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "registry.json"
            RegistryStateStore(state_file).save(NodeRegistry())
            remote_payload = {
                "node_id": "cluster-5090x2-live",
                "host": "209.50.14.20",
                "available_until": "2035-01-01T00:00:00Z",
                "gpu_count": 2,
                "gpus": [
                    {
                        "index": 0,
                        "uuid": "GPU-0",
                        "total_memory_mib": 32607,
                        "free_memory_mib": 32000,
                    },
                    {
                        "index": 1,
                        "uuid": "GPU-1",
                        "total_memory_mib": 32607,
                        "free_memory_mib": 32000,
                    },
                ],
                "cached_models": [],
                "access": {
                    "ssh_user": "root",
                    "ssh_port": 11866,
                    "repo_root": "$HOME/Claude-Code-Game-Studios",
                },
                "runtime_capabilities": [
                    {
                        "name": "vllm",
                        "installed": True,
                        "version": "0.19.0",
                        "supported_topologies": ["single-gpu", "tp"],
                    }
                ],
                "topology": {
                    "single_node_multi_gpu": True,
                    "interconnect": "pcie",
                },
            }

            buffer = io.StringIO()
            with patch("cluster.orchestrator.clusterctl.subprocess.run") as run_mock:
                run_mock.return_value.stdout = json.dumps(remote_payload)
                run_mock.return_value.stderr = ""
                with redirect_stdout(buffer):
                    exit_code = clusterctl_main(
                        [
                            "probe-remote-node",
                            "--node-id",
                            "cluster-5090x2-live",
                            "--host",
                            "209.50.14.20",
                            "--ssh-user",
                            "root",
                            "--ssh-port",
                            "11866",
                            "--state-file",
                            str(state_file),
                        ]
                    )
            payload = json.loads(buffer.getvalue())
            reloaded = RegistryStateStore(state_file).load()
            record = reloaded.get_record("cluster-5090x2-live")

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "registered")
        assert record is not None
        assert record.node.access is not None
        self.assertEqual(record.node.access.ssh_port, 11866)
        self.assertEqual(record.node.runtime_capabilities[0].name, "vllm")

    def test_probe_runtime_capabilities_expands_repo_root_env_vars(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            python_path = repo_root / ".venv-vllm" / "bin" / "python"
            python_path.parent.mkdir(parents=True, exist_ok=True)
            os.symlink("/usr/bin/python3", python_path)

            with patch.dict("os.environ", {"TEST_REPO_ROOT": str(repo_root)}):
                with patch(
                    "cluster.node_agent.probe_gpu._probe_distribution_version",
                    return_value="0.19.0",
                ):
                    capabilities = probe_runtime_capabilities(repo_root="$TEST_REPO_ROOT")

        vllm_capability = next(item for item in capabilities if item.name == "vllm")
        self.assertTrue(vllm_capability.installed)
        self.assertEqual(vllm_capability.version, "0.19.0")
        self.assertEqual(vllm_capability.executable, str(python_path))


if __name__ == "__main__":
    unittest.main()
