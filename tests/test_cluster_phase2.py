from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from cluster.models import AgentRequest, GPUInventory, LeaseInfo, NodeInventory, parse_datetime
from cluster.node_agent.heartbeat import HeartbeatPayload, build_heartbeat_payload
from cluster.orchestrator.clusterctl import main as clusterctl_main
from cluster.orchestrator.launcher import build_remote_ssh_command, launch_agent
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
) -> NodeInventory:
    return NodeInventory(
        node_id=node_id,
        host=host or f"{node_id}.example",
        lease=LeaseInfo(available_until=parse_datetime(available_until)),
        gpus=tuple(gpus or ()),
        cached_models=cached_models,
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


if __name__ == "__main__":
    unittest.main()
