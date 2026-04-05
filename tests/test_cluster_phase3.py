from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cluster.models import AgentRequest, GPUInventory, LeaseInfo, NodeInventory, PlacementDecision, parse_datetime
from cluster.orchestrator.launcher import build_worker_command, launch_agent
from cluster.orchestrator.model_profiles import get_runtime_profile, load_runtime_profiles
from cluster.orchestrator.remote_worker import (
    build_vllm_server_command,
    choose_runtime_port,
    find_conflicting_session,
)


def make_gpu(index: int, free_mib: int, total_mib: int = 97887) -> GPUInventory:
    return GPUInventory(
        index=index,
        uuid=f"GPU-{index}",
        total_memory_mib=total_mib,
        free_memory_mib=free_mib,
        utilization_pct=0,
    )


def make_placement(node_id: str, host: str, gpu_index: int, required_vram_mib: int) -> PlacementDecision:
    return PlacementDecision(
        status="placed",
        reason="test placement",
        agent_id="agent-a",
        node_id=node_id,
        host=host,
        gpu_index=gpu_index,
        available_until=parse_datetime("2035-01-01T00:00:00Z"),
        source="remote",
        available_vram_mib=64000,
        required_vram_mib=required_vram_mib,
    )


class ClusterPhase3Tests(unittest.TestCase):
    def test_runtime_profiles_include_real_backend_launch_variants(self) -> None:
        profiles = load_runtime_profiles()
        self.assertIn("gemma3-4b", profiles)
        self.assertIn("qwen-coder-30b-vllm", profiles)
        self.assertEqual(profiles["gemma3-4b"].preferred_backend, "ollama")
        self.assertEqual(profiles["qwen-coder-30b-vllm"].preferred_backend, "vllm")

    def test_build_worker_command_includes_real_ollama_launch_args(self) -> None:
        profile = get_runtime_profile("qwen-coder-30b")
        request = AgentRequest(
            agent_id="agent-a",
            model_id=profile.model_name,
            required_vram_mib=profile.required_free_vram_mib,
        )
        decision = make_placement("node-a", "10.0.0.58", 1, request.required_vram_mib)

        command = build_worker_command(request, decision, profile)

        self.assertIn("--launch-mode", command)
        self.assertIn("ollama-server", command)
        self.assertIn("--port-base", command)
        self.assertIn("17434", command)
        self.assertIn("--session-dir", command)
        self.assertIn("production/session-state/remote-workers", command)

    def test_vllm_server_command_and_port_are_single_gpu_deterministic(self) -> None:
        command = build_vllm_server_command(
            python_executable="python3",
            launch_module="vllm.entrypoints.openai.api_server",
            host="127.0.0.1",
            port=choose_runtime_port(18000, 1),
            model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=65536,
        )

        self.assertEqual(choose_runtime_port(18000, 1), 18001)
        self.assertEqual(command[0:3], ["python3", "-m", "vllm.entrypoints.openai.api_server"])
        self.assertIn("--tensor-parallel-size", command)
        self.assertIn("1", command)
        self.assertIn("--max-model-len", command)
        self.assertIn("65536", command)

    def test_conflicting_session_detects_same_gpu_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            payload = {
                "status": "launched",
                "agent_id": "other-agent",
                "node_id": "node-a",
                "gpu_index": 0,
                "listen_port": 17434,
            }
            (session_dir / "other-agent.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            conflict = find_conflicting_session(
                session_dir,
                agent_id="agent-a",
                node_id="node-a",
                gpu_index=0,
                listen_port=17434,
            )

        assert conflict is not None
        self.assertEqual(conflict["agent_id"], "other-agent")

    def test_launch_agent_parses_real_worker_payload(self) -> None:
        profile = get_runtime_profile("qwen-coder-30b")
        request = AgentRequest(
            agent_id="agent-a",
            model_id=profile.model_name,
            required_vram_mib=profile.required_free_vram_mib,
        )
        decision = make_placement("node-a", "10.0.0.58", 0, request.required_vram_mib)
        worker_payload = {
            "status": "launched",
            "backend": "ollama",
            "endpoint_url": "http://127.0.0.1:17434",
            "listen_port": 17434,
            "single_gpu_only": True,
        }

        with patch("cluster.orchestrator.launcher.subprocess.run") as run_mock:
            run_mock.return_value.stdout = json.dumps(worker_payload)
            run_mock.return_value.stderr = ""
            result = launch_agent(
                request,
                decision,
                profile,
                dry_run=False,
                ssh_user="root",
                ssh_port=2222,
            )

        self.assertEqual(result.status, "launched")
        self.assertEqual(result.worker_payload, worker_payload)
        assert result.stdout is not None
        self.assertIn("17434", result.stdout)


if __name__ == "__main__":
    unittest.main()
