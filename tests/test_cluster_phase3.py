from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import io
import argparse
from unittest.mock import patch

from cluster.models import (
    AgentDeploymentSpec,
    AgentRequest,
    GPUInventory,
    LeaseInfo,
    NodeInventory,
    PlacementDecision,
    RuntimeLaunchPreferences,
    parse_datetime,
)
from cluster.orchestrator.clusterctl import _filter_nodes_by_remote_session_claims, main as clusterctl_main
from cluster.orchestrator.launcher import build_worker_command, launch_agent
from cluster.orchestrator.model_profiles import get_runtime_profile, load_runtime_profiles
from cluster.orchestrator.remote_sessions import collect_session_claims
from cluster.orchestrator.remote_worker import (
    build_sglang_server_command,
    build_trtllm_serve_command,
    build_vllm_server_command,
    choose_runtime_port,
    find_conflicting_session,
)
from cluster.orchestrator.runtime_adapters import TrtllmAdapter, infer_packaged_cuda_home
from cluster.orchestrator.runtime_adapters import infer_packaged_library_dirs
from cluster.orchestrator.runtime_adapters import prepend_env_path_entries, prepend_executable_dir_to_path
from cluster.orchestrator.registry import NodeRegistry
from cluster.orchestrator.scheduler import schedule_agent
from cluster.orchestrator.state_store import RegistryStateStore


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
        self.assertIn("qwen-coder-30b-vllm-tp2", profiles)
        self.assertIn("qwen-coder-30b-vllm-tp4", profiles)
        self.assertIn("qwen-coder-30b-sglang-tp2", profiles)
        self.assertIn("qwen-coder-30b-sglang-tp4", profiles)
        self.assertIn("qwen-coder-30b-trtllm-tp2", profiles)
        self.assertIn("qwen-coder-30b-trtllm-tp4", profiles)
        self.assertEqual(profiles["gemma3-4b"].preferred_backend, "ollama")
        self.assertEqual(profiles["gemma3-4b"].runtime_adapter, "ollama-server")
        self.assertEqual(profiles["qwen-coder-30b-vllm"].preferred_backend, "vllm")
        self.assertEqual(profiles["qwen-coder-30b-vllm"].runtime_adapter, "vllm-server")
        self.assertEqual(profiles["qwen-coder-30b-vllm-tp2"].required_gpu_count, 2)
        self.assertTrue(profiles["qwen-coder-30b-vllm-tp2"].runtime_options["enforce_eager"])
        self.assertEqual(profiles["qwen-coder-30b-vllm-tp4"].required_gpu_count, 4)
        self.assertEqual(profiles["qwen-coder-30b-vllm-tp4"].runtime_options["tensor_parallel_size"], 4)
        self.assertEqual(profiles["qwen-coder-30b-sglang-tp2"].runtime_adapter, "sglang-server")
        self.assertEqual(profiles["qwen-coder-30b-sglang-tp2"].runtime_options["tensor_parallel_size"], 2)
        self.assertTrue(profiles["qwen-coder-30b-sglang-tp4"].runtime_options["trust_remote_code"])
        self.assertEqual(profiles["qwen-coder-30b-trtllm-tp2"].runtime_adapter, "trtllm-server")
        self.assertEqual(profiles["qwen-coder-30b-trtllm-tp2"].runtime_options["tensor_parallel_size"], 2)
        self.assertEqual(profiles["qwen-coder-30b-trtllm-tp4"].runtime_options["pipeline_parallel_size"], 1)

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
        self.assertIn("--gpu-indices", command)
        self.assertIn("1", command)

    def test_build_worker_command_includes_gpu_group_for_tp2_profile(self) -> None:
        profile = get_runtime_profile("qwen-coder-30b-vllm-tp2")
        request = AgentRequest(
            agent_id="agent-tp2",
            model_id=profile.model_name,
            required_vram_mib=profile.required_free_vram_mib,
            required_gpu_count=profile.required_gpu_count,
        )
        decision = PlacementDecision(
            status="placed",
            reason="test placement",
            agent_id=request.agent_id,
            node_id="node-b",
            host="10.0.0.59",
            gpu_index=0,
            gpu_indices=(0, 1),
            available_until=parse_datetime("2035-01-01T00:00:00Z"),
            source="remote",
            available_vram_mib=32000,
            required_vram_mib=request.required_vram_mib,
        )

        command = build_worker_command(request, decision, profile)

        self.assertIn("--gpu-indices", command)
        self.assertIn("0,1", command)
        self.assertIn("--tensor-parallel-size", command)
        self.assertIn("2", command)
        self.assertIn("--enforce-eager", command)

    def test_build_worker_command_includes_gpu_group_for_tp4_profile(self) -> None:
        profile = get_runtime_profile("qwen-coder-30b-vllm-tp4")
        request = AgentRequest(
            agent_id="agent-tp4",
            model_id=profile.model_name,
            required_vram_mib=profile.required_free_vram_mib,
            required_gpu_count=profile.required_gpu_count,
        )
        decision = PlacementDecision(
            status="placed",
            reason="test placement",
            agent_id=request.agent_id,
            node_id="node-c",
            host="10.0.0.60",
            gpu_index=0,
            gpu_indices=(0, 1, 2, 3),
            available_until=parse_datetime("2035-01-01T00:00:00Z"),
            source="remote",
            available_vram_mib=32110,
            required_vram_mib=request.required_vram_mib,
        )

        command = build_worker_command(request, decision, profile)

        self.assertIn("--gpu-indices", command)
        self.assertIn("0,1,2,3", command)
        self.assertIn("--tensor-parallel-size", command)
        self.assertIn("4", command)
        self.assertIn("--max-model-len", command)
        self.assertIn("32768", command)

    def test_build_worker_command_includes_sglang_launch_args_for_tp2_profile(self) -> None:
        profile = get_runtime_profile("qwen-coder-30b-sglang-tp2")
        request = AgentRequest(
            agent_id="agent-sglang-tp2",
            model_id=profile.model_name,
            required_vram_mib=profile.required_free_vram_mib,
            required_gpu_count=profile.required_gpu_count,
        )
        decision = PlacementDecision(
            status="placed",
            reason="test placement",
            agent_id=request.agent_id,
            node_id="node-sg2",
            host="10.0.0.61",
            gpu_index=0,
            gpu_indices=(0, 1),
            available_until=parse_datetime("2035-01-01T00:00:00Z"),
            source="remote",
            available_vram_mib=32100,
            required_vram_mib=request.required_vram_mib,
        )

        command = build_worker_command(request, decision, profile)

        self.assertIn("--launch-mode", command)
        self.assertIn("sglang-server", command)
        self.assertIn("--gpu-indices", command)
        self.assertIn("0,1", command)
        self.assertIn("--tensor-parallel-size", command)
        self.assertIn("2", command)
        self.assertIn("--mem-fraction-static", command)
        self.assertIn("0.96", command)
        self.assertIn("--context-length", command)
        self.assertIn("16384", command)
        self.assertIn("--trust-remote-code", command)
        self.assertIn("--disable-custom-all-reduce", command)
        self.assertIn("--disable-cuda-graph", command)

    def test_build_worker_command_includes_sglang_launch_args_for_tp4_profile(self) -> None:
        profile = get_runtime_profile("qwen-coder-30b-sglang-tp4")
        request = AgentRequest(
            agent_id="agent-sglang-tp4",
            model_id=profile.model_name,
            required_vram_mib=profile.required_free_vram_mib,
            required_gpu_count=profile.required_gpu_count,
        )
        decision = PlacementDecision(
            status="placed",
            reason="test placement",
            agent_id=request.agent_id,
            node_id="node-sg4",
            host="10.0.0.62",
            gpu_index=0,
            gpu_indices=(0, 1, 2, 3),
            available_until=parse_datetime("2035-01-01T00:00:00Z"),
            source="remote",
            available_vram_mib=32100,
            required_vram_mib=request.required_vram_mib,
        )

        command = build_worker_command(request, decision, profile)

        self.assertIn("--gpu-indices", command)
        self.assertIn("0,1,2,3", command)
        self.assertIn("--sglang-launch-module", command)
        self.assertIn("sglang.launch_server", command)
        self.assertIn("--mem-fraction-static", command)
        self.assertIn("0.88", command)
        self.assertIn("--context-length", command)
        self.assertIn("16384", command)
        self.assertIn("--tensor-parallel-size", command)
        self.assertIn("4", command)
        self.assertIn("--disable-custom-all-reduce", command)
        self.assertIn("--disable-cuda-graph", command)

    def test_build_worker_command_includes_trtllm_launch_args_for_tp2_profile(self) -> None:
        profile = get_runtime_profile("qwen-coder-30b-trtllm-tp2")
        request = AgentRequest(
            agent_id="agent-trtllm-tp2",
            model_id=profile.model_name,
            required_vram_mib=profile.required_free_vram_mib,
            required_gpu_count=profile.required_gpu_count,
        )
        decision = PlacementDecision(
            status="placed",
            reason="test placement",
            agent_id=request.agent_id,
            node_id="node-trt2",
            host="10.0.0.63",
            gpu_index=0,
            gpu_indices=(0, 1),
            available_until=parse_datetime("2035-01-01T00:00:00Z"),
            source="remote",
            available_vram_mib=32050,
            required_vram_mib=request.required_vram_mib,
        )

        command = build_worker_command(request, decision, profile)

        self.assertIn("--launch-mode", command)
        self.assertIn("trtllm-server", command)
        self.assertIn("--trtllm-executable", command)
        self.assertTrue(
            any(item.endswith(".venv-trtllm/bin/trtllm-serve") for item in command),
            msg=str(command),
        )
        self.assertIn("--trtllm-backend", command)
        self.assertIn("pytorch", command)
        self.assertIn("--gpu-indices", command)
        self.assertIn("0,1", command)
        self.assertIn("--tensor-parallel-size", command)
        self.assertIn("2", command)
        self.assertIn("--pipeline-parallel-size", command)
        self.assertIn("1", command)
        self.assertIn("--trtllm-max-seq-len", command)
        self.assertIn("16384", command)

    def test_build_worker_command_includes_trtllm_launch_args_for_tp4_profile(self) -> None:
        profile = get_runtime_profile("qwen-coder-30b-trtllm-tp4")
        request = AgentRequest(
            agent_id="agent-trtllm-tp4",
            model_id=profile.model_name,
            required_vram_mib=profile.required_free_vram_mib,
            required_gpu_count=profile.required_gpu_count,
        )
        decision = PlacementDecision(
            status="placed",
            reason="test placement",
            agent_id=request.agent_id,
            node_id="node-trt4",
            host="10.0.0.64",
            gpu_index=0,
            gpu_indices=(0, 1, 2, 3),
            available_until=parse_datetime("2035-01-01T00:00:00Z"),
            source="remote",
            available_vram_mib=24000,
            required_vram_mib=request.required_vram_mib,
        )

        command = build_worker_command(request, decision, profile)

        self.assertIn("--gpu-indices", command)
        self.assertIn("0,1,2,3", command)
        self.assertIn("--tensor-parallel-size", command)
        self.assertIn("4", command)
        self.assertIn("--trtllm-max-num-tokens", command)
        self.assertIn("16384", command)

    def test_build_worker_command_respects_sglang_cuda_graph_overrides(self) -> None:
        profile = get_runtime_profile("qwen-coder-30b-sglang-tp2").with_launch_overrides(
            {
                "disable_cuda_graph": False,
                "cuda_graph_max_bs": 32,
            }
        )
        request = AgentRequest(
            agent_id="agent-sglang-cudagraph",
            model_id=profile.model_name,
            required_vram_mib=profile.required_free_vram_mib,
            required_gpu_count=profile.required_gpu_count,
        )
        decision = PlacementDecision(
            status="placed",
            reason="test placement",
            agent_id=request.agent_id,
            node_id="node-sg2",
            host="10.0.0.61",
            gpu_index=0,
            gpu_indices=(0, 1),
            available_until=parse_datetime("2035-01-01T00:00:00Z"),
            source="remote",
            available_vram_mib=32100,
            required_vram_mib=request.required_vram_mib,
        )

        command = build_worker_command(request, decision, profile)

        self.assertNotIn("--disable-cuda-graph", command)
        self.assertIn("--cuda-graph-max-bs", command)
        self.assertIn("32", command)

    def test_agent_deployment_spec_roundtrip_preserves_launch_preferences(self) -> None:
        spec = AgentDeploymentSpec(
            agent_id="agent-ui-1",
            profile="qwen-coder-30b-sglang-tp2",
            required_vram_mib=30000,
            required_gpu_count=2,
            model_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            labels=(("role", "coder"),),
            trust_tier="burst",
            network_tier="public",
            launch_preferences=RuntimeLaunchPreferences(
                cuda_graph_mode="enabled",
                cuda_graph_max_bs=32,
            ),
        )

        restored = AgentDeploymentSpec.from_dict(spec.to_dict())

        self.assertEqual(restored.to_dict(), spec.to_dict())
        self.assertEqual(restored.to_agent_request().required_gpu_count, 2)

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
            enforce_eager=True,
        )

        self.assertEqual(choose_runtime_port(18000, 1), 18001)
        self.assertEqual(command[0:3], ["python3", "-m", "vllm.entrypoints.openai.api_server"])
        self.assertIn("--tensor-parallel-size", command)
        self.assertIn("1", command)
        self.assertIn("--max-model-len", command)
        self.assertIn("65536", command)
        self.assertIn("--enforce-eager", command)

    def test_sglang_server_command_and_port_are_multi_gpu_deterministic(self) -> None:
        command = build_sglang_server_command(
            python_executable="python3",
            launch_module="sglang.launch_server",
            host="127.0.0.1",
            port=choose_runtime_port(19140, 0),
            model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            tensor_parallel_size=4,
            mem_fraction_static=0.9,
            context_length=32768,
            trust_remote_code=True,
            disable_custom_all_reduce=True,
            disable_cuda_graph=True,
            cuda_graph_max_bs=24,
        )

        self.assertEqual(choose_runtime_port(19140, 0), 19140)
        self.assertEqual(command[0:3], ["python3", "-m", "sglang.launch_server"])
        self.assertIn("--model-path", command)
        self.assertIn("--tp", command)
        self.assertIn("4", command)
        self.assertIn("--mem-fraction-static", command)
        self.assertIn("0.9", command)
        self.assertIn("--context-length", command)
        self.assertIn("32768", command)
        self.assertIn("--trust-remote-code", command)
        self.assertIn("--disable-custom-all-reduce", command)
        self.assertIn("--disable-cuda-graph", command)
        self.assertIn("--cuda-graph-max-bs", command)
        self.assertIn("24", command)

    def test_trtllm_serve_command_and_port_are_multi_gpu_deterministic(self) -> None:
        command = build_trtllm_serve_command(
            executable="trtllm-serve",
            model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            host="127.0.0.1",
            port=choose_runtime_port(20000, 4),
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            backend="pytorch",
            max_batch_size=16,
            max_num_tokens=16384,
            max_seq_len=32768,
            log_level="info",
        )

        self.assertEqual(choose_runtime_port(20000, 4), 20004)
        self.assertEqual(command[0:3], ["trtllm-serve", "serve", "Qwen/Qwen3-Coder-30B-A3B-Instruct"])
        self.assertIn("--backend", command)
        self.assertIn("pytorch", command)
        self.assertIn("--tp_size", command)
        self.assertIn("4", command)
        self.assertIn("--max_seq_len", command)
        self.assertIn("32768", command)

    def test_infer_packaged_cuda_home_prefers_packaged_runtime_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_root = Path(tmpdir) / ".venv-sglang"
            python_path = venv_root / "bin" / "python"
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("", encoding="utf-8")
            cuda_runtime = (
                venv_root
                / "lib"
                / "python3.10"
                / "site-packages"
                / "nvidia"
                / "cuda_runtime"
                / "include"
            )
            cuda_runtime.mkdir(parents=True, exist_ok=True)
            (cuda_runtime / "cuda_runtime.h").write_text("", encoding="utf-8")

            detected = infer_packaged_cuda_home(str(python_path))

        self.assertEqual(
            detected,
            str(
                venv_root
                / "lib"
                / "python3.10"
                / "site-packages"
                / "nvidia"
                / "cuda_runtime"
            ),
        )

    def test_infer_packaged_cuda_home_accepts_trtllm_executable_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_root = Path(tmpdir) / ".venv-trtllm"
            executable_path = venv_root / "bin" / "trtllm-serve"
            executable_path.parent.mkdir(parents=True, exist_ok=True)
            executable_path.write_text("", encoding="utf-8")
            cuda_runtime = (
                venv_root
                / "lib"
                / "python3.10"
                / "site-packages"
                / "nvidia"
                / "cuda_runtime"
                / "include"
            )
            cuda_runtime.mkdir(parents=True, exist_ok=True)
            (cuda_runtime / "cuda_runtime.h").write_text("", encoding="utf-8")

            detected = infer_packaged_cuda_home(str(executable_path))

        self.assertEqual(
            detected,
            str(
                venv_root
                / "lib"
                / "python3.10"
                / "site-packages"
                / "nvidia"
                / "cuda_runtime"
            ),
        )

    def test_prepend_executable_dir_to_path_puts_venv_bin_first(self) -> None:
        env = {"PATH": "/usr/local/bin:/usr/bin"}

        prepend_executable_dir_to_path(
            env,
            "/tmp/.venv-sglang/bin/python",
        )

        self.assertEqual(
            env["PATH"].split(":")[0],
            "/tmp/.venv-sglang/bin",
        )

    def test_infer_packaged_library_dirs_discovers_trtllm_runtime_libs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_root = Path(tmpdir) / ".venv-trtllm"
            executable_path = venv_root / "bin" / "trtllm-serve"
            executable_path.parent.mkdir(parents=True, exist_ok=True)
            executable_path.write_text("", encoding="utf-8")

            site_packages = venv_root / "lib" / "python3.10" / "site-packages"
            (site_packages / "nvidia" / "cu13" / "lib").mkdir(parents=True, exist_ok=True)
            (site_packages / "nvidia" / "cuda_runtime" / "lib").mkdir(parents=True, exist_ok=True)
            (site_packages / "torch" / "lib").mkdir(parents=True, exist_ok=True)
            (site_packages / "tensorrt_libs").mkdir(parents=True, exist_ok=True)

            detected = infer_packaged_library_dirs(str(executable_path))

        self.assertIn(str(site_packages / "nvidia" / "cu13" / "lib"), detected)
        self.assertIn(str(site_packages / "torch" / "lib"), detected)
        self.assertIn(str(site_packages / "tensorrt_libs"), detected)

    def test_prepend_env_path_entries_puts_runtime_libs_first(self) -> None:
        env = {"LD_LIBRARY_PATH": "/usr/local/lib:/usr/lib"}

        prepend_env_path_entries(
            env,
            "LD_LIBRARY_PATH",
            ["/tmp/.venv-trtllm/lib-a", "/tmp/.venv-trtllm/lib-b", "/usr/lib"],
        )

        self.assertEqual(
            env["LD_LIBRARY_PATH"].split(":")[0:3],
            ["/tmp/.venv-trtllm/lib-a", "/tmp/.venv-trtllm/lib-b", "/usr/lib"],
        )

    def test_launch_agent_dry_run_emits_deployment_spec(self) -> None:
        local_node = NodeInventory(
            node_id="local-node",
            host="127.0.0.1",
            lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
            gpus=(make_gpu(0, 16000, total_mib=16384),),
        )
        remote_node = NodeInventory(
            node_id="remote-sg",
            host="10.0.0.62",
            lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
            gpus=(make_gpu(0, 32100, total_mib=32607), make_gpu(1, 32100, total_mib=32607)),
            cached_models=("Qwen/Qwen3-Coder-30B-A3B-Instruct",),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "local.json"
            remote_path = Path(tmpdir) / "remote.json"
            local_path.write_text(json.dumps([local_node.to_dict()]), encoding="utf-8")
            remote_path.write_text(json.dumps([remote_node.to_dict()]), encoding="utf-8")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = clusterctl_main(
                    [
                        "launch-agent",
                        "--local-file",
                        str(local_path),
                        "--remote-file",
                        str(remote_path),
                        "--local-node-id",
                        "local-node",
                        "--agent-id",
                        "agent-ui-launch",
                        "--profile",
                        "qwen-coder-30b-sglang-tp2",
                        "--cuda-graph-mode",
                        "enabled",
                        "--cuda-graph-max-bs",
                        "32",
                        "--dry-run",
                    ]
                )
        payload = json.loads(buffer.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["deployment"]["profile"], "qwen-coder-30b-sglang-tp2")
        self.assertEqual(
            payload["deployment"]["launch_preferences"]["cuda_graph_mode"],
            "enabled",
        )
        self.assertEqual(
            payload["deployment"]["launch_preferences"]["cuda_graph_max_bs"],
            32,
        )

    def test_launch_agent_uses_local_file_as_local_anchor_without_local_node_id(self) -> None:
        local_node = NodeInventory(
            node_id="local-dummy",
            host="127.0.0.1",
            lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
            gpus=(make_gpu(0, 4096, total_mib=8192),),
        )
        remote_node = NodeInventory(
            node_id="remote-trt",
            host="10.0.0.70",
            lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
            gpus=(make_gpu(0, 32100, total_mib=32607), make_gpu(1, 32100, total_mib=32607)),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "local.json"
            remote_path = Path(tmpdir) / "remote.json"
            local_path.write_text(json.dumps([local_node.to_dict()]), encoding="utf-8")
            remote_path.write_text(json.dumps([remote_node.to_dict()]), encoding="utf-8")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = clusterctl_main(
                    [
                        "launch-agent",
                        "--local-file",
                        str(local_path),
                        "--remote-file",
                        str(remote_path),
                        "--agent-id",
                        "agent-trt-anchor",
                        "--profile",
                        "qwen-coder-30b-trtllm-tp2",
                        "--dry-run",
                    ]
                )
        payload = json.loads(buffer.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["placement"]["node_id"], "remote-trt")
        self.assertEqual(payload["placement"]["source"], "remote")
        self.assertEqual(payload["launch"]["mode"], "remote-ssh")

    def test_trtllm_launch_writes_starting_session_with_server_pid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "sessions"
            bin_dir = Path(tmpdir) / "bin"
            bin_dir.mkdir()
            executable = bin_dir / "trtllm-serve"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

            args = argparse.Namespace(
                agent_id="trt-agent",
                node_id="node-a",
                backend="tensorrt-llm",
                runtime_class="remote-runtime",
                model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
                gpu_index=0,
                gpu_indices="0,1",
                tensor_parallel_size=2,
                pipeline_parallel_size=1,
                trtllm_executable=str(executable),
                trtllm_backend="pytorch",
                trtllm_tokenizer=None,
                trtllm_max_batch_size=None,
                trtllm_max_num_tokens=None,
                trtllm_max_seq_len=None,
                trtllm_log_level="info",
                port_base=20000,
                server_host="127.0.0.1",
                startup_timeout_seconds=30,
                request_timeout_seconds=30,
            )

            writes: list[dict[str, object]] = []

            def capture_write(_path: Path, payload: dict[str, object]) -> None:
                writes.append(dict(payload))

            with (
                patch(
                    "cluster.orchestrator.runtime_adapters.start_background_process",
                    return_value=4321,
                ),
                patch(
                    "cluster.orchestrator.runtime_adapters.wait_for_json_endpoint_or_process_exit",
                    return_value=("http://127.0.0.1:20000/health", {}),
                ),
                patch(
                    "cluster.orchestrator.runtime_adapters.write_session_payload",
                    side_effect=capture_write,
                ),
            ):
                session = TrtllmAdapter().launch(args, session_dir)

        self.assertEqual(session.server_pid, 4321)
        self.assertGreaterEqual(len(writes), 2)
        self.assertEqual(writes[0]["status"], "starting")
        self.assertEqual(writes[0]["server_pid"], 4321)
        self.assertEqual(writes[-1]["status"], "launched")
        self.assertEqual(writes[-1]["server_pid"], 4321)

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

    def test_conflicting_session_detects_overlapping_multi_gpu_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            payload = {
                "status": "launched",
                "agent_id": "other-agent",
                "node_id": "node-a",
                "gpu_index": 0,
                "gpu_indices": [0, 1],
                "listen_port": 18020,
            }
            (session_dir / "other-agent.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            conflict = find_conflicting_session(
                session_dir,
                agent_id="agent-a",
                node_id="node-a",
                gpu_index=1,
                gpu_indices=(1, 2),
                listen_port=18020,
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

    def test_collect_session_claims_only_returns_active_gpu_owners(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            (session_dir / "active.json").write_text(
                json.dumps(
                    {
                        "status": "launched",
                        "agent_id": "active-agent",
                        "node_id": "node-a",
                        "gpu_index": 0,
                        "backend": "ollama",
                    }
                ),
                encoding="utf-8",
            )
            (session_dir / "done.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "agent_id": "done-agent",
                        "node_id": "node-a",
                        "gpu_index": 1,
                    }
                ),
                encoding="utf-8",
            )

            claims = collect_session_claims(session_dir)

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].agent_id, "active-agent")
        self.assertEqual(claims[0].gpu_index, 0)

    def test_collect_session_claims_expands_multi_gpu_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            (session_dir / "tp2.json").write_text(
                json.dumps(
                    {
                        "status": "launched",
                        "agent_id": "tp2-agent",
                        "node_id": "node-a",
                        "gpu_index": 0,
                        "gpu_indices": [0, 1],
                    }
                ),
                encoding="utf-8",
            )

            claims = collect_session_claims(session_dir)

        self.assertEqual([claim.gpu_index for claim in claims], [0, 1])

    def test_collect_session_claims_ignores_dead_server_pid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            (session_dir / "dead.json").write_text(
                json.dumps(
                    {
                        "status": "starting",
                        "agent_id": "dead-agent",
                        "node_id": "node-a",
                        "gpu_index": 0,
                        "server_pid": 4242,
                    }
                ),
                encoding="utf-8",
            )

            with patch("cluster.orchestrator.remote_sessions.os.kill", side_effect=ProcessLookupError):
                claims = collect_session_claims(session_dir)

        self.assertEqual(claims, [])

    def test_filter_nodes_by_remote_session_claims_removes_conflicting_gpus(self) -> None:
        nodes = [
            NodeInventory(
                node_id="node-a",
                host="node-a.example",
                lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
                gpus=(make_gpu(0, 64000), make_gpu(1, 64000)),
            )
        ]
        remote_payload = {
            "sessions": [],
            "claims": [
                {
                    "agent_id": "other-agent",
                    "node_id": "node-a",
                    "gpu_index": 0,
                    "status": "launched",
                },
                {
                    "agent_id": "same-agent",
                    "node_id": "node-a",
                    "gpu_index": 1,
                    "status": "reused",
                },
            ],
        }

        with patch("cluster.orchestrator.clusterctl.subprocess.run") as run_mock:
            run_mock.return_value.stdout = json.dumps(remote_payload)
            filtered_nodes, claims = _filter_nodes_by_remote_session_claims(
                nodes,
                request_agent_id="same-agent",
                ssh_user=None,
                ssh_port=None,
                repo_root="$HOME/Claude-Code-Game-Studios",
                session_dir="production/session-state/remote-workers",
            )

        self.assertEqual(len(claims), 2)
        self.assertEqual([gpu.index for gpu in filtered_nodes[0].gpus], [1])

    def test_launch_agent_can_block_before_remote_exec_when_session_probe_finds_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "registry.json"
            registry = NodeRegistry(
                [
                    NodeInventory(
                        node_id="local-node",
                        host="127.0.0.1",
                        lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
                        gpus=(make_gpu(0, 8000, total_mib=16384),),
                    ),
                    NodeInventory(
                        node_id="remote-node",
                        host="remote-node.example",
                        lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
                        gpus=(make_gpu(0, 70000),),
                        cached_models=("qwen3-coder:30b",),
                    ),
                ]
            )
            RegistryStateStore(state_file).save(registry)

            remote_payload = {
                "sessions": [],
                "claims": [
                    {
                        "agent_id": "busy-agent",
                        "node_id": "remote-node",
                        "gpu_index": 0,
                        "status": "launched",
                    }
                ],
            }

            buffer = io.StringIO()
            with patch("cluster.orchestrator.clusterctl.subprocess.run") as run_mock:
                run_mock.return_value.stdout = json.dumps(remote_payload)
                with redirect_stdout(buffer):
                    exit_code = clusterctl_main(
                        [
                            "launch-agent",
                            "--state-file",
                            str(state_file),
                            "--local-node-id",
                            "local-node",
                            "--agent-id",
                            "new-agent",
                            "--profile",
                            "qwen-coder-30b",
                            "--dry-run",
                            "--probe-remote-sessions",
                        ]
                    )
            payload = json.loads(buffer.getvalue())

        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["launch"]["status"], "blocked")
        self.assertIn("busy-agent", payload["launch"]["reason"])

    def test_schedule_agent_can_place_tp2_request_on_same_node_gpu_group(self) -> None:
        profile = get_runtime_profile("qwen-coder-30b-vllm-tp2")
        request = AgentRequest(
            agent_id="agent-tp2",
            model_id=profile.model_name,
            required_vram_mib=profile.required_free_vram_mib,
            required_gpu_count=profile.required_gpu_count,
        )
        local_node = NodeInventory(
            node_id="local-node",
            host="127.0.0.1",
            lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
            gpus=(make_gpu(0, 12000, total_mib=16384),),
        )
        remote_node = NodeInventory(
            node_id="remote-tp2",
            host="remote-tp2.example",
            lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
            gpus=(make_gpu(0, 32000, total_mib=32607), make_gpu(1, 32100, total_mib=32607)),
        )

        decision = schedule_agent(local_node, [remote_node], request)

        self.assertTrue(decision.is_placed)
        self.assertEqual(decision.node_id, "remote-tp2")
        self.assertEqual(decision.gpu_indices, (0, 1))
        self.assertEqual(decision.gpu_index, 0)

    def test_schedule_agent_can_place_tp4_request_on_same_node_gpu_group(self) -> None:
        profile = get_runtime_profile("qwen-coder-30b-vllm-tp4")
        request = AgentRequest(
            agent_id="agent-tp4",
            model_id=profile.model_name,
            required_vram_mib=profile.required_free_vram_mib,
            required_gpu_count=profile.required_gpu_count,
        )
        local_node = NodeInventory(
            node_id="local-node",
            host="127.0.0.1",
            lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
            gpus=(make_gpu(0, 12000, total_mib=16384),),
        )
        remote_node = NodeInventory(
            node_id="remote-tp4",
            host="remote-tp4.example",
            lease=LeaseInfo(available_until=parse_datetime("2035-01-01T00:00:00Z")),
            gpus=(
                make_gpu(0, 32110, total_mib=32607),
                make_gpu(1, 32110, total_mib=32607),
                make_gpu(2, 32110, total_mib=32607),
                make_gpu(3, 32110, total_mib=32607),
            ),
        )

        decision = schedule_agent(local_node, [remote_node], request)

        self.assertTrue(decision.is_placed)
        self.assertEqual(decision.node_id, "remote-tp4")
        self.assertEqual(decision.gpu_indices, (0, 1, 2, 3))
        self.assertEqual(decision.gpu_index, 0)


if __name__ == "__main__":
    unittest.main()
