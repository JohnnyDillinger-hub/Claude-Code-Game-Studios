from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pwd
import re
import signal
import shutil
import subprocess
import time
from typing import Any, Iterable, Protocol
from urllib import error, request


DEFAULT_SESSION_DIR = "production/session-state/remote-workers"
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_WARMUP_PROMPT = "Reply with exactly READY"
DEFAULT_STARTUP_TIMEOUT_SECONDS = 180
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180
DEFAULT_MAX_NEW_TOKENS = 8
DEFAULT_OLLAMA_PORT_BASE = 17434
DEFAULT_VLLM_PORT_BASE = 18000
DEFAULT_VLLM_LAUNCH_MODULE = "vllm.entrypoints.openai.api_server"
DEFAULT_SGLANG_PORT_BASE = 19000
DEFAULT_SGLANG_LAUNCH_MODULE = "sglang.launch_server"
DEFAULT_TRTLLM_PORT_BASE = 20000
DEFAULT_TRTLLM_EXECUTABLE = "trtllm-serve"
DEFAULT_TRTLLM_BACKEND = "pytorch"
DEFAULT_DEEPSPEED_PORT_BASE = 21000
DEFAULT_DEEPSPEED_EXECUTABLE = "deepspeed"
DEFAULT_DEEPSPEED_DTYPE = "fp16"
DEFAULT_MEM_FRACTION_STATIC = 0.9
DEFAULT_STALE_STARTING_SECONDS = 900
DEFAULT_PROCESS_STOP_GRACE_SECONDS = 2.0
SESSION_OK_STATUSES = {"starting", "launched", "reused"}


class WorkerError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class WorkerSession:
    status: str
    agent_id: str
    node_id: str
    backend: str
    runtime_class: str
    model: str
    gpu_index: int
    single_gpu_only: bool
    launched_at: str
    session_file: str
    gpu_indices: tuple[int, ...] = ()
    tensor_parallel_size: int = 1
    endpoint_url: str | None = None
    listen_port: int | None = None
    server_pid: int | None = None
    process_group_id: int | None = None
    stdout_log: str | None = None
    stderr_log: str | None = None
    warmup_response: str | None = None
    reused: bool = False
    command: list[str] | None = None
    health_url: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "agent_id": self.agent_id,
            "node_id": self.node_id,
            "backend": self.backend,
            "runtime_class": self.runtime_class,
            "model": self.model,
            "gpu_index": self.gpu_index,
            "tensor_parallel_size": self.tensor_parallel_size,
            "single_gpu_only": self.single_gpu_only,
            "launched_at": self.launched_at,
            "session_file": self.session_file,
            "reused": self.reused,
        }
        if self.gpu_indices:
            payload["gpu_indices"] = list(self.gpu_indices)
        if self.endpoint_url is not None:
            payload["endpoint_url"] = self.endpoint_url
        if self.listen_port is not None:
            payload["listen_port"] = self.listen_port
        if self.server_pid is not None:
            payload["server_pid"] = self.server_pid
        if self.process_group_id is not None:
            payload["process_group_id"] = self.process_group_id
        if self.stdout_log is not None:
            payload["stdout_log"] = self.stdout_log
        if self.stderr_log is not None:
            payload["stderr_log"] = self.stderr_log
        if self.warmup_response is not None:
            payload["warmup_response"] = self.warmup_response
        if self.command is not None:
            payload["command"] = self.command
        if self.health_url is not None:
            payload["health_url"] = self.health_url
        if self.notes is not None:
            payload["notes"] = self.notes
        return payload


def utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_user_home_dir() -> str:
    try:
        passwd_home = pwd.getpwuid(os.getuid()).pw_dir
    except (KeyError, OSError):
        passwd_home = ""
    env_home = os.environ.get("HOME", "")
    return passwd_home or env_home or "~"


def expand_path_text(value: str) -> str:
    home_dir = resolve_user_home_dir()
    expanded = value.replace("${HOME}", home_dir).replace("$HOME", home_dir)
    if expanded == "~":
        return home_dir
    if expanded.startswith("~/"):
        return str(Path(home_dir) / expanded[2:])
    return os.path.expanduser(os.path.expandvars(expanded))


def choose_runtime_port(port_base: int, gpu_index: int) -> int:
    return int(port_base) + int(gpu_index)


def choose_runtime_port_for_group(port_base: int, gpu_indices: tuple[int, ...]) -> int:
    if not gpu_indices:
        raise WorkerError("At least one GPU index is required to choose a runtime port")
    return choose_runtime_port(port_base, gpu_indices[0])


def sanitize_agent_id(agent_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", agent_id)


def resolve_gpu_indices(args: argparse.Namespace) -> tuple[int, ...]:
    if getattr(args, "gpu_indices", None):
        items = [item.strip() for item in str(args.gpu_indices).split(",") if item.strip()]
        if not items:
            raise WorkerError("gpu_indices cannot be empty")
        try:
            gpu_indices = tuple(int(item) for item in items)
        except ValueError as exc:
            raise WorkerError(f"gpu_indices must be integers: {args.gpu_indices!r}") from exc
    else:
        gpu_indices = (int(args.gpu_index),)
    if len(set(gpu_indices)) != len(gpu_indices):
        raise WorkerError(f"gpu_indices contains duplicates: {gpu_indices}")
    if int(args.gpu_index) != gpu_indices[0]:
        raise WorkerError("gpu_index must match the first element of gpu_indices")
    return gpu_indices


def build_ollama_server_command() -> list[str]:
    return ["ollama", "serve"]


def build_vllm_server_command(
    *,
    python_executable: str,
    launch_module: str,
    host: str,
    port: int,
    model: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int | None = None,
    enforce_eager: bool = False,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        launch_module,
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if max_model_len is not None:
        command.extend(["--max-model-len", str(max_model_len)])
    if enforce_eager:
        command.append("--enforce-eager")
    return command


def build_sglang_server_command(
    *,
    python_executable: str,
    launch_module: str,
    host: str,
    port: int,
    model: str,
    tensor_parallel_size: int,
    mem_fraction_static: float,
    context_length: int | None = None,
    trust_remote_code: bool = False,
    enable_p2p_check: bool = False,
    disable_custom_all_reduce: bool = False,
    disable_overlap_schedule: bool = False,
    disable_cuda_graph: bool = False,
    disable_piecewise_cuda_graph: bool = True,
    cuda_graph_max_bs: int | None = None,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        launch_module,
        "--model-path",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--tp",
        str(tensor_parallel_size),
        "--mem-fraction-static",
        str(mem_fraction_static),
    ]
    if context_length is not None:
        command.extend(["--context-length", str(context_length)])
    if trust_remote_code:
        command.append("--trust-remote-code")
    if enable_p2p_check:
        command.append("--enable-p2p-check")
    if disable_custom_all_reduce:
        command.append("--disable-custom-all-reduce")
    if disable_overlap_schedule:
        command.append("--disable-overlap-schedule")
    if disable_cuda_graph:
        command.append("--disable-cuda-graph")
    if disable_piecewise_cuda_graph:
        command.append("--disable-piecewise-cuda-graph")
    if cuda_graph_max_bs is not None:
        command.extend(["--cuda-graph-max-bs", str(cuda_graph_max_bs)])
    return command


def build_trtllm_serve_command(
    *,
    executable: str,
    model: str,
    host: str,
    port: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int = 1,
    backend: str = DEFAULT_TRTLLM_BACKEND,
    tokenizer: str | None = None,
    max_batch_size: int | None = None,
    max_num_tokens: int | None = None,
    max_seq_len: int | None = None,
    log_level: str | None = None,
) -> list[str]:
    command = [
        executable,
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--backend",
        backend,
        "--tp_size",
        str(tensor_parallel_size),
    ]
    if pipeline_parallel_size != 1:
        command.extend(["--pp_size", str(pipeline_parallel_size)])
    if tokenizer is not None:
        command.extend(["--tokenizer", tokenizer])
    if max_batch_size is not None:
        command.extend(["--max_batch_size", str(max_batch_size)])
    if max_num_tokens is not None:
        command.extend(["--max_num_tokens", str(max_num_tokens)])
    if max_seq_len is not None:
        command.extend(["--max_seq_len", str(max_seq_len)])
    if log_level is not None:
        command.extend(["--log_level", str(log_level)])
    return command


def build_deepspeed_server_command(
    *,
    executable: str,
    host: str,
    port: int,
    model: str,
    tensor_parallel_size: int,
    script_path: str | None = None,
    launch_module: str | None = None,
    dtype: str = DEFAULT_DEEPSPEED_DTYPE,
    kernel_inject: bool = True,
    enable_cuda_graph: bool = False,
    use_triton: bool = False,
    triton_autotune: bool = False,
    checkpoint_dir: str | None = None,
    max_model_len: int | None = None,
) -> list[str]:
    command = [
        executable,
        "--num_gpus",
        str(tensor_parallel_size),
    ]
    if launch_module is not None:
        command.extend(["--module", launch_module])
    elif script_path is not None:
        command.append(script_path)
    else:
        raise WorkerError(
            "DeepSpeed launch requires either --script-path or --deepspeed-launch-module"
        )
    command.extend(
        [
            "--host",
            host,
            "--port",
            str(port),
            "--model",
            model,
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--dtype",
            dtype,
        ]
    )
    if max_model_len is not None:
        command.extend(["--max-model-len", str(max_model_len)])
    if kernel_inject:
        command.append("--kernel-inject")
    if enable_cuda_graph:
        command.append("--enable-cuda-graph")
    if use_triton:
        command.append("--use-triton")
    if triton_autotune:
        command.append("--triton-autotune")
    if checkpoint_dir is not None:
        command.extend(["--checkpoint-dir", checkpoint_dir])
    return command


def infer_packaged_cuda_home(python_executable: str) -> str | None:
    python_path = Path(expand_path_text(python_executable))
    venv_root = python_path.parent.parent
    candidate_paths: list[Path] = []
    for nvidia_root in sorted(venv_root.glob("lib/python*/site-packages/nvidia")):
        candidate_paths.extend(
            [
                nvidia_root / "cuda_nvcc",
                nvidia_root / "cuda_runtime",
                nvidia_root / "cu13",
                nvidia_root / "cu12",
            ]
        )
        for child in sorted(nvidia_root.iterdir()):
            if child.is_dir() and child.name.startswith("cu") and child not in candidate_paths:
                candidate_paths.append(child)
    fallback_candidate: Path | None = None
    fallback_nvcc_candidate: Path | None = None
    for candidate in candidate_paths:
        if fallback_candidate is None and candidate.exists():
            fallback_candidate = candidate
        if fallback_nvcc_candidate is None and (candidate / "bin" / "nvcc").exists():
            fallback_nvcc_candidate = candidate
            if (candidate / "include").exists() or (candidate / "nvvm").exists():
                return str(candidate)
        if (candidate / "include" / "cuda_runtime.h").exists():
            if fallback_nvcc_candidate is not None:
                return str(fallback_nvcc_candidate)
            return str(candidate)
    if fallback_nvcc_candidate is not None:
        return str(fallback_nvcc_candidate)
    return str(fallback_candidate) if fallback_candidate is not None else None


def infer_packaged_nvcc_path(executable_path: str) -> str | None:
    executable = Path(expand_path_text(executable_path))
    venv_root = executable.parent.parent
    for nvidia_root in sorted(venv_root.glob("lib/python*/site-packages/nvidia")):
        for candidate in (
            nvidia_root / "cuda_nvcc" / "bin" / "nvcc",
            nvidia_root / "cuda_runtime" / "bin" / "nvcc",
        ):
            if candidate.exists():
                return str(candidate.resolve())
    return None


def infer_system_nvcc_path() -> str | None:
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is not None:
        return str(Path(nvcc_path).resolve())
    for candidate in [
        Path("/usr/local/cuda/bin/nvcc"),
        *sorted(Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse=True),
    ]:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def infer_system_cuda_home() -> str | None:
    nvcc_path = infer_system_nvcc_path()
    if nvcc_path is not None:
        return str(Path(nvcc_path).parent.parent)
    for candidate in (
        Path("/usr/local/cuda"),
        *sorted(Path("/usr/local").glob("cuda-*"), reverse=True),
    ):
        if (candidate / "bin" / "nvcc").exists():
            return str(candidate)
    return None


def infer_packaged_cuda_bin_dirs(executable_path: str) -> tuple[str, ...]:
    executable = Path(expand_path_text(executable_path))
    venv_root = executable.parent.parent
    directories: list[str] = []
    for nvidia_root in sorted(venv_root.glob("lib/python*/site-packages/nvidia")):
        for candidate in (
            nvidia_root / "cuda_runtime" / "bin",
            nvidia_root / "cuda_nvcc" / "bin",
        ):
            if candidate.is_dir():
                value = str(candidate)
                if value not in directories:
                    directories.append(value)
    return tuple(directories)


def infer_runtime_python_executable(executable_path: str) -> str | None:
    executable = Path(expand_path_text(executable_path))
    venv_root = executable.parent.parent
    for candidate_name in ("python", "python3"):
        candidate = venv_root / "bin" / candidate_name
        if candidate.exists():
            return str(candidate)
    return None


def infer_torch_cuda_release(executable_path: str) -> str | None:
    python_executable = infer_runtime_python_executable(executable_path)
    if python_executable is None:
        return None
    try:
        result = subprocess.run(
            [
                python_executable,
                "-c",
                "import torch; print(torch.version.cuda or '')",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    release = result.stdout.strip()
    return release or None


def _write_nvcc_version_shim(target_path: Path, cuda_release: str) -> None:
    target_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'if [ "${1:-}" = "-V" ] || [ "${1:-}" = "--version" ]; then',
                f'  echo "Cuda compilation tools, release {cuda_release}, V{cuda_release}.0"',
                "  exit 0",
                "fi",
                'echo "nvcc shim: a real nvcc compiler is not installed in this runtime environment" >&2',
                "exit 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    target_path.chmod(0o755)


def repair_packaged_runtime_nvcc_link(
    executable_path: str,
    *,
    allow_version_shim: bool = False,
) -> dict[str, Any]:
    executable = Path(expand_path_text(executable_path))
    venv_root = executable.parent.parent
    selected_nvcc = infer_packaged_nvcc_path(executable_path) or infer_system_nvcc_path()
    result: dict[str, Any] = {
        "attempted": False,
        "updated": False,
        "runtime_bin": None,
        "nvcc_path": selected_nvcc,
        "shimmed": False,
    }
    for nvidia_root in sorted(venv_root.glob("lib/python*/site-packages/nvidia")):
        runtime_bin = nvidia_root / "cuda_runtime" / "bin"
        target_link = runtime_bin / "nvcc"
        result["runtime_bin"] = str(runtime_bin)
        result["attempted"] = True
        runtime_bin.mkdir(parents=True, exist_ok=True)

        if selected_nvcc is None and allow_version_shim:
            cuda_release = infer_torch_cuda_release(executable_path)
            if cuda_release:
                if target_link.exists():
                    if target_link.is_symlink():
                        target_link.unlink()
                    elif os.access(target_link, os.X_OK):
                        return result
                    else:
                        target_link.unlink()
                elif target_link.is_symlink():
                    target_link.unlink()
                _write_nvcc_version_shim(target_link, cuda_release)
                result["updated"] = True
                result["shimmed"] = True
                result["nvcc_path"] = str(target_link)
            return result

        if selected_nvcc is None:
            return result
        desired = Path(selected_nvcc)

        if target_link.exists():
            try:
                if target_link.resolve() == desired.resolve():
                    return result
            except OSError:
                pass
            if target_link.is_symlink():
                target_link.unlink()
            elif os.access(target_link, os.X_OK):
                return result
            else:
                target_link.unlink()
        elif target_link.is_symlink():
            target_link.unlink()

        target_link.symlink_to(desired)
        result["updated"] = True
        return result
    return result


def launch_module_supports_flag(
    python_executable: str,
    launch_module: str,
    flag: str,
    *,
    timeout_seconds: int = 30,
) -> bool:
    try:
        result = subprocess.run(
            [python_executable, "-m", launch_module, "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    help_text = f"{result.stdout}\n{result.stderr}"
    return flag in help_text


def prepend_executable_dir_to_path(env: dict[str, str], executable_path: str) -> None:
    executable_dir = str(Path(expand_path_text(executable_path)).parent)
    current_path = env.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if path_entries and path_entries[0] == executable_dir:
        return
    if executable_dir in path_entries:
        path_entries = [entry for entry in path_entries if entry != executable_dir]
    path_entries.insert(0, executable_dir)
    env["PATH"] = os.pathsep.join(path_entries)


def prepend_env_path_entries(env: dict[str, str], variable_name: str, entries: Iterable[str]) -> None:
    current_value = env.get(variable_name, "")
    current_entries = [entry for entry in current_value.split(os.pathsep) if entry] if current_value else []
    desired_entries = [entry for entry in entries if entry]
    for entry in reversed(desired_entries):
        if entry in current_entries:
            current_entries = [existing for existing in current_entries if existing != entry]
        current_entries.insert(0, entry)
    env[variable_name] = os.pathsep.join(current_entries)


def infer_packaged_library_dirs(executable_path: str) -> tuple[str, ...]:
    executable = Path(expand_path_text(executable_path))
    venv_root = executable.parent.parent
    candidate_paths: list[Path] = []
    for site_packages_root in sorted(venv_root.glob("lib/python*/site-packages")):
        tensorrt_libs = site_packages_root / "tensorrt_libs"
        if tensorrt_libs.is_dir():
            candidate_paths.append(tensorrt_libs)

        torch_lib = site_packages_root / "torch" / "lib"
        if torch_lib.is_dir():
            candidate_paths.append(torch_lib)

        nvidia_root = site_packages_root / "nvidia"
        if nvidia_root.is_dir():
            children = sorted(
                (child for child in nvidia_root.iterdir() if child.is_dir()),
                key=lambda child: (0 if child.name == "cu13" else 1 if child.name == "cu12" else 2, child.name),
            )
            for child in children:
                lib_dir = child / "lib"
                if lib_dir.is_dir():
                    candidate_paths.append(lib_dir)

    ordered_paths: list[str] = []
    for candidate in candidate_paths:
        candidate_text = str(candidate)
        if candidate_text not in ordered_paths:
            ordered_paths.append(candidate_text)
    return tuple(ordered_paths)


def read_json_url(url: str, *, timeout_seconds: int) -> dict[str, Any]:
    with request.urlopen(url, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise WorkerError(f"Expected JSON object from {url}")
    return payload


def post_json(url: str, payload: dict[str, Any], *, timeout_seconds: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8")
    reply = json.loads(raw)
    if not isinstance(reply, dict):
        raise WorkerError(f"Expected JSON object from {url}")
    return reply


def wait_for_json_endpoint(
    urls: Iterable[str],
    *,
    startup_timeout_seconds: int,
    request_timeout_seconds: int,
) -> tuple[str, dict[str, Any]]:
    deadline = time.monotonic() + startup_timeout_seconds
    last_error: str | None = None
    url_list = list(urls)
    while time.monotonic() < deadline:
        for url in url_list:
            try:
                return url, read_json_url(url, timeout_seconds=request_timeout_seconds)
            except (error.URLError, TimeoutError, json.JSONDecodeError, WorkerError) as exc:
                last_error = str(exc)
        time.sleep(0.5)
    raise WorkerError(
        "Timed out waiting for backend endpoint to become ready"
        + (f": {last_error}" if last_error else "")
    )


def wait_for_json_endpoint_or_process_exit(
    urls: Iterable[str],
    *,
    startup_timeout_seconds: int,
    request_timeout_seconds: int,
    process_pid: int,
    process_name: str,
    stderr_log: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    deadline = time.monotonic() + startup_timeout_seconds
    last_error: str | None = None
    url_list = list(urls)
    while time.monotonic() < deadline:
        try:
            waited_pid, wait_status = os.waitpid(process_pid, os.WNOHANG)
        except ChildProcessError:
            waited_pid = 0
            wait_status = 0
        if waited_pid == process_pid:
            if os.WIFEXITED(wait_status):
                process_status = f"exit code {os.WEXITSTATUS(wait_status)}"
            elif os.WIFSIGNALED(wait_status):
                process_status = f"signal {os.WTERMSIG(wait_status)}"
            else:
                process_status = "unknown exit status"
            log_hint = f"; see {stderr_log}" if stderr_log is not None else ""
            raise WorkerError(
                f"{process_name} exited before becoming ready ({process_status}){log_hint}"
            )
        for url in url_list:
            try:
                return url, read_json_url(url, timeout_seconds=request_timeout_seconds)
            except (error.URLError, TimeoutError, json.JSONDecodeError, WorkerError) as exc:
                last_error = str(exc)
        time.sleep(0.5)
    raise WorkerError(
        "Timed out waiting for backend endpoint to become ready"
        + (f": {last_error}" if last_error else "")
    )


def start_background_process(
    command: list[str],
    *,
    env: dict[str, str],
    stdout_log: Path,
    stderr_log: Path,
) -> int:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    with stdout_log.open("ab") as stdout_handle, stderr_log.open("ab") as stderr_handle:
        process = subprocess.Popen(
            command,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
            close_fds=True,
        )
    return int(process.pid)


def ensure_ollama_model_available(
    model: str,
    *,
    env: dict[str, str],
    timeout_seconds: int,
) -> None:
    completed = subprocess.run(
        ["ollama", "pull", model],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=True,
    )
    if completed.returncode != 0:
        raise WorkerError(f"Failed to make model {model!r} available in the dedicated Ollama runtime")


def load_session_payload(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def write_session_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def session_payload_gpu_indices(payload: dict[str, Any]) -> tuple[int, ...]:
    raw_gpu_indices = payload.get("gpu_indices")
    if isinstance(raw_gpu_indices, list) and raw_gpu_indices:
        try:
            return tuple(int(item) for item in raw_gpu_indices)
        except (TypeError, ValueError):
            return ()
    try:
        return (int(payload.get("gpu_index", -1)),)
    except (TypeError, ValueError):
        return ()


def session_payload_tensor_parallel_size(payload: dict[str, Any]) -> int:
    try:
        return int(payload.get("tensor_parallel_size", 1))
    except (TypeError, ValueError):
        return 1


def _coerce_positive_int(value: Any) -> int | None:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return coerced if coerced > 0 else None


def session_payload_process_group_id(payload: dict[str, Any]) -> int | None:
    explicit_group_id = _coerce_positive_int(payload.get("process_group_id"))
    if explicit_group_id is not None:
        return explicit_group_id
    return _coerce_positive_int(payload.get("server_pid"))


def session_payload_process_is_alive(payload: dict[str, Any]) -> bool:
    pid = _coerce_positive_int(payload.get("server_pid"))
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def list_process_group_members(process_group_id: int) -> list[dict[str, Any]]:
    if process_group_id <= 0:
        return []
    completed = subprocess.run(
        ["ps", "-ax", "-o", "pid=,pgid=,command="],
        capture_output=True,
        text=True,
        check=True,
    )
    members: list[dict[str, Any]] = []
    for raw_line in (completed.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            pgid = int(parts[1])
        except ValueError:
            continue
        if pgid != process_group_id:
            continue
        members.append({"pid": pid, "pgid": pgid, "command": parts[2]})
    return members


def session_payload_runtime_markers(payload: dict[str, Any]) -> tuple[str, ...]:
    backend = str(payload.get("backend") or "")
    runtime_markers: dict[str, tuple[str, ...]] = {
        "ollama": ("ollama",),
        "vllm": ("vllm.entrypoints.openai.api_server", "VLLM::EngineCore", "VLLM::Worker"),
        "sglang": ("sglang.launch_server", "sglang"),
        "deepspeed": ("deepspeed", "server_qwen_coder.py"),
        "tensorrt-llm": ("trtllm-serve", "tensorrt_llm", "TensorRT-LLM"),
        "trtllm": ("trtllm-serve", "tensorrt_llm", "TensorRT-LLM"),
        "python-hf": ("infer_gemma_pt.py",),
    }
    markers: list[str] = list(runtime_markers.get(backend, ()))
    command = payload.get("command")
    if isinstance(command, list):
        for item in command:
            text = str(item).strip()
            if not text:
                continue
            basename = Path(text).name
            if len(basename) >= 3:
                markers.append(basename)
            if "." in text or "/" in text:
                markers.append(text)
    deduped: list[str] = []
    seen: set[str] = set()
    for marker in markers:
        normalized = marker.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return tuple(deduped)


def _process_group_matches_runtime(
    members: list[dict[str, Any]],
    *,
    markers: tuple[str, ...],
) -> bool:
    if not members:
        return False
    if not markers:
        return True
    lower_markers = tuple(marker.lower() for marker in markers)
    for member in members:
        command = str(member.get("command") or "").lower()
        if any(marker in command for marker in lower_markers):
            return True
    return False


def terminate_session_processes(
    payload: dict[str, Any],
    *,
    grace_seconds: float = DEFAULT_PROCESS_STOP_GRACE_SECONDS,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "attempted": False,
        "signaled": False,
        "used_force_kill": False,
        "target": "none",
    }
    process_group_id = session_payload_process_group_id(payload)
    server_pid = _coerce_positive_int(payload.get("server_pid"))

    if process_group_id is not None:
        try:
            members_before = list_process_group_members(process_group_id)
        except (OSError, subprocess.SubprocessError):
            members_before = []
        result["process_group_id"] = process_group_id
        result["members_before"] = members_before
        markers = session_payload_runtime_markers(payload)
        if members_before and _process_group_matches_runtime(members_before, markers=markers):
            result["attempted"] = True
            result["target"] = "process-group"
            try:
                os.killpg(process_group_id, signal.SIGTERM)
                result["signaled"] = True
            except ProcessLookupError:
                result["already_gone"] = True
            except PermissionError:
                result["permission_denied"] = True
            except OSError:
                result["signal_error"] = True
            if result["signaled"]:
                deadline = time.monotonic() + max(float(grace_seconds), 0.0)
                members_after = members_before
                while True:
                    try:
                        members_after = list_process_group_members(process_group_id)
                    except (OSError, subprocess.SubprocessError):
                        members_after = []
                    if not members_after:
                        break
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.1)
                result["members_after_term"] = members_after
                if members_after:
                    try:
                        os.killpg(process_group_id, signal.SIGKILL)
                        result["used_force_kill"] = True
                    except ProcessLookupError:
                        result["already_gone"] = True
                    except PermissionError:
                        result["permission_denied"] = True
                    except OSError:
                        result["signal_error"] = True
                    try:
                        result["members_after_kill"] = list_process_group_members(process_group_id)
                    except (OSError, subprocess.SubprocessError):
                        result["members_after_kill"] = []
            return result
        if members_before:
            result["skipped_reason"] = "process-group-not-owned-by-managed-runtime"

    if server_pid is None:
        return result
    result["server_pid"] = server_pid
    result["attempted"] = True
    result["target"] = "pid"
    try:
        os.kill(server_pid, signal.SIGTERM)
        result["signaled"] = True
    except ProcessLookupError:
        result["already_gone"] = True
        return result
    except PermissionError:
        result["permission_denied"] = True
        return result
    except OSError:
        result["signal_error"] = True
        return result
    deadline = time.monotonic() + max(float(grace_seconds), 0.0)
    while True:
        if not session_payload_process_is_alive({"server_pid": server_pid}):
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    if session_payload_process_is_alive({"server_pid": server_pid}):
        try:
            os.kill(server_pid, signal.SIGKILL)
            result["used_force_kill"] = True
        except ProcessLookupError:
            result["already_gone"] = True
        except PermissionError:
            result["permission_denied"] = True
        except OSError:
            result["signal_error"] = True
    return result


def _parse_session_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def session_payload_is_stale_starting(
    payload: dict[str, Any],
    *,
    stale_after_seconds: int = DEFAULT_STALE_STARTING_SECONDS,
    now: datetime | None = None,
) -> bool:
    if stale_after_seconds <= 0:
        return False
    if str(payload.get("status") or "") != "starting":
        return False
    launched_at = _parse_session_timestamp(payload.get("launched_at"))
    if launched_at is None:
        return False
    current_time = now or datetime.now(tz=timezone.utc)
    return (current_time - launched_at).total_seconds() >= stale_after_seconds


def reap_stale_session_payload(path: Path, payload: dict[str, Any], *, reason: str) -> dict[str, Any] | None:
    previous_status = str(payload.get("status") or "")
    if previous_status not in SESSION_OK_STATUSES:
        return None
    updated = dict(payload)
    updated["status"] = "failed" if previous_status == "starting" else "stopped"
    updated["reused"] = False
    updated["reap_reason"] = reason
    updated["reaped_at"] = utc_timestamp()
    notes = str(updated.get("notes") or "").strip()
    suffix = f"Session reaped automatically ({reason})."
    updated["notes"] = f"{notes} {suffix}".strip() if notes else suffix
    write_session_payload(path, updated)
    return updated


def find_conflicting_session(
    session_dir: Path,
    *,
    agent_id: str,
    node_id: str,
    gpu_index: int,
    gpu_indices: tuple[int, ...] | None = None,
    listen_port: int | None,
) -> dict[str, Any] | None:
    if not session_dir.exists():
        return None
    requested_gpu_indices = gpu_indices or (gpu_index,)
    stale_reason = f"stale-starting>{DEFAULT_STALE_STARTING_SECONDS}s"
    for candidate in sorted(session_dir.glob("*.json")):
        payload = load_session_payload(candidate)
        if payload is None:
            continue
        if not session_payload_process_is_alive(payload):
            terminate_session_processes(payload)
            reap_stale_session_payload(candidate, payload, reason="dead-server-pid")
            continue
        if session_payload_is_stale_starting(
            payload,
            stale_after_seconds=DEFAULT_STALE_STARTING_SECONDS,
        ):
            terminate_session_processes(payload)
            reap_stale_session_payload(candidate, payload, reason=stale_reason)
            continue
        if payload.get("agent_id") == agent_id:
            continue
        if payload.get("node_id") != node_id:
            continue
        payload_gpu_indices = session_payload_gpu_indices(payload)
        if not payload_gpu_indices or not set(payload_gpu_indices).intersection(requested_gpu_indices):
            continue
        if listen_port is not None and payload.get("listen_port") not in (None, listen_port):
            continue
        if str(payload.get("status")) not in SESSION_OK_STATUSES:
            continue
        return payload
    return None


def resolve_launch_mode(args: argparse.Namespace) -> str:
    if args.launch_mode != "auto":
        return str(args.launch_mode)
    backend = str(args.backend)
    if backend == "ollama":
        return "ollama-server"
    if backend == "vllm":
        return "vllm-server"
    if backend == "sglang":
        return "sglang-server"
    if backend in {"tensorrt-llm", "trtllm"}:
        return "trtllm-server"
    if backend == "deepspeed":
        return "deepspeed-server"
    if backend == "python-hf":
        return "python-hf-probe"
    raise WorkerError(f"Unsupported backend {backend!r}")


def resolve_session_paths(session_dir: Path, agent_id: str) -> tuple[Path, Path, Path]:
    safe_agent_id = sanitize_agent_id(agent_id)
    session_path = session_dir / f"{safe_agent_id}.json"
    logs_dir = session_dir / "logs"
    return (
        session_path,
        logs_dir / f"{safe_agent_id}.stdout.log",
        logs_dir / f"{safe_agent_id}.stderr.log",
    )


class RuntimeAdapter(Protocol):
    name: str

    def launch(self, args: argparse.Namespace, session_dir: Path) -> WorkerSession:
        ...


class OllamaAdapter:
    name = "ollama-server"

    def launch(self, args: argparse.Namespace, session_dir: Path) -> WorkerSession:
        if shutil.which("ollama") is None:
            raise WorkerError("ollama is not installed on the target node")

        gpu_indices = resolve_gpu_indices(args)
        if len(gpu_indices) != 1:
            raise WorkerError("Ollama launch currently supports exactly one GPU")

        port_base = args.port_base or DEFAULT_OLLAMA_PORT_BASE
        port = choose_runtime_port_for_group(port_base, gpu_indices)
        conflict = find_conflicting_session(
            session_dir,
            agent_id=args.agent_id,
            node_id=args.node_id,
            gpu_index=args.gpu_index,
            gpu_indices=gpu_indices,
            listen_port=port,
        )
        if conflict is not None:
            raise WorkerError(
                f"GPU group {list(gpu_indices)} on {args.node_id} already has an active session for "
                f"agent {conflict.get('agent_id')!r}"
            )

        session_path, stdout_log, stderr_log = resolve_session_paths(session_dir, args.agent_id)
        endpoint_url = f"http://{args.server_host}:{port}"
        health_url = f"{endpoint_url}/api/tags"
        existing = load_session_payload(session_path)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_indices)
        env["OLLAMA_HOST"] = f"{args.server_host}:{port}"
        if (
            existing is not None
            and existing.get("listen_port") == port
            and existing.get("endpoint_url") == endpoint_url
            and session_payload_gpu_indices(existing) == gpu_indices
            and str(existing.get("status")) in SESSION_OK_STATUSES
        ):
            try:
                wait_for_json_endpoint(
                    [health_url],
                    startup_timeout_seconds=3,
                    request_timeout_seconds=2,
                )
                ensure_ollama_model_available(
                    args.model,
                    env=env,
                    timeout_seconds=args.request_timeout_seconds,
                )
                warmup = post_json(
                    f"{endpoint_url}/api/generate",
                    {
                        "model": args.model,
                        "prompt": args.warmup_prompt,
                        "stream": False,
                        "keep_alive": args.warmup_keep_alive,
                        "options": {"num_predict": args.max_new_tokens},
                    },
                    timeout_seconds=args.request_timeout_seconds,
                )
                response_text = str(warmup.get("response", "")).strip() or None
                session = WorkerSession(
                    status="reused",
                    agent_id=args.agent_id,
                    node_id=args.node_id,
                    backend=args.backend,
                    runtime_class=args.runtime_class,
                    model=args.model,
                    gpu_index=args.gpu_index,
                    gpu_indices=gpu_indices,
                    tensor_parallel_size=1,
                    single_gpu_only=True,
                    launched_at=utc_timestamp(),
                    session_file=str(session_path),
                    endpoint_url=endpoint_url,
                    listen_port=port,
                    server_pid=(
                        int(existing["server_pid"]) if existing.get("server_pid") is not None else None
                    ),
                    stdout_log=str(stdout_log),
                    stderr_log=str(stderr_log),
                    warmup_response=response_text,
                    command=build_ollama_server_command(),
                    health_url=health_url,
                    reused=True,
                    notes="Reused an existing dedicated Ollama runtime for this agent.",
                )
                write_session_payload(session_path, session.to_dict())
                return session
            except (WorkerError, error.URLError, json.JSONDecodeError):
                pass
        try:
            wait_for_json_endpoint(
                [health_url],
                startup_timeout_seconds=1,
                request_timeout_seconds=1,
            )
            raise WorkerError(
                f"Ollama endpoint {endpoint_url} is already live without a reusable session record"
            )
        except WorkerError as exc:
            if "already live" in str(exc):
                raise
        except (error.URLError, json.JSONDecodeError):
            pass

        write_session_payload(
            session_path,
            WorkerSession(
                status="starting",
                agent_id=args.agent_id,
                node_id=args.node_id,
                backend=args.backend,
                runtime_class=args.runtime_class,
                model=args.model,
                gpu_index=args.gpu_index,
                gpu_indices=gpu_indices,
                tensor_parallel_size=1,
                single_gpu_only=True,
                launched_at=utc_timestamp(),
                session_file=str(session_path),
                endpoint_url=endpoint_url,
                listen_port=port,
                stdout_log=str(stdout_log),
                stderr_log=str(stderr_log),
                health_url=health_url,
                notes="Phase 3 dedicated Ollama runtime is starting.",
            ).to_dict(),
        )

        server_pid = start_background_process(
            build_ollama_server_command(),
            env=env,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )
        write_session_payload(
            session_path,
            {
                **(load_session_payload(session_path) or {}),
                "server_pid": server_pid,
                "process_group_id": server_pid,
            },
        )
        wait_for_json_endpoint(
            [health_url],
            startup_timeout_seconds=args.startup_timeout_seconds,
            request_timeout_seconds=min(args.request_timeout_seconds, 10),
        )
        ensure_ollama_model_available(
            args.model,
            env=env,
            timeout_seconds=args.request_timeout_seconds,
        )
        warmup = post_json(
            f"{endpoint_url}/api/generate",
            {
                "model": args.model,
                "prompt": args.warmup_prompt,
                "stream": False,
                "keep_alive": args.warmup_keep_alive,
                "options": {"num_predict": args.max_new_tokens},
            },
            timeout_seconds=args.request_timeout_seconds,
        )
        response_text = str(warmup.get("response", "")).strip() or None

        session = WorkerSession(
            status="launched",
            agent_id=args.agent_id,
            node_id=args.node_id,
            backend=args.backend,
            runtime_class=args.runtime_class,
            model=args.model,
            gpu_index=args.gpu_index,
            gpu_indices=gpu_indices,
            tensor_parallel_size=1,
            single_gpu_only=True,
            launched_at=utc_timestamp(),
            session_file=str(session_path),
            endpoint_url=endpoint_url,
            listen_port=port,
            server_pid=server_pid,
            process_group_id=server_pid,
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            warmup_response=response_text,
            command=build_ollama_server_command(),
            health_url=health_url,
            notes="Dedicated Ollama server launched and the target model was warmed.",
        )
        write_session_payload(session_path, session.to_dict())
        return session


class VllmAdapter:
    name = "vllm-server"

    def launch(self, args: argparse.Namespace, session_dir: Path) -> WorkerSession:
        gpu_indices = resolve_gpu_indices(args)
        if args.tensor_parallel_size != len(gpu_indices):
            raise WorkerError(
                f"tensor_parallel_size={args.tensor_parallel_size} requires exactly "
                f"{args.tensor_parallel_size} GPUs, but gpu_indices={gpu_indices}"
            )
        python_executable = expand_path_text(args.python_executable)
        port_base = args.port_base or DEFAULT_VLLM_PORT_BASE
        port = choose_runtime_port_for_group(port_base, gpu_indices)
        conflict = find_conflicting_session(
            session_dir,
            agent_id=args.agent_id,
            node_id=args.node_id,
            gpu_index=args.gpu_index,
            gpu_indices=gpu_indices,
            listen_port=port,
        )
        if conflict is not None:
            raise WorkerError(
                f"GPU group {list(gpu_indices)} on {args.node_id} already has an active session for "
                f"agent {conflict.get('agent_id')!r}"
            )

        session_path, stdout_log, stderr_log = resolve_session_paths(session_dir, args.agent_id)
        endpoint_url = f"http://{args.server_host}:{port}"
        health_candidates = [f"{endpoint_url}/health", f"{endpoint_url}/v1/models"]
        existing = load_session_payload(session_path)
        command = build_vllm_server_command(
            python_executable=python_executable,
            launch_module=args.vllm_launch_module,
            host=args.server_host,
            port=port,
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            enforce_eager=args.enforce_eager,
        )
        if (
            existing is not None
            and existing.get("listen_port") == port
            and existing.get("endpoint_url") == endpoint_url
            and session_payload_gpu_indices(existing) == gpu_indices
            and session_payload_tensor_parallel_size(existing) == args.tensor_parallel_size
            and str(existing.get("status")) in SESSION_OK_STATUSES
        ):
            try:
                ready_url, _ = wait_for_json_endpoint(
                    health_candidates,
                    startup_timeout_seconds=3,
                    request_timeout_seconds=2,
                )
                session = WorkerSession(
                    status="reused",
                    agent_id=args.agent_id,
                    node_id=args.node_id,
                    backend=args.backend,
                    runtime_class=args.runtime_class,
                    model=args.model,
                    gpu_index=args.gpu_index,
                    gpu_indices=gpu_indices,
                    tensor_parallel_size=args.tensor_parallel_size,
                    single_gpu_only=len(gpu_indices) == 1,
                    launched_at=utc_timestamp(),
                    session_file=str(session_path),
                    endpoint_url=endpoint_url,
                    listen_port=port,
                    server_pid=(
                        int(existing["server_pid"]) if existing.get("server_pid") is not None else None
                    ),
                    stdout_log=str(stdout_log),
                    stderr_log=str(stderr_log),
                    command=command,
                    health_url=ready_url,
                    reused=True,
                    notes="Reused an existing dedicated vLLM runtime for this agent.",
                )
                write_session_payload(session_path, session.to_dict())
                return session
            except (WorkerError, error.URLError, json.JSONDecodeError):
                pass
        try:
            wait_for_json_endpoint(
                health_candidates,
                startup_timeout_seconds=1,
                request_timeout_seconds=1,
            )
            raise WorkerError(
                f"vLLM endpoint {endpoint_url} is already live without a reusable session record"
            )
        except WorkerError as exc:
            if "already live" in str(exc):
                raise
        except (error.URLError, json.JSONDecodeError):
            pass

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_indices)
        prepend_executable_dir_to_path(env, python_executable)
        write_session_payload(
            session_path,
            WorkerSession(
                status="starting",
                agent_id=args.agent_id,
                node_id=args.node_id,
                backend=args.backend,
                runtime_class=args.runtime_class,
                model=args.model,
                gpu_index=args.gpu_index,
                gpu_indices=gpu_indices,
                tensor_parallel_size=args.tensor_parallel_size,
                single_gpu_only=len(gpu_indices) == 1,
                launched_at=utc_timestamp(),
                session_file=str(session_path),
                endpoint_url=endpoint_url,
                listen_port=port,
                stdout_log=str(stdout_log),
                stderr_log=str(stderr_log),
                command=command,
                health_url=health_candidates[0],
                notes="Phase 3 dedicated vLLM runtime is starting.",
            ).to_dict(),
        )

        server_pid = start_background_process(
            command,
            env=env,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )
        write_session_payload(
            session_path,
            {
                **(load_session_payload(session_path) or {}),
                "server_pid": server_pid,
                "process_group_id": server_pid,
            },
        )
        ready_url, _ = wait_for_json_endpoint(
            health_candidates,
            startup_timeout_seconds=args.startup_timeout_seconds,
            request_timeout_seconds=min(args.request_timeout_seconds, 10),
        )
        session = WorkerSession(
            status="launched",
            agent_id=args.agent_id,
            node_id=args.node_id,
            backend=args.backend,
            runtime_class=args.runtime_class,
            model=args.model,
            gpu_index=args.gpu_index,
            gpu_indices=gpu_indices,
            tensor_parallel_size=args.tensor_parallel_size,
            single_gpu_only=len(gpu_indices) == 1,
            launched_at=utc_timestamp(),
            session_file=str(session_path),
            endpoint_url=endpoint_url,
            listen_port=port,
            server_pid=server_pid,
            process_group_id=server_pid,
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            command=command,
            health_url=ready_url,
            notes="Dedicated vLLM server launched and passed a health probe.",
        )
        write_session_payload(session_path, session.to_dict())
        return session


class SglangAdapter:
    name = "sglang-server"

    def launch(self, args: argparse.Namespace, session_dir: Path) -> WorkerSession:
        gpu_indices = resolve_gpu_indices(args)
        if args.tensor_parallel_size != len(gpu_indices):
            raise WorkerError(
                f"tensor_parallel_size={args.tensor_parallel_size} requires exactly "
                f"{args.tensor_parallel_size} GPUs, but gpu_indices={gpu_indices}"
            )
        python_executable = expand_path_text(args.python_executable)
        port_base = args.port_base or DEFAULT_SGLANG_PORT_BASE
        port = choose_runtime_port_for_group(port_base, gpu_indices)
        conflict = find_conflicting_session(
            session_dir,
            agent_id=args.agent_id,
            node_id=args.node_id,
            gpu_index=args.gpu_index,
            gpu_indices=gpu_indices,
            listen_port=port,
        )
        if conflict is not None:
            raise WorkerError(
                f"GPU group {list(gpu_indices)} on {args.node_id} already has an active session for "
                f"agent {conflict.get('agent_id')!r}"
            )

        session_path, stdout_log, stderr_log = resolve_session_paths(session_dir, args.agent_id)
        endpoint_url = f"http://{args.server_host}:{port}"
        health_candidates = [f"{endpoint_url}/health", f"{endpoint_url}/v1/models"]
        existing = load_session_payload(session_path)
        command = build_sglang_server_command(
            python_executable=python_executable,
            launch_module=args.sglang_launch_module,
            host=args.server_host,
            port=port,
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            mem_fraction_static=args.mem_fraction_static,
            context_length=args.context_length,
            trust_remote_code=args.trust_remote_code,
            enable_p2p_check=args.enable_p2p_check,
            disable_custom_all_reduce=args.disable_custom_all_reduce,
            disable_overlap_schedule=args.disable_overlap_schedule,
            disable_cuda_graph=args.disable_cuda_graph,
            disable_piecewise_cuda_graph=launch_module_supports_flag(
                python_executable,
                args.sglang_launch_module,
                "--disable-piecewise-cuda-graph",
            ),
            cuda_graph_max_bs=args.cuda_graph_max_bs,
        )
        if (
            existing is not None
            and existing.get("listen_port") == port
            and existing.get("endpoint_url") == endpoint_url
            and session_payload_gpu_indices(existing) == gpu_indices
            and session_payload_tensor_parallel_size(existing) == args.tensor_parallel_size
            and str(existing.get("status")) in SESSION_OK_STATUSES
        ):
            try:
                ready_url, _ = wait_for_json_endpoint(
                    health_candidates,
                    startup_timeout_seconds=3,
                    request_timeout_seconds=2,
                )
                session = WorkerSession(
                    status="reused",
                    agent_id=args.agent_id,
                    node_id=args.node_id,
                    backend=args.backend,
                    runtime_class=args.runtime_class,
                    model=args.model,
                    gpu_index=args.gpu_index,
                    gpu_indices=gpu_indices,
                    tensor_parallel_size=args.tensor_parallel_size,
                    single_gpu_only=len(gpu_indices) == 1,
                    launched_at=utc_timestamp(),
                    session_file=str(session_path),
                    endpoint_url=endpoint_url,
                    listen_port=port,
                    server_pid=(
                        int(existing["server_pid"]) if existing.get("server_pid") is not None else None
                    ),
                    stdout_log=str(stdout_log),
                    stderr_log=str(stderr_log),
                    command=command,
                    health_url=ready_url,
                    reused=True,
                    notes="Reused an existing dedicated SGLang runtime for this agent.",
                )
                write_session_payload(session_path, session.to_dict())
                return session
            except (WorkerError, error.URLError, json.JSONDecodeError):
                pass
        try:
            wait_for_json_endpoint(
                health_candidates,
                startup_timeout_seconds=1,
                request_timeout_seconds=1,
            )
            raise WorkerError(
                f"SGLang endpoint {endpoint_url} is already live without a reusable session record"
            )
        except WorkerError as exc:
            if "already live" in str(exc):
                raise
        except (error.URLError, json.JSONDecodeError):
            pass

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_indices)
        prepend_executable_dir_to_path(env, python_executable)
        repair_packaged_runtime_nvcc_link(python_executable)
        prepend_env_path_entries(env, "PATH", infer_packaged_cuda_bin_dirs(python_executable))
        cuda_home = (
            env.get("CUDA_HOME")
            or env.get("CUDA_PATH")
            or infer_packaged_cuda_home(python_executable)
            or infer_system_cuda_home()
        )
        if cuda_home is not None:
            env.setdefault("CUDA_HOME", cuda_home)
            env.setdefault("CUDA_PATH", cuda_home)
        server_pid = start_background_process(
            command,
            env=env,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )
        write_session_payload(
            session_path,
            WorkerSession(
                status="starting",
                agent_id=args.agent_id,
                node_id=args.node_id,
                backend=args.backend,
                runtime_class=args.runtime_class,
                model=args.model,
                gpu_index=args.gpu_index,
                gpu_indices=gpu_indices,
                tensor_parallel_size=args.tensor_parallel_size,
                single_gpu_only=len(gpu_indices) == 1,
                launched_at=utc_timestamp(),
                session_file=str(session_path),
                endpoint_url=endpoint_url,
                listen_port=port,
                server_pid=server_pid,
                process_group_id=server_pid,
                stdout_log=str(stdout_log),
                stderr_log=str(stderr_log),
                command=command,
                health_url=health_candidates[0],
                notes="Dedicated SGLang runtime is starting.",
            ).to_dict(),
        )

        ready_url, _ = wait_for_json_endpoint_or_process_exit(
            health_candidates,
            startup_timeout_seconds=args.startup_timeout_seconds,
            request_timeout_seconds=min(args.request_timeout_seconds, 10),
            process_pid=server_pid,
            process_name="SGLang server",
            stderr_log=stderr_log,
        )
        session = WorkerSession(
            status="launched",
            agent_id=args.agent_id,
            node_id=args.node_id,
            backend=args.backend,
            runtime_class=args.runtime_class,
            model=args.model,
            gpu_index=args.gpu_index,
            gpu_indices=gpu_indices,
            tensor_parallel_size=args.tensor_parallel_size,
            single_gpu_only=len(gpu_indices) == 1,
            launched_at=utc_timestamp(),
            session_file=str(session_path),
            endpoint_url=endpoint_url,
            listen_port=port,
            server_pid=server_pid,
            process_group_id=server_pid,
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            command=command,
            health_url=ready_url,
            notes="Dedicated SGLang server launched and passed a health probe.",
        )
        write_session_payload(session_path, session.to_dict())
        return session


class DeepSpeedAdapter:
    name = "deepspeed-server"

    def launch(self, args: argparse.Namespace, session_dir: Path) -> WorkerSession:
        gpu_indices = resolve_gpu_indices(args)
        if args.tensor_parallel_size != len(gpu_indices):
            raise WorkerError(
                f"tensor_parallel_size={args.tensor_parallel_size} requires exactly "
                f"{args.tensor_parallel_size} GPUs, but gpu_indices={gpu_indices}"
            )

        deepspeed_executable = expand_path_text(args.deepspeed_executable)
        if "/" in args.deepspeed_executable and not Path(deepspeed_executable).exists():
            raise WorkerError(
                f"DeepSpeed executable {deepspeed_executable!r} is not present on the target node"
            )
        if "/" not in args.deepspeed_executable:
            resolved_executable = shutil.which(deepspeed_executable)
            if resolved_executable is None:
                raise WorkerError(
                    f"DeepSpeed executable {deepspeed_executable!r} is not installed on the target node"
                )
            deepspeed_executable = resolved_executable

        port_base = args.port_base or DEFAULT_DEEPSPEED_PORT_BASE
        port = choose_runtime_port_for_group(port_base, gpu_indices)
        conflict = find_conflicting_session(
            session_dir,
            agent_id=args.agent_id,
            node_id=args.node_id,
            gpu_index=args.gpu_index,
            gpu_indices=gpu_indices,
            listen_port=port,
        )
        if conflict is not None:
            raise WorkerError(
                f"GPU group {list(gpu_indices)} on {args.node_id} already has an active session for "
                f"agent {conflict.get('agent_id')!r}"
            )

        session_path, stdout_log, stderr_log = resolve_session_paths(session_dir, args.agent_id)
        endpoint_url = f"http://{args.server_host}:{port}"
        health_candidates = [f"{endpoint_url}/health", f"{endpoint_url}/v1/models"]
        existing = load_session_payload(session_path)
        command = build_deepspeed_server_command(
            executable=deepspeed_executable,
            host=args.server_host,
            port=port,
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            script_path=args.script_path,
            launch_module=args.deepspeed_launch_module,
            dtype=args.deepspeed_dtype,
            kernel_inject=args.deepspeed_kernel_inject,
            enable_cuda_graph=args.deepspeed_enable_cuda_graph,
            use_triton=args.deepspeed_use_triton,
            triton_autotune=args.deepspeed_triton_autotune,
            checkpoint_dir=args.deepspeed_checkpoint_dir,
            max_model_len=args.max_model_len,
        )
        if (
            existing is not None
            and existing.get("listen_port") == port
            and existing.get("endpoint_url") == endpoint_url
            and session_payload_gpu_indices(existing) == gpu_indices
            and session_payload_tensor_parallel_size(existing) == args.tensor_parallel_size
            and str(existing.get("status")) in SESSION_OK_STATUSES
        ):
            try:
                ready_url, _ = wait_for_json_endpoint(
                    health_candidates,
                    startup_timeout_seconds=3,
                    request_timeout_seconds=2,
                )
                session = WorkerSession(
                    status="reused",
                    agent_id=args.agent_id,
                    node_id=args.node_id,
                    backend=args.backend,
                    runtime_class=args.runtime_class,
                    model=args.model,
                    gpu_index=args.gpu_index,
                    gpu_indices=gpu_indices,
                    tensor_parallel_size=args.tensor_parallel_size,
                    single_gpu_only=len(gpu_indices) == 1,
                    launched_at=utc_timestamp(),
                    session_file=str(session_path),
                    endpoint_url=endpoint_url,
                    listen_port=port,
                    server_pid=(
                        int(existing["server_pid"]) if existing.get("server_pid") is not None else None
                    ),
                    stdout_log=str(stdout_log),
                    stderr_log=str(stderr_log),
                    command=command,
                    health_url=ready_url,
                    reused=True,
                    notes="Reused an existing dedicated DeepSpeed runtime for this agent.",
                )
                write_session_payload(session_path, session.to_dict())
                return session
            except (WorkerError, error.URLError, json.JSONDecodeError):
                pass
        try:
            wait_for_json_endpoint(
                health_candidates,
                startup_timeout_seconds=1,
                request_timeout_seconds=1,
            )
            raise WorkerError(
                f"DeepSpeed endpoint {endpoint_url} is already live without a reusable session record"
            )
        except WorkerError as exc:
            if "already live" in str(exc):
                raise
        except (error.URLError, json.JSONDecodeError):
            pass

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_indices)
        prepend_executable_dir_to_path(env, deepspeed_executable)
        repair_packaged_runtime_nvcc_link(deepspeed_executable, allow_version_shim=True)
        prepend_env_path_entries(env, "PATH", infer_packaged_cuda_bin_dirs(deepspeed_executable))
        cuda_home = (
            env.get("CUDA_HOME")
            or env.get("CUDA_PATH")
            or infer_packaged_cuda_home(deepspeed_executable)
            or infer_system_cuda_home()
        )
        if cuda_home is not None:
            env.setdefault("CUDA_HOME", cuda_home)
            env.setdefault("CUDA_PATH", cuda_home)
        write_session_payload(
            session_path,
            WorkerSession(
                status="starting",
                agent_id=args.agent_id,
                node_id=args.node_id,
                backend=args.backend,
                runtime_class=args.runtime_class,
                model=args.model,
                gpu_index=args.gpu_index,
                gpu_indices=gpu_indices,
                tensor_parallel_size=args.tensor_parallel_size,
                single_gpu_only=len(gpu_indices) == 1,
                launched_at=utc_timestamp(),
                session_file=str(session_path),
                endpoint_url=endpoint_url,
                listen_port=port,
                stdout_log=str(stdout_log),
                stderr_log=str(stderr_log),
                command=command,
                health_url=health_candidates[0],
                notes="Dedicated DeepSpeed runtime is starting.",
            ).to_dict(),
        )

        server_pid = start_background_process(
            command,
            env=env,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )
        write_session_payload(
            session_path,
            {
                **(load_session_payload(session_path) or {}),
                "server_pid": server_pid,
                "process_group_id": server_pid,
            },
        )
        ready_url, _ = wait_for_json_endpoint_or_process_exit(
            health_candidates,
            startup_timeout_seconds=args.startup_timeout_seconds,
            request_timeout_seconds=min(args.request_timeout_seconds, 10),
            process_pid=server_pid,
            process_name="DeepSpeed server",
            stderr_log=stderr_log,
        )
        session = WorkerSession(
            status="launched",
            agent_id=args.agent_id,
            node_id=args.node_id,
            backend=args.backend,
            runtime_class=args.runtime_class,
            model=args.model,
            gpu_index=args.gpu_index,
            gpu_indices=gpu_indices,
            tensor_parallel_size=args.tensor_parallel_size,
            single_gpu_only=len(gpu_indices) == 1,
            launched_at=utc_timestamp(),
            session_file=str(session_path),
            endpoint_url=endpoint_url,
            listen_port=port,
            server_pid=server_pid,
            process_group_id=server_pid,
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            command=command,
            health_url=ready_url,
            notes="Dedicated DeepSpeed server launched and passed a health probe.",
        )
        write_session_payload(session_path, session.to_dict())
        return session


class TrtllmAdapter:
    name = "trtllm-server"

    def launch(self, args: argparse.Namespace, session_dir: Path) -> WorkerSession:
        gpu_indices = resolve_gpu_indices(args)
        expected_gpu_count = int(args.tensor_parallel_size) * int(args.pipeline_parallel_size)
        if expected_gpu_count <= 0:
            raise WorkerError("TensorRT-LLM requires positive tp/pp sizes")
        if len(gpu_indices) != expected_gpu_count:
            raise WorkerError(
                f"TensorRT-LLM requires tp*pp={expected_gpu_count} GPUs, "
                f"but gpu_indices={gpu_indices}"
            )
        trtllm_executable = expand_path_text(args.trtllm_executable)
        if "/" in args.trtllm_executable and not Path(trtllm_executable).exists():
            raise WorkerError(
                f"TensorRT-LLM executable {trtllm_executable!r} is not present on the target node"
            )
        if "/" not in args.trtllm_executable:
            resolved_executable = shutil.which(trtllm_executable)
            if resolved_executable is None:
                raise WorkerError(
                    f"TensorRT-LLM executable {trtllm_executable!r} is not installed on the target node"
                )
            trtllm_executable = resolved_executable
        if args.trtllm_backend == "trt" and not args.trtllm_tokenizer:
            raise WorkerError(
                "TensorRT-LLM backend=trt requires --trtllm-tokenizer when serving an engine path"
            )

        port_base = args.port_base or DEFAULT_TRTLLM_PORT_BASE
        port = choose_runtime_port_for_group(port_base, gpu_indices)
        conflict = find_conflicting_session(
            session_dir,
            agent_id=args.agent_id,
            node_id=args.node_id,
            gpu_index=args.gpu_index,
            gpu_indices=gpu_indices,
            listen_port=port,
        )
        if conflict is not None:
            raise WorkerError(
                f"GPU group {list(gpu_indices)} on {args.node_id} already has an active session for "
                f"agent {conflict.get('agent_id')!r}"
            )

        session_path, stdout_log, stderr_log = resolve_session_paths(session_dir, args.agent_id)
        endpoint_url = f"http://{args.server_host}:{port}"
        health_candidates = [f"{endpoint_url}/health", f"{endpoint_url}/v1/models"]
        existing = load_session_payload(session_path)
        command = build_trtllm_serve_command(
            executable=trtllm_executable,
            model=args.model,
            host=args.server_host,
            port=port,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            backend=args.trtllm_backend,
            tokenizer=args.trtllm_tokenizer,
            max_batch_size=args.trtllm_max_batch_size,
            max_num_tokens=args.trtllm_max_num_tokens,
            max_seq_len=args.trtllm_max_seq_len,
            log_level=args.trtllm_log_level,
        )
        if (
            existing is not None
            and existing.get("listen_port") == port
            and existing.get("endpoint_url") == endpoint_url
            and session_payload_gpu_indices(existing) == gpu_indices
            and existing.get("command") == command
            and str(existing.get("status")) in SESSION_OK_STATUSES
        ):
            try:
                ready_url, _ = wait_for_json_endpoint(
                    health_candidates,
                    startup_timeout_seconds=3,
                    request_timeout_seconds=2,
                )
                session = WorkerSession(
                    status="reused",
                    agent_id=args.agent_id,
                    node_id=args.node_id,
                    backend=args.backend,
                    runtime_class=args.runtime_class,
                    model=args.model,
                    gpu_index=args.gpu_index,
                    gpu_indices=gpu_indices,
                    tensor_parallel_size=args.tensor_parallel_size,
                    single_gpu_only=len(gpu_indices) == 1,
                    launched_at=utc_timestamp(),
                    session_file=str(session_path),
                    endpoint_url=endpoint_url,
                    listen_port=port,
                    server_pid=(
                        int(existing["server_pid"]) if existing.get("server_pid") is not None else None
                    ),
                    stdout_log=str(stdout_log),
                    stderr_log=str(stderr_log),
                    command=command,
                    health_url=ready_url,
                    reused=True,
                    notes="Reused an existing dedicated TensorRT-LLM runtime for this agent.",
                )
                write_session_payload(session_path, session.to_dict())
                return session
            except (WorkerError, error.URLError, json.JSONDecodeError):
                pass
        try:
            wait_for_json_endpoint(
                health_candidates,
                startup_timeout_seconds=1,
                request_timeout_seconds=1,
            )
            raise WorkerError(
                f"TensorRT-LLM endpoint {endpoint_url} is already live without a reusable session record"
            )
        except WorkerError as exc:
            if "already live" in str(exc):
                raise
        except (error.URLError, json.JSONDecodeError):
            pass

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_indices)
        prepend_executable_dir_to_path(env, trtllm_executable)
        repair_packaged_runtime_nvcc_link(trtllm_executable)
        prepend_env_path_entries(env, "PATH", infer_packaged_cuda_bin_dirs(trtllm_executable))
        prepend_env_path_entries(env, "LD_LIBRARY_PATH", infer_packaged_library_dirs(trtllm_executable))
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        packaged_cuda_home = infer_packaged_cuda_home(trtllm_executable)
        if packaged_cuda_home is not None:
            env["CUDA_HOME"] = packaged_cuda_home
        launched_at = utc_timestamp()

        server_pid = start_background_process(
            command,
            env=env,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )
        write_session_payload(
            session_path,
            WorkerSession(
                status="starting",
                agent_id=args.agent_id,
                node_id=args.node_id,
                backend=args.backend,
                runtime_class=args.runtime_class,
                model=args.model,
                gpu_index=args.gpu_index,
                gpu_indices=gpu_indices,
                tensor_parallel_size=args.tensor_parallel_size,
                single_gpu_only=len(gpu_indices) == 1,
                launched_at=launched_at,
                session_file=str(session_path),
                endpoint_url=endpoint_url,
                listen_port=port,
                server_pid=server_pid,
                process_group_id=server_pid,
                stdout_log=str(stdout_log),
                stderr_log=str(stderr_log),
                command=command,
                health_url=health_candidates[0],
                notes="Dedicated TensorRT-LLM runtime is starting.",
            ).to_dict(),
        )
        ready_url, _ = wait_for_json_endpoint_or_process_exit(
            health_candidates,
            startup_timeout_seconds=args.startup_timeout_seconds,
            request_timeout_seconds=min(args.request_timeout_seconds, 10),
            process_pid=server_pid,
            process_name="TensorRT-LLM server",
            stderr_log=stderr_log,
        )
        session = WorkerSession(
            status="launched",
            agent_id=args.agent_id,
            node_id=args.node_id,
            backend=args.backend,
            runtime_class=args.runtime_class,
            model=args.model,
            gpu_index=args.gpu_index,
            gpu_indices=gpu_indices,
            tensor_parallel_size=args.tensor_parallel_size,
            single_gpu_only=len(gpu_indices) == 1,
            launched_at=launched_at,
            session_file=str(session_path),
            endpoint_url=endpoint_url,
            listen_port=port,
            server_pid=server_pid,
            process_group_id=server_pid,
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            command=command,
            health_url=ready_url,
            notes="Dedicated TensorRT-LLM server launched and passed a health probe.",
        )
        write_session_payload(session_path, session.to_dict())
        return session


class PythonHfProbeAdapter:
    name = "python-hf-probe"

    def launch(self, args: argparse.Namespace, session_dir: Path) -> WorkerSession:
        gpu_indices = resolve_gpu_indices(args)
        if len(gpu_indices) != 1:
            raise WorkerError("Python/HF probe currently supports exactly one GPU")
        python_executable = expand_path_text(args.python_executable)
        script_path = expand_path_text(args.script_path or "scripts/gemma_pt/infer_gemma_pt.py")
        session_path, stdout_log, stderr_log = resolve_session_paths(session_dir, args.agent_id)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_indices)
        prepend_executable_dir_to_path(env, python_executable)
        command = [
            python_executable,
            script_path,
            "--model-id",
            args.model,
            "--prompt",
            args.warmup_prompt,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            "0",
        ]

        completed = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            timeout=args.request_timeout_seconds,
            check=True,
        )
        stdout_text = (completed.stdout or "").strip() or None
        stdout_log.parent.mkdir(parents=True, exist_ok=True)
        stdout_log.write_text((completed.stdout or "") + "\n", encoding="utf-8")
        stderr_log.write_text((completed.stderr or "") + "\n", encoding="utf-8")

        session = WorkerSession(
            status="completed",
            agent_id=args.agent_id,
            node_id=args.node_id,
            backend=args.backend,
            runtime_class=args.runtime_class,
            model=args.model,
            gpu_index=args.gpu_index,
            gpu_indices=gpu_indices,
            tensor_parallel_size=1,
            single_gpu_only=True,
            launched_at=utc_timestamp(),
            session_file=str(session_path),
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            warmup_response=stdout_text,
            command=command,
            notes="Python/HF probe completed on the target GPU.",
        )
        write_session_payload(session_path, session.to_dict())
        return session


_RUNTIME_ADAPTERS: dict[str, RuntimeAdapter] = {
    adapter.name: adapter
    for adapter in (
        OllamaAdapter(),
        VllmAdapter(),
        SglangAdapter(),
        DeepSpeedAdapter(),
        TrtllmAdapter(),
        PythonHfProbeAdapter(),
    )
}


def get_runtime_adapter(name: str) -> RuntimeAdapter:
    try:
        return _RUNTIME_ADAPTERS[name]
    except KeyError as exc:
        raise WorkerError(f"Unsupported launch mode {name!r}") from exc


def launch_with_adapter(args: argparse.Namespace, session_dir: Path) -> WorkerSession:
    adapter = get_runtime_adapter(resolve_launch_mode(args))
    return adapter.launch(args, session_dir)


__all__ = [
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_OLLAMA_PORT_BASE",
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "DEFAULT_DEEPSPEED_DTYPE",
    "DEFAULT_DEEPSPEED_EXECUTABLE",
    "DEFAULT_DEEPSPEED_PORT_BASE",
    "DEFAULT_SERVER_HOST",
    "DEFAULT_SESSION_DIR",
    "DEFAULT_STALE_STARTING_SECONDS",
    "DEFAULT_STARTUP_TIMEOUT_SECONDS",
    "DEFAULT_SGLANG_LAUNCH_MODULE",
    "DEFAULT_SGLANG_PORT_BASE",
    "DEFAULT_TRTLLM_BACKEND",
    "DEFAULT_TRTLLM_EXECUTABLE",
    "DEFAULT_TRTLLM_PORT_BASE",
    "DEFAULT_MEM_FRACTION_STATIC",
    "DEFAULT_VLLM_LAUNCH_MODULE",
    "DEFAULT_VLLM_PORT_BASE",
    "DEFAULT_WARMUP_PROMPT",
    "SESSION_OK_STATUSES",
    "WorkerError",
    "WorkerSession",
    "build_ollama_server_command",
    "build_deepspeed_server_command",
    "build_sglang_server_command",
    "build_trtllm_serve_command",
    "build_vllm_server_command",
    "choose_runtime_port",
    "choose_runtime_port_for_group",
    "ensure_ollama_model_available",
    "expand_path_text",
    "find_conflicting_session",
    "get_runtime_adapter",
    "infer_packaged_cuda_bin_dirs",
    "infer_packaged_cuda_home",
    "infer_packaged_nvcc_path",
    "infer_system_cuda_home",
    "infer_system_nvcc_path",
    "infer_packaged_library_dirs",
    "launch_module_supports_flag",
    "launch_with_adapter",
    "load_session_payload",
    "post_json",
    "prepend_executable_dir_to_path",
    "prepend_env_path_entries",
    "read_json_url",
    "resolve_gpu_indices",
    "resolve_launch_mode",
    "resolve_session_paths",
    "repair_packaged_runtime_nvcc_link",
    "sanitize_agent_id",
    "session_payload_is_stale_starting",
    "session_payload_process_is_alive",
    "session_payload_gpu_indices",
    "session_payload_tensor_parallel_size",
    "reap_stale_session_payload",
    "session_payload_process_group_id",
    "start_background_process",
    "terminate_session_processes",
    "utc_timestamp",
    "wait_for_json_endpoint",
    "wait_for_json_endpoint_or_process_exit",
    "write_session_payload",
]
