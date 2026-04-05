from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable
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
    endpoint_url: str | None = None
    listen_port: int | None = None
    server_pid: int | None = None
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
            "single_gpu_only": self.single_gpu_only,
            "launched_at": self.launched_at,
            "session_file": self.session_file,
            "reused": self.reused,
        }
        if self.endpoint_url is not None:
            payload["endpoint_url"] = self.endpoint_url
        if self.listen_port is not None:
            payload["listen_port"] = self.listen_port
        if self.server_pid is not None:
            payload["server_pid"] = self.server_pid
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


def expand_path_text(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))


def choose_runtime_port(port_base: int, gpu_index: int) -> int:
    return int(port_base) + int(gpu_index)


def sanitize_agent_id(agent_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", agent_id)


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
    return command


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


def find_conflicting_session(
    session_dir: Path,
    *,
    agent_id: str,
    node_id: str,
    gpu_index: int,
    listen_port: int | None,
) -> dict[str, Any] | None:
    if not session_dir.exists():
        return None
    for candidate in sorted(session_dir.glob("*.json")):
        payload = load_session_payload(candidate)
        if payload is None:
            continue
        if payload.get("agent_id") == agent_id:
            continue
        if payload.get("node_id") != node_id:
            continue
        if int(payload.get("gpu_index", -1)) != gpu_index:
            continue
        if listen_port is not None and payload.get("listen_port") not in (None, listen_port):
            continue
        if str(payload.get("status")) not in SESSION_OK_STATUSES:
            continue
        return payload
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 3 backend-aware remote worker entrypoint."
    )
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--runtime-class", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument(
        "--launch-mode",
        default="auto",
        choices=("auto", "ollama-server", "vllm-server", "python-hf-probe"),
    )
    parser.add_argument("--session-dir", default=DEFAULT_SESSION_DIR)
    parser.add_argument("--server-host", default=DEFAULT_SERVER_HOST)
    parser.add_argument("--port-base", type=int)
    parser.add_argument(
        "--startup-timeout-seconds",
        type=int,
        default=DEFAULT_STARTUP_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument("--warmup-prompt", default=DEFAULT_WARMUP_PROMPT)
    parser.add_argument("--warmup-keep-alive", default="15m")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--python-executable", default="python3")
    parser.add_argument("--script-path")
    parser.add_argument("--vllm-launch-module", default=DEFAULT_VLLM_LAUNCH_MODULE)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    return parser


def resolve_launch_mode(args: argparse.Namespace) -> str:
    if args.launch_mode != "auto":
        return str(args.launch_mode)
    backend = str(args.backend)
    if backend == "ollama":
        return "ollama-server"
    if backend == "vllm":
        return "vllm-server"
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


def launch_ollama_worker(args: argparse.Namespace, session_dir: Path) -> WorkerSession:
    if shutil.which("ollama") is None:
        raise WorkerError("ollama is not installed on the target node")

    port_base = args.port_base or DEFAULT_OLLAMA_PORT_BASE
    port = choose_runtime_port(port_base, args.gpu_index)
    conflict = find_conflicting_session(
        session_dir,
        agent_id=args.agent_id,
        node_id=args.node_id,
        gpu_index=args.gpu_index,
        listen_port=port,
    )
    if conflict is not None:
        raise WorkerError(
            f"GPU {args.gpu_index} on {args.node_id} already has an active session for "
            f"agent {conflict.get('agent_id')!r}"
        )

    session_path, stdout_log, stderr_log = resolve_session_paths(session_dir, args.agent_id)
    endpoint_url = f"http://{args.server_host}:{port}"
    health_url = f"{endpoint_url}/api/tags"
    existing = load_session_payload(session_path)
    if (
        existing is not None
        and existing.get("listen_port") == port
        and existing.get("endpoint_url") == endpoint_url
        and str(existing.get("status")) in SESSION_OK_STATUSES
    ):
        try:
            wait_for_json_endpoint(
                [health_url],
                startup_timeout_seconds=3,
                request_timeout_seconds=2,
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

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    env["OLLAMA_HOST"] = f"{args.server_host}:{port}"

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
    wait_for_json_endpoint(
        [health_url],
        startup_timeout_seconds=args.startup_timeout_seconds,
        request_timeout_seconds=min(args.request_timeout_seconds, 10),
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
        single_gpu_only=True,
        launched_at=utc_timestamp(),
        session_file=str(session_path),
        endpoint_url=endpoint_url,
        listen_port=port,
        server_pid=server_pid,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        warmup_response=response_text,
        command=build_ollama_server_command(),
        health_url=health_url,
        notes="Dedicated Ollama server launched and the target model was warmed.",
    )
    write_session_payload(session_path, session.to_dict())
    return session


def launch_vllm_worker(args: argparse.Namespace, session_dir: Path) -> WorkerSession:
    if args.tensor_parallel_size != 1:
        raise WorkerError("Phase 3 keeps tensor_parallel_size fixed at 1")

    python_executable = expand_path_text(args.python_executable)
    port_base = args.port_base or DEFAULT_VLLM_PORT_BASE
    port = choose_runtime_port(port_base, args.gpu_index)
    conflict = find_conflicting_session(
        session_dir,
        agent_id=args.agent_id,
        node_id=args.node_id,
        gpu_index=args.gpu_index,
        listen_port=port,
    )
    if conflict is not None:
        raise WorkerError(
            f"GPU {args.gpu_index} on {args.node_id} already has an active session for "
            f"agent {conflict.get('agent_id')!r}"
        )

    session_path, stdout_log, stderr_log = resolve_session_paths(session_dir, args.agent_id)
    endpoint_url = f"http://{args.server_host}:{port}"
    health_candidates = [f"{endpoint_url}/health", f"{endpoint_url}/v1/models"]
    existing = load_session_payload(session_path)
    if (
        existing is not None
        and existing.get("listen_port") == port
        and existing.get("endpoint_url") == endpoint_url
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
                command=build_vllm_server_command(
                    python_executable=python_executable,
                    launch_module=args.vllm_launch_module,
                    host=args.server_host,
                    port=port,
                    model=args.model,
                    tensor_parallel_size=args.tensor_parallel_size,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.max_model_len,
                ),
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
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    command = build_vllm_server_command(
        python_executable=python_executable,
        launch_module=args.vllm_launch_module,
        host=args.server_host,
        port=port,
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
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
            single_gpu_only=True,
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
        single_gpu_only=True,
        launched_at=utc_timestamp(),
        session_file=str(session_path),
        endpoint_url=endpoint_url,
        listen_port=port,
        server_pid=server_pid,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        command=command,
        health_url=ready_url,
        notes="Dedicated vLLM server launched and passed a health probe.",
    )
    write_session_payload(session_path, session.to_dict())
    return session


def launch_python_hf_probe(args: argparse.Namespace, session_dir: Path) -> WorkerSession:
    python_executable = expand_path_text(args.python_executable)
    script_path = expand_path_text(args.script_path or "scripts/gemma_pt/infer_gemma_pt.py")
    session_path, stdout_log, stderr_log = resolve_session_paths(session_dir, args.agent_id)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    session_dir = Path(expand_path_text(args.session_dir)).resolve()
    try:
        launch_mode = resolve_launch_mode(args)
        if launch_mode == "ollama-server":
            session = launch_ollama_worker(args, session_dir)
        elif launch_mode == "vllm-server":
            session = launch_vllm_worker(args, session_dir)
        elif launch_mode == "python-hf-probe":
            session = launch_python_hf_probe(args, session_dir)
        else:
            raise WorkerError(f"Unsupported launch mode {launch_mode!r}")
    except subprocess.CalledProcessError as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "agent_id": args.agent_id,
                    "node_id": args.node_id,
                    "backend": args.backend,
                    "reason": f"Worker command failed with exit code {exc.returncode}",
                    "stdout": (exc.stdout or "").strip() or None,
                    "stderr": (exc.stderr or "").strip() or None,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    except (WorkerError, OSError, TimeoutError, error.URLError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "agent_id": args.agent_id,
                    "node_id": args.node_id,
                    "backend": args.backend,
                    "reason": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    print(json.dumps(session.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
