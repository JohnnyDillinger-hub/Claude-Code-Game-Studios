from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib import error

from cluster.orchestrator.runtime_adapters import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_SERVER_HOST,
    DEFAULT_SESSION_DIR,
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
    DEFAULT_VLLM_LAUNCH_MODULE,
    DEFAULT_WARMUP_PROMPT,
    SESSION_OK_STATUSES,
    WorkerError,
    WorkerSession,
    build_ollama_server_command,
    build_vllm_server_command,
    choose_runtime_port,
    expand_path_text,
    find_conflicting_session,
    get_runtime_adapter,
    launch_with_adapter,
    load_session_payload,
    resolve_gpu_indices,
    resolve_launch_mode,
    sanitize_agent_id,
    write_session_payload,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 3 backend-aware remote worker entrypoint."
    )
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--runtime-class", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--gpu-indices")
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
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    return parser


def launch_ollama_worker(args: argparse.Namespace, session_dir: Path) -> WorkerSession:
    return get_runtime_adapter("ollama-server").launch(args, session_dir)


def launch_vllm_worker(args: argparse.Namespace, session_dir: Path) -> WorkerSession:
    return get_runtime_adapter("vllm-server").launch(args, session_dir)


def launch_python_hf_probe(args: argparse.Namespace, session_dir: Path) -> WorkerSession:
    return get_runtime_adapter("python-hf-probe").launch(args, session_dir)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    session_dir = Path(expand_path_text(args.session_dir)).resolve()
    try:
        gpu_indices = resolve_gpu_indices(args)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_indices)
        session = launch_with_adapter(args, session_dir)
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
