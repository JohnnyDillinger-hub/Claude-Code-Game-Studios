from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import signal
from pathlib import Path
from typing import Any, Sequence

from cluster.orchestrator.remote_worker import (
    DEFAULT_SESSION_DIR,
    SESSION_OK_STATUSES,
    expand_path_text,
    load_session_payload,
    sanitize_agent_id,
    write_session_payload,
)


@dataclass(frozen=True, slots=True)
class RemoteSessionClaim:
    agent_id: str
    node_id: str
    gpu_index: int
    status: str
    backend: str | None = None
    model: str | None = None
    listen_port: int | None = None
    session_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "agent_id": self.agent_id,
            "node_id": self.node_id,
            "gpu_index": self.gpu_index,
            "status": self.status,
        }
        if self.backend is not None:
            payload["backend"] = self.backend
        if self.model is not None:
            payload["model"] = self.model
        if self.listen_port is not None:
            payload["listen_port"] = self.listen_port
        if self.session_file is not None:
            payload["session_file"] = self.session_file
        return payload


def list_session_payloads(session_dir: Path) -> list[dict[str, Any]]:
    if not session_dir.exists():
        return []
    payloads: list[dict[str, Any]] = []
    for candidate in sorted(session_dir.glob("*.json")):
        payload = load_session_payload(candidate)
        if payload is not None:
            payloads.append(payload)
    return payloads


def _payload_gpu_indices(payload: dict[str, Any]) -> tuple[int, ...]:
    raw_gpu_indices = payload.get("gpu_indices")
    if isinstance(raw_gpu_indices, list) and raw_gpu_indices:
        try:
            return tuple(int(item) for item in raw_gpu_indices)
        except (TypeError, ValueError):
            return ()
    gpu_value = payload.get("gpu_index")
    if gpu_value is None:
        return ()
    try:
        return (int(gpu_value),)
    except (TypeError, ValueError):
        return ()


def payload_to_claim(payload: dict[str, Any]) -> RemoteSessionClaim | None:
    status = str(payload.get("status") or "")
    if status not in SESSION_OK_STATUSES:
        return None
    agent_id = str(payload.get("agent_id") or "").strip()
    node_id = str(payload.get("node_id") or "").strip()
    if not agent_id or not node_id:
        return None
    gpu_indices = _payload_gpu_indices(payload)
    if not gpu_indices:
        return None
    listen_port = payload.get("listen_port")
    try:
        listen_port_value = int(listen_port) if listen_port is not None else None
    except (TypeError, ValueError):
        listen_port_value = None
    return RemoteSessionClaim(
        agent_id=agent_id,
        node_id=node_id,
        gpu_index=gpu_indices[0],
        status=status,
        backend=str(payload.get("backend")) if payload.get("backend") is not None else None,
        model=str(payload.get("model")) if payload.get("model") is not None else None,
        listen_port=listen_port_value,
        session_file=(
            str(payload.get("session_file")) if payload.get("session_file") is not None else None
        ),
    )


def collect_session_claims(session_dir: Path) -> list[RemoteSessionClaim]:
    claims: list[RemoteSessionClaim] = []
    for payload in list_session_payloads(session_dir):
        template = payload_to_claim(payload)
        if template is None:
            continue
        gpu_indices = _payload_gpu_indices(payload)
        for gpu_index in gpu_indices:
            claims.append(
                RemoteSessionClaim(
                    agent_id=template.agent_id,
                    node_id=template.node_id,
                    gpu_index=gpu_index,
                    status=template.status,
                    backend=template.backend,
                    model=template.model,
                    listen_port=template.listen_port,
                    session_file=template.session_file,
                )
            )
    return claims


def stop_session(session_dir: Path, agent_id: str) -> dict[str, Any]:
    session_path = session_dir / f"{sanitize_agent_id(agent_id)}.json"
    payload = load_session_payload(session_path)
    if payload is None:
        return {
            "status": "not-found",
            "agent_id": agent_id,
            "session_file": str(session_path),
        }
    server_pid = payload.get("server_pid")
    stopped_pid: int | None = None
    if server_pid is not None:
        try:
            stopped_pid = int(server_pid)
            os.kill(stopped_pid, signal.SIGTERM)
        except (OSError, TypeError, ValueError):
            stopped_pid = None
    updated = dict(payload)
    updated["status"] = "stopped"
    updated["reused"] = False
    notes = str(updated.get("notes") or "").strip()
    updated["notes"] = (
        f"{notes} Session marked stopped via remote session control.".strip()
        if notes
        else "Session marked stopped via remote session control."
    )
    write_session_payload(session_path, updated)
    result = {
        "status": "stopped",
        "agent_id": agent_id,
        "session_file": str(session_path),
        "payload": updated,
    }
    if stopped_pid is not None:
        result["server_pid"] = stopped_pid
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect or control remote worker sessions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List session files and active GPU claims.")
    list_parser.add_argument("--session-dir", default=DEFAULT_SESSION_DIR)

    stop_parser = subparsers.add_parser("stop", help="Stop a session by agent id.")
    stop_parser.add_argument("--session-dir", default=DEFAULT_SESSION_DIR)
    stop_parser.add_argument("--agent-id", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    session_dir = Path(expand_path_text(args.session_dir)).resolve()

    if args.command == "list":
        payloads = list_session_payloads(session_dir)
        claims = [claim.to_dict() for claim in collect_session_claims(session_dir)]
        print(
            json.dumps(
                {
                    "session_dir": str(session_dir),
                    "session_count": len(payloads),
                    "sessions": payloads,
                    "claims": claims,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "stop":
        print(json.dumps(stop_session(session_dir, args.agent_id), sort_keys=True))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
