from __future__ import annotations

import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
from typing import Any
from urllib.parse import urlparse

from cluster.models import NodeInventory, format_datetime, parse_datetime, utc_now
from cluster.node_agent.heartbeat import (
    PeriodicHeartbeatService,
    apply_heartbeat_to_state_file,
    post_heartbeat,
)
from cluster.node_agent.probe_gpu import ProbeError, build_local_inventory, parse_label_items


@dataclass(frozen=True, slots=True)
class NodeAgentConfig:
    node_id: str
    host: str
    bind_host: str
    port: int
    available_until: str | None
    lease_duration_seconds: int | None
    cached_models: tuple[str, ...]
    labels: dict[str, str]
    trust_tier: str | None
    network_tier: str | None
    allow_empty: bool
    heartbeat_interval_seconds: int | None
    heartbeat_url: str | None
    heartbeat_state_file: str | None
    heartbeat_timeout_seconds: int

    @classmethod
    def from_sources(cls, args: argparse.Namespace) -> "NodeAgentConfig":
        raw_config: dict[str, Any] = {}
        if getattr(args, "config", None):
            raw_config = json.loads(Path(args.config).read_text(encoding="utf-8"))

        def choose(name: str, fallback: Any) -> Any:
            value = getattr(args, name, None)
            if isinstance(value, bool):
                return value if value else raw_config.get(name, fallback)
            if value not in (None, [], ()):
                return value
            return raw_config.get(name, fallback)

        cached_models = choose("cached_model", raw_config.get("cached_models", []))
        labels_value = choose("label", raw_config.get("labels", []))
        labels = (
            labels_value
            if isinstance(labels_value, dict)
            else parse_label_items(list(labels_value))
        )

        return cls(
            node_id=choose("node_id", "local-node"),
            host=choose("host", socket.gethostname()),
            bind_host=choose("bind_host", "127.0.0.1"),
            port=int(choose("port", 8787)),
            available_until=choose("available_until", None),
            lease_duration_seconds=int(choose("lease_duration_seconds", 3600)),
            cached_models=tuple(cached_models),
            labels=dict(labels),
            trust_tier=choose("trust_tier", None),
            network_tier=choose("network_tier", None),
            allow_empty=not bool(choose("no_allow_empty", False)),
            heartbeat_interval_seconds=choose("heartbeat_interval_seconds", None),
            heartbeat_url=choose("heartbeat_url", None),
            heartbeat_state_file=choose("heartbeat_state_file", None),
            heartbeat_timeout_seconds=int(choose("heartbeat_timeout_seconds", 5)),
        )

    def current_inventory(self) -> NodeInventory:
        available_until = (
            parse_datetime(self.available_until)
            if self.available_until is not None
            else None
        )
        return build_local_inventory(
            node_id=self.node_id,
            host=self.host,
            available_until=available_until,
            lease_duration_seconds=self.lease_duration_seconds,
            cached_models=self.cached_models,
            labels=self.labels,
            trust_tier=self.trust_tier,
            network_tier=self.network_tier,
            allow_empty=self.allow_empty,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 2 node inventory daemon.")
    parser.add_argument("--config", help="Optional JSON config file for the node agent.")
    parser.add_argument("--node-id", help="Stable node identifier.")
    parser.add_argument("--host", help="Advertised host or address for the node.")
    parser.add_argument("--bind-host", help="Interface to bind the local HTTP server to.")
    parser.add_argument("--port", type=int, help="HTTP port for the node agent.")
    parser.add_argument("--available-until", help="Absolute ISO-8601 timestamp for lease end.")
    parser.add_argument(
        "--lease-duration-seconds",
        type=int,
        help="Relative lease duration when --available-until is omitted.",
    )
    parser.add_argument("--cached-model", action="append", default=[], help="Repeat cached model ids.")
    parser.add_argument("--label", action="append", default=[], help="Repeat key=value labels.")
    parser.add_argument("--trust-tier", help="Optional trust classification.")
    parser.add_argument("--network-tier", help="Optional network classification.")
    parser.add_argument(
        "--no-allow-empty",
        action="store_true",
        help="Fail inventory requests if the machine has no GPUs.",
    )
    parser.add_argument(
        "--heartbeat-interval-seconds",
        type=int,
        help="If set, emit periodic heartbeats using the configured destination.",
    )
    parser.add_argument(
        "--heartbeat-url",
        help="Optional orchestrator heartbeat endpoint. Example: http://127.0.0.1:8790/heartbeat",
    )
    parser.add_argument(
        "--heartbeat-state-file",
        help="Optional local registry state file for Phase 2 demo heartbeats.",
    )
    parser.add_argument(
        "--heartbeat-timeout-seconds",
        type=int,
        help="Timeout for heartbeat HTTP POST requests.",
    )
    return parser


class NodeAgentHandler(BaseHTTPRequestHandler):
    server_version = "ClusterNodeAgent/0.2"

    def _json_response(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _inventory_payload(self) -> dict[str, Any]:
        config: NodeAgentConfig = self.server.node_agent_config  # type: ignore[attr-defined]
        return config.current_inventory().to_dict()

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            config: NodeAgentConfig = self.server.node_agent_config  # type: ignore[attr-defined]
            payload = {
                "status": "ok",
                "node_id": config.node_id,
                "time": format_datetime(utc_now()),
                "heartbeat_enabled": bool(config.heartbeat_interval_seconds),
            }
            self._json_response(200, payload)
            return
        if parsed.path == "/inventory":
            try:
                self._json_response(200, self._inventory_payload())
            except ProbeError as exc:
                self._json_response(503, {"status": "error", "reason": str(exc)})
            return
        self._json_response(404, {"status": "not_found", "path": parsed.path})

    def log_message(self, fmt: str, *args: object) -> None:
        return


def _build_heartbeat_sender(config: NodeAgentConfig):
    if config.heartbeat_url:
        return lambda payload: post_heartbeat(
            config.heartbeat_url,
            payload,
            timeout_seconds=config.heartbeat_timeout_seconds,
        )
    if config.heartbeat_state_file:
        return lambda payload: apply_heartbeat_to_state_file(
            config.heartbeat_state_file,
            payload,
        )
    return None


def serve(config: NodeAgentConfig) -> int:
    server = ThreadingHTTPServer((config.bind_host, config.port), NodeAgentHandler)
    server.node_agent_config = config  # type: ignore[attr-defined]

    heartbeat_service = None
    sender = _build_heartbeat_sender(config)
    if config.heartbeat_interval_seconds and sender is not None:
        heartbeat_service = PeriodicHeartbeatService(
            interval_seconds=config.heartbeat_interval_seconds,
            inventory_builder=config.current_inventory,
            on_heartbeat=sender,
        )
        heartbeat_service.start()

    print(
        f"node-agent listening on http://{config.bind_host}:{config.port} "
        f"for node_id={config.node_id}"
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if heartbeat_service is not None:
            heartbeat_service.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return serve(NodeAgentConfig.from_sources(args))


if __name__ == "__main__":
    raise SystemExit(main())
