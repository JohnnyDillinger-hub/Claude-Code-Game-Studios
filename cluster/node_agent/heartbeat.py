from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import threading
from typing import Any, Callable
from urllib import error, request

from cluster.models import NodeInventory, format_datetime, parse_datetime, utc_now
from cluster.orchestrator.state_store import RegistryStateStore


@dataclass(frozen=True, slots=True)
class HeartbeatPayload:
    inventory: NodeInventory
    sent_at: datetime
    heartbeat_interval_seconds: int
    source: str = "node-agent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "inventory": self.inventory.to_dict(),
            "sent_at": format_datetime(self.sent_at),
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HeartbeatPayload":
        return cls(
            inventory=NodeInventory.from_dict(payload["inventory"]),
            sent_at=parse_datetime(payload["sent_at"]),
            heartbeat_interval_seconds=int(payload["heartbeat_interval_seconds"]),
            source=str(payload.get("source", "node-agent")),
        )


def build_heartbeat_payload(
    inventory: NodeInventory,
    *,
    heartbeat_interval_seconds: int,
    sent_at: datetime | None = None,
    source: str = "node-agent",
) -> HeartbeatPayload:
    return HeartbeatPayload(
        inventory=inventory,
        sent_at=sent_at or utc_now(),
        heartbeat_interval_seconds=heartbeat_interval_seconds,
        source=source,
    )


def fetch_inventory_from_url(url: str, timeout_seconds: int = 5) -> NodeInventory:
    with request.urlopen(url, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return NodeInventory.from_dict(payload)


def post_heartbeat(url: str, payload: HeartbeatPayload, timeout_seconds: int = 5) -> dict[str, Any]:
    body = json.dumps(payload.to_dict()).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        raw = response.read()
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))


def apply_heartbeat_to_state_file(path: str, payload: HeartbeatPayload) -> dict[str, Any]:
    store = RegistryStateStore(path)
    registry = store.load()
    registry.register_heartbeat(
        payload.inventory,
        received_at=payload.sent_at,
        heartbeat_interval_seconds=payload.heartbeat_interval_seconds,
        source=payload.source,
    )
    store.save(registry)
    return {
        "status": "ok",
        "node_id": payload.inventory.node_id,
        "record_count": len(registry.list_nodes()),
    }


class PeriodicHeartbeatService:
    def __init__(
        self,
        *,
        interval_seconds: int,
        inventory_builder: Callable[[], NodeInventory],
        on_heartbeat: Callable[[HeartbeatPayload], dict[str, Any] | None],
        source: str = "node-agent",
    ) -> None:
        self.interval_seconds = interval_seconds
        self.inventory_builder = inventory_builder
        self.on_heartbeat = on_heartbeat
        self.source = source
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=self.interval_seconds + 1)

    def heartbeat_once(self) -> dict[str, Any] | None:
        inventory = self.inventory_builder()
        payload = build_heartbeat_payload(
            inventory,
            heartbeat_interval_seconds=self.interval_seconds,
            source=self.source,
        )
        return self.on_heartbeat(payload)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.heartbeat_once()
            except (OSError, ValueError, error.URLError):
                # Phase 2 keeps heartbeat failure handling simple and non-fatal.
                pass
            self._stop_event.wait(self.interval_seconds)
