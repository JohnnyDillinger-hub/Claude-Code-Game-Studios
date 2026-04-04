from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
from typing import Any, Iterable

from cluster.models import (
    NodeInventory,
    format_datetime,
    load_node_inventory_file,
    parse_datetime,
    utc_now,
)


@dataclass(frozen=True, slots=True)
class RegistryRecord:
    node: NodeInventory
    last_heartbeat_at: datetime | None = None
    heartbeat_interval_seconds: int | None = None
    source: str = "manual"

    def is_stale(
        self,
        *,
        now: datetime,
        stale_after_seconds: int | None = None,
    ) -> bool:
        if self.last_heartbeat_at is None:
            return False
        if stale_after_seconds is None:
            return False
        age_seconds = (now - self.last_heartbeat_at).total_seconds()
        return age_seconds > stale_after_seconds

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "node": self.node.to_dict(),
            "source": self.source,
        }
        if self.last_heartbeat_at is not None:
            payload["last_heartbeat_at"] = format_datetime(self.last_heartbeat_at)
        if self.heartbeat_interval_seconds is not None:
            payload["heartbeat_interval_seconds"] = self.heartbeat_interval_seconds
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RegistryRecord":
        return cls(
            node=NodeInventory.from_dict(payload["node"]),
            last_heartbeat_at=(
                parse_datetime(payload["last_heartbeat_at"])
                if payload.get("last_heartbeat_at")
                else None
            ),
            heartbeat_interval_seconds=(
                int(payload["heartbeat_interval_seconds"])
                if payload.get("heartbeat_interval_seconds") is not None
                else None
            ),
            source=str(payload.get("source", "manual")),
        )


class NodeRegistry:
    def __init__(self, nodes: Iterable[NodeInventory] | None = None) -> None:
        self._records: dict[str, RegistryRecord] = {}
        for node in nodes or ():
            self.register(node)

    def register(
        self,
        node_inventory: NodeInventory,
        *,
        last_heartbeat_at: datetime | None = None,
        heartbeat_interval_seconds: int | None = None,
        source: str = "manual",
    ) -> None:
        self._records[node_inventory.node_id] = RegistryRecord(
            node=node_inventory,
            last_heartbeat_at=last_heartbeat_at,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            source=source,
        )

    def register_heartbeat(
        self,
        node_inventory: NodeInventory,
        *,
        received_at: datetime,
        heartbeat_interval_seconds: int | None = None,
        source: str = "heartbeat",
    ) -> None:
        self.register(
            node_inventory,
            last_heartbeat_at=received_at,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            source=source,
        )

    def remove(self, node_id: str) -> NodeInventory | None:
        record = self._records.pop(node_id, None)
        return None if record is None else record.node

    def get_record(self, node_id: str) -> RegistryRecord | None:
        return self._records.get(node_id)

    def list_records(self) -> list[RegistryRecord]:
        return [self._records[node_id] for node_id in sorted(self._records)]

    def list_nodes(self) -> list[NodeInventory]:
        return [record.node for record in self.list_records()]

    def active_nodes(
        self,
        *,
        now: datetime | None = None,
        stale_after_seconds: int | None = None,
    ) -> list[NodeInventory]:
        current_time = now or utc_now()
        active: list[NodeInventory] = []
        for record in self.list_records():
            if record.node.is_expired(now=current_time):
                continue
            if record.is_stale(now=current_time, stale_after_seconds=stale_after_seconds):
                continue
            active.append(record.node)
        return active

    def prune_expired(
        self,
        *,
        now: datetime | None = None,
        stale_after_seconds: int | None = None,
    ) -> list[str]:
        current_time = now or utc_now()
        removed: list[str] = []
        for record in self.list_records():
            if record.node.is_expired(now=current_time) or record.is_stale(
                now=current_time,
                stale_after_seconds=stale_after_seconds,
            ):
                removed.append(record.node.node_id)
                self._records.pop(record.node.node_id, None)
        return removed

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "records": [record.to_dict() for record in self.list_records()],
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> "NodeRegistry":
        records = payload.get("records", [])
        registry = cls()
        for item in records:
            record = RegistryRecord.from_dict(item)
            registry.register(
                record.node,
                last_heartbeat_at=record.last_heartbeat_at,
                heartbeat_interval_seconds=record.heartbeat_interval_seconds,
                source=record.source,
            )
        return registry


def load_registry_file(path: str | Path) -> NodeRegistry:
    raw_text = Path(path).read_text(encoding="utf-8")
    payload = json.loads(raw_text)
    if isinstance(payload, dict) and "records" in payload:
        return NodeRegistry.from_state_dict(payload)
    return NodeRegistry(load_node_inventory_file(path))


def dump_registry_file(path: str | Path, nodes: Iterable[NodeInventory]) -> None:
    payload = [node.to_dict() for node in nodes]
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
