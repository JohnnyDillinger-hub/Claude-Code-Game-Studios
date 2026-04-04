from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile

from cluster.orchestrator.registry import NodeRegistry


class RegistryStateStore:
    """Simple JSON-backed persistent state for the cluster registry."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> NodeRegistry:
        if not self.exists():
            return NodeRegistry()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return NodeRegistry.from_state_dict(payload)

    def save(self, registry: NodeRegistry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = registry.to_state_dict()
        data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(self.path.parent),
        ) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, self.path)
