from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


MODEL_PROFILES_PATH = Path(__file__).with_name("model_profiles.yaml")


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    name: str
    model_name: str
    required_free_vram_mib: int
    runtime_class: str
    preferred_backend: str
    launch_metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, name: str, payload: dict[str, Any]) -> "RuntimeProfile":
        required_keys = {
            "model_name",
            "required_free_vram_mib",
            "runtime_class",
            "preferred_backend",
            "launch",
        }
        missing = required_keys - payload.keys()
        if missing:
            raise ValueError(f"Profile {name} is missing keys: {sorted(missing)}")
        return cls(
            name=name,
            model_name=str(payload["model_name"]),
            required_free_vram_mib=int(payload["required_free_vram_mib"]),
            runtime_class=str(payload["runtime_class"]),
            preferred_backend=str(payload["preferred_backend"]),
            launch_metadata=dict(payload["launch"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_name": self.model_name,
            "required_free_vram_mib": self.required_free_vram_mib,
            "runtime_class": self.runtime_class,
            "preferred_backend": self.preferred_backend,
            "launch": dict(self.launch_metadata),
        }


def load_runtime_profiles(path: str | Path | None = None) -> dict[str, RuntimeProfile]:
    raw = Path(path or MODEL_PROFILES_PATH).read_text(encoding="utf-8")
    payload = json.loads(raw)
    profiles = payload.get("profiles", payload)
    if not isinstance(profiles, dict):
        raise ValueError("model profile file must contain a mapping of profiles")
    return {
        name: RuntimeProfile.from_dict(name, dict(profile_payload))
        for name, profile_payload in sorted(profiles.items())
    }


def get_runtime_profile(name: str, path: str | Path | None = None) -> RuntimeProfile:
    profiles = load_runtime_profiles(path)
    if name not in profiles:
        raise KeyError(f"Unknown runtime profile: {name}")
    return profiles[name]
