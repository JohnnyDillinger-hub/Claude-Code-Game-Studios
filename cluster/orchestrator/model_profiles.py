from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import json
from typing import Any


MODEL_PROFILES_PATH = Path(__file__).with_name("model_profiles.yaml")
COMMON_LAUNCH_METADATA_KEYS = {
    "entrypoint_module",
    "repo_root_default",
    "launch_mode",
    "session_dir",
    "server_host",
    "port_base",
    "startup_timeout_seconds",
    "request_timeout_seconds",
    "warmup_prompt",
    "warmup_keep_alive",
    "max_new_tokens",
    "python_executable",
    "script_path",
}


def infer_runtime_adapter(preferred_backend: str, launch_metadata: dict[str, Any]) -> str:
    launch_mode = launch_metadata.get("launch_mode")
    if launch_mode is not None:
        return str(launch_mode)
    backend = str(preferred_backend)
    if backend == "ollama":
        return "ollama-server"
    if backend == "vllm":
        return "vllm-server"
    if backend == "sglang":
        return "sglang-server"
    if backend == "deepspeed":
        return "deepspeed-server"
    if backend in {"tensorrt-llm", "trtllm"}:
        return "trtllm-server"
    if backend == "python-hf":
        return "python-hf-probe"
    return backend


def normalize_runtime_options(
    launch_metadata: dict[str, Any],
    explicit_runtime_options: Any,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    embedded_options = launch_metadata.get("runtime_options")
    if isinstance(embedded_options, dict):
        normalized.update(embedded_options)
    if isinstance(explicit_runtime_options, dict):
        normalized.update(explicit_runtime_options)
    for key, value in launch_metadata.items():
        if key in COMMON_LAUNCH_METADATA_KEYS or key in {"runtime_adapter", "runtime_options"}:
            continue
        normalized.setdefault(key, value)
    return normalized


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    name: str
    model_name: str
    required_free_vram_mib: int
    required_gpu_count: int
    runtime_class: str
    preferred_backend: str
    runtime_adapter: str
    runtime_options: dict[str, Any]
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
        launch_metadata = dict(payload["launch"])
        runtime_adapter = str(
            payload.get("runtime_adapter")
            or launch_metadata.get("runtime_adapter")
            or infer_runtime_adapter(str(payload["preferred_backend"]), launch_metadata)
        )
        runtime_options = normalize_runtime_options(launch_metadata, payload.get("runtime_options"))
        launch_metadata["runtime_adapter"] = runtime_adapter
        launch_metadata["runtime_options"] = dict(runtime_options)
        return cls(
            name=name,
            model_name=str(payload["model_name"]),
            required_free_vram_mib=int(payload["required_free_vram_mib"]),
            required_gpu_count=int(payload.get("required_gpu_count", 1)),
            runtime_class=str(payload["runtime_class"]),
            preferred_backend=str(payload["preferred_backend"]),
            runtime_adapter=runtime_adapter,
            runtime_options=runtime_options,
            launch_metadata=launch_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_name": self.model_name,
            "required_free_vram_mib": self.required_free_vram_mib,
            "required_gpu_count": self.required_gpu_count,
            "runtime_class": self.runtime_class,
            "preferred_backend": self.preferred_backend,
            "runtime_adapter": self.runtime_adapter,
            "runtime_options": dict(self.runtime_options),
            "launch": dict(self.launch_metadata),
        }

    def with_launch_overrides(self, overrides: dict[str, Any]) -> "RuntimeProfile":
        if not overrides:
            return self
        launch_metadata = dict(self.launch_metadata)
        runtime_options = dict(self.runtime_options)
        for key, value in overrides.items():
            if value is None:
                launch_metadata.pop(key, None)
                runtime_options.pop(key, None)
            else:
                launch_metadata[key] = value
                runtime_options[key] = value
        launch_metadata["runtime_adapter"] = self.runtime_adapter
        launch_metadata["runtime_options"] = dict(runtime_options)
        return replace(
            self,
            runtime_options=runtime_options,
            launch_metadata=launch_metadata,
        )


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
