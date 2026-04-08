from __future__ import annotations

from cluster.providers.models import ProviderBlueprint


def load_builtin_blueprints() -> tuple[ProviderBlueprint, ...]:
    return (
        ProviderBlueprint(
            blueprint_id="qwen-coder-node-vast",
            name="Qwen Coder Node",
            description="Burst coder node for Qwen3-Coder with template-capable Vast provisioning.",
            provider="vast",
            runtime_family="vllm",
            model_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            cached_models=("Qwen/Qwen3-Coder-30B-A3B-Instruct",),
            labels=(("role", "coder"), ("provider", "vast")),
            trust_tier="burst",
            network_tier="public",
            default_gpu_count=2,
            provider_config_template={
                "image": "vllm/vllm-openai:latest",
                "runtype": "ssh_direct",
                "supports_template": True,
            },
        ),
        ProviderBlueprint(
            blueprint_id="gemma-review-node-runpod",
            name="Gemma Review Node",
            description="Template-driven Runpod review worker for Gemma class models.",
            provider="runpod",
            runtime_family="ollama",
            model_id="google/gemma-3-12b-it",
            cached_models=("google/gemma-3-12b-it",),
            labels=(("role", "review"), ("provider", "runpod")),
            trust_tier="burst",
            network_tier="public",
            default_gpu_count=1,
            provider_config_template={
                "cloudType": "SECURE",
                "templateId": "runpod-template-placeholder",
                "containerDiskInGb": 80,
            },
        ),
        ProviderBlueprint(
            blueprint_id="trusted-nebius-worker",
            name="Trusted Nebius Worker",
            description="Long-lived Nebius worker with cloud-init bootstrap and trusted labels.",
            provider="nebius",
            runtime_family="system",
            cached_models=(),
            labels=(("role", "trusted-worker"), ("provider", "nebius")),
            trust_tier="trusted",
            network_tier="private",
            default_gpu_count=1,
            provider_config_template={
                "platform": "gpu-h100-sxm",
                "preset": "gpu1",
                "supports_cloud_init": True,
                "vm_type": "regular",
            },
        ),
        ProviderBlueprint(
            blueprint_id="custom-gpu-node",
            name="Custom GPU Node",
            description="Bring-your-own provider request with explicit offer or preset selection.",
            provider="any",
            runtime_family="custom",
            cached_models=(),
            labels=(("role", "custom-node"),),
            provider_config_template={},
        ),
    )
