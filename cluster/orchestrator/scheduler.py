from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from cluster.models import AgentRequest, GPUInventory, NodeInventory, PlacementDecision


@dataclass(frozen=True, slots=True)
class _Candidate:
    node: NodeInventory
    gpu: GPUInventory
    model_cached: bool
    lease_remaining_seconds: float


def _candidate_sort_key(candidate: _Candidate) -> tuple[float, float, float, str, int]:
    return (
        -float(candidate.model_cached),
        -candidate.lease_remaining_seconds,
        -float(candidate.gpu.free_memory_mib),
        candidate.node.node_id,
        candidate.gpu.index,
    )


def _matching_labels(request: AgentRequest, node: NodeInventory) -> bool:
    request_labels = request.labels_map
    if not request_labels:
        return True
    node_labels = node.labels_map
    return all(node_labels.get(key) == value for key, value in request_labels.items())


def _candidate_gpus(
    nodes: Iterable[NodeInventory],
    request: AgentRequest,
    now: datetime,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for node in nodes:
        if node.is_expired(now=now):
            continue
        if request.trust_tier and node.trust_tier != request.trust_tier:
            continue
        if request.network_tier and node.network_tier != request.network_tier:
            continue
        if not _matching_labels(request, node):
            continue
        cached = bool(request.model_id and request.model_id in node.cached_models)
        for gpu in node.gpus:
            if gpu.free_memory_mib >= request.required_vram_mib:
                candidates.append(
                    _Candidate(
                        node=node,
                        gpu=gpu,
                        model_cached=cached,
                        lease_remaining_seconds=node.lease.remaining_seconds(now=now),
                    )
                )
    return sorted(candidates, key=_candidate_sort_key)


def _placed_decision(request: AgentRequest, candidate: _Candidate, source: str) -> PlacementDecision:
    cache_note = "cached model" if candidate.model_cached else "cold model"
    reason = (
        f"Placed on {source} node {candidate.node.node_id} gpu {candidate.gpu.index} "
        f"with {candidate.gpu.free_memory_mib} MiB free VRAM ({cache_note})."
    )
    return PlacementDecision(
        status="placed",
        reason=reason,
        agent_id=request.agent_id,
        node_id=candidate.node.node_id,
        host=candidate.node.host,
        gpu_index=candidate.gpu.index,
        gpu_uuid=candidate.gpu.uuid,
        source=source,
        model_cached=candidate.model_cached,
        required_vram_mib=request.required_vram_mib,
        available_vram_mib=candidate.gpu.free_memory_mib,
        available_until=candidate.node.available_until,
    )


def schedule_agent(
    local_node: NodeInventory,
    remote_nodes: Iterable[NodeInventory],
    request: AgentRequest,
    *,
    now: datetime | None = None,
) -> PlacementDecision:
    current_time = now or datetime.now(tz=local_node.available_until.tzinfo)

    local_candidates = _candidate_gpus([local_node], request, current_time)
    if local_candidates:
        return _placed_decision(request, local_candidates[0], source="local")

    remote_candidates = _candidate_gpus(remote_nodes, request, current_time)
    if remote_candidates:
        return _placed_decision(request, remote_candidates[0], source="remote")

    return PlacementDecision(
        status="rejected",
        reason=(
            f"No non-expired single GPU has at least {request.required_vram_mib} MiB "
            "of free VRAM for this Phase 1 placement request."
        ),
        agent_id=request.agent_id,
        required_vram_mib=request.required_vram_mib,
    )


def reserve_gpu_capacity(
    node: NodeInventory,
    *,
    gpu_index: int,
    amount_mib: int,
) -> NodeInventory:
    updated: list[GPUInventory] = []
    found = False
    for gpu in node.gpus:
        if gpu.index != gpu_index:
            updated.append(gpu)
            continue
        found = True
        if gpu.free_memory_mib < amount_mib:
            raise ValueError(
                f"GPU {gpu_index} on {node.node_id} has only {gpu.free_memory_mib} MiB free"
            )
        updated.append(
            GPUInventory(
                index=gpu.index,
                uuid=gpu.uuid,
                total_memory_mib=gpu.total_memory_mib,
                free_memory_mib=gpu.free_memory_mib - amount_mib,
                utilization_pct=gpu.utilization_pct,
            )
        )
    if not found:
        raise ValueError(f"GPU {gpu_index} does not exist on node {node.node_id}")
    return node.with_gpus(updated)
