from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from typing import Iterable

from cluster.models import AgentRequest, GPUInventory, NodeInventory, PlacementDecision


@dataclass(frozen=True, slots=True)
class _Candidate:
    node: NodeInventory
    gpus: tuple[GPUInventory, ...]
    model_cached: bool
    lease_remaining_seconds: float

    @property
    def gpu_indices(self) -> tuple[int, ...]:
        return tuple(gpu.index for gpu in self.gpus)

    @property
    def gpu_uuids(self) -> tuple[str, ...]:
        return tuple(gpu.uuid for gpu in self.gpus if gpu.uuid is not None)

    @property
    def min_free_memory_mib(self) -> int:
        return min(gpu.free_memory_mib for gpu in self.gpus)

    @property
    def total_free_memory_mib(self) -> int:
        return sum(gpu.free_memory_mib for gpu in self.gpus)


def _candidate_sort_key(
    candidate: _Candidate,
) -> tuple[float, float, float, float, str, tuple[int, ...]]:
    return (
        -float(candidate.model_cached),
        -candidate.lease_remaining_seconds,
        -float(candidate.min_free_memory_mib),
        -float(candidate.total_free_memory_mib),
        candidate.node.node_id,
        candidate.gpu_indices,
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
    required_gpu_count = max(int(request.required_gpu_count), 1)
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
        eligible_gpus = tuple(
            gpu for gpu in node.gpus if gpu.free_memory_mib >= request.required_vram_mib
        )
        if len(eligible_gpus) < required_gpu_count:
            continue
        for gpu_group in combinations(eligible_gpus, required_gpu_count):
            candidates.append(
                _Candidate(
                    node=node,
                    gpus=tuple(gpu_group),
                    model_cached=cached,
                    lease_remaining_seconds=node.lease.remaining_seconds(now=now),
                )
            )
    return sorted(candidates, key=_candidate_sort_key)


def _placed_decision(request: AgentRequest, candidate: _Candidate, source: str) -> PlacementDecision:
    cache_note = "cached model" if candidate.model_cached else "cold model"
    gpu_indices = candidate.gpu_indices
    if len(gpu_indices) == 1:
        reason = (
            f"Placed on {source} node {candidate.node.node_id} gpu {gpu_indices[0]} "
            f"with {candidate.min_free_memory_mib} MiB free VRAM ({cache_note})."
        )
    else:
        gpu_text = ",".join(str(index) for index in gpu_indices)
        reason = (
            f"Placed on {source} node {candidate.node.node_id} gpus [{gpu_text}] "
            f"with at least {candidate.min_free_memory_mib} MiB free VRAM on each GPU "
            f"({cache_note})."
        )
    return PlacementDecision(
        status="placed",
        reason=reason,
        agent_id=request.agent_id,
        node_id=candidate.node.node_id,
        host=candidate.node.host,
        gpu_index=gpu_indices[0],
        gpu_indices=gpu_indices,
        gpu_uuid=candidate.gpu_uuids[0] if candidate.gpu_uuids else None,
        gpu_uuids=candidate.gpu_uuids,
        source=source,
        model_cached=candidate.model_cached,
        required_vram_mib=request.required_vram_mib,
        available_vram_mib=candidate.min_free_memory_mib,
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
            (
                f"No non-expired single GPU has at least {request.required_vram_mib} MiB "
                "of free VRAM for this Phase 1 placement request."
            )
            if request.required_gpu_count == 1
            else (
                f"No non-expired single node has {request.required_gpu_count} GPUs with at least "
                f"{request.required_vram_mib} MiB of free VRAM each for this placement request."
            )
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
