from __future__ import annotations

import unittest

from cluster.demo.load_demo import load_demo_local_node, load_demo_remote_nodes
from cluster.models import AgentRequest, GPUInventory, LeaseInfo, NodeInventory, parse_datetime
from cluster.orchestrator.registry import NodeRegistry
from cluster.orchestrator.scheduler import reserve_gpu_capacity, schedule_agent


def make_node(
    node_id: str,
    *,
    host: str | None = None,
    available_until: str = "2035-01-01T00:00:00Z",
    gpus: list[GPUInventory] | None = None,
    cached_models: tuple[str, ...] = (),
) -> NodeInventory:
    return NodeInventory(
        node_id=node_id,
        host=host or f"{node_id}.example",
        lease=LeaseInfo(available_until=parse_datetime(available_until)),
        gpus=tuple(gpus or ()),
        cached_models=cached_models,
    )


def make_gpu(index: int, free_mib: int, total_mib: int = 49152) -> GPUInventory:
    return GPUInventory(
        index=index,
        uuid=f"GPU-{index}",
        total_memory_mib=total_mib,
        free_memory_mib=free_mib,
        utilization_pct=0,
    )


class ClusterPhase1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = parse_datetime("2026-04-04T00:00:00Z")

    def test_demo_inventories_load(self) -> None:
        local = load_demo_local_node()
        remotes = load_demo_remote_nodes()
        self.assertEqual(local.node_id, "local-lab")
        self.assertGreaterEqual(len(remotes), 3)
        self.assertTrue(any(node.gpu_count == 4 for node in remotes))

    def test_node_expiration_handling(self) -> None:
        expired = make_node(
            "expired-node",
            available_until="2026-04-03T00:00:00Z",
            gpus=[make_gpu(0, 12000)],
        )
        active = make_node("active-node", gpus=[make_gpu(0, 12000)])
        registry = NodeRegistry([expired, active])

        removed = registry.prune_expired(now=self.now)

        self.assertEqual(removed, ["expired-node"])
        self.assertEqual([node.node_id for node in registry.list_nodes()], ["active-node"])

    def test_scheduler_prefers_local_first(self) -> None:
        local = make_node("local", gpus=[make_gpu(0, 24000)])
        remote = make_node(
            "remote",
            gpus=[make_gpu(0, 64000)],
            cached_models=("qwen3-coder:30b",),
        )
        request = AgentRequest(
            agent_id="local-first",
            model_id="qwen3-coder:30b",
            required_vram_mib=20000,
        )

        decision = schedule_agent(local, [remote], request, now=self.now)

        self.assertTrue(decision.is_placed)
        self.assertEqual(decision.source, "local")
        self.assertEqual(decision.node_id, "local")

    def test_scheduler_prefers_cached_model_when_local_is_insufficient(self) -> None:
        local = make_node("local", gpus=[make_gpu(0, 12000)])
        remote_uncached = make_node(
            "remote-b",
            available_until="2035-01-03T00:00:00Z",
            gpus=[make_gpu(0, 70000)],
        )
        remote_cached = make_node(
            "remote-a",
            available_until="2035-01-02T00:00:00Z",
            gpus=[make_gpu(0, 65000)],
            cached_models=("qwen3-coder:30b",),
        )
        request = AgentRequest(
            agent_id="cache-pref",
            model_id="qwen3-coder:30b",
            required_vram_mib=45000,
        )

        decision = schedule_agent(local, [remote_uncached, remote_cached], request, now=self.now)

        self.assertEqual(decision.node_id, "remote-a")
        self.assertTrue(decision.model_cached)

    def test_scheduler_prefers_longest_lease_after_cache_rule(self) -> None:
        local = make_node("local", gpus=[make_gpu(0, 8000)])
        shorter = make_node(
            "node-b",
            available_until="2035-01-02T00:00:00Z",
            gpus=[make_gpu(0, 40000)],
            cached_models=("gemma3:12b",),
        )
        longer = make_node(
            "node-a",
            available_until="2035-01-03T00:00:00Z",
            gpus=[make_gpu(0, 39000)],
            cached_models=("gemma3:12b",),
        )
        request = AgentRequest(
            agent_id="lease-pref",
            model_id="gemma3:12b",
            required_vram_mib=20000,
        )

        decision = schedule_agent(local, [shorter, longer], request, now=self.now)

        self.assertEqual(decision.node_id, "node-a")

    def test_tie_breaking_is_stable_by_node_id(self) -> None:
        local = make_node("local", gpus=[make_gpu(0, 8000)])
        node_b = make_node("node-b", gpus=[make_gpu(0, 32000)])
        node_a = make_node("node-a", gpus=[make_gpu(0, 32000)])
        request = AgentRequest(agent_id="tie-break", required_vram_mib=16000)

        decision = schedule_agent(local, [node_b, node_a], request, now=self.now)

        self.assertEqual(decision.node_id, "node-a")

    def test_multi_gpu_node_hosts_multiple_single_gpu_agents_with_explicit_reservation(self) -> None:
        local = make_node("local", gpus=[make_gpu(0, 5000)])
        four_gpu = make_node(
            "remote-4gpu",
            gpus=[make_gpu(index, 12000) for index in range(4)],
        )
        request = AgentRequest(
            agent_id="parallel-agents",
            model_id="gemma3:12b",
            required_vram_mib=10000,
        )

        chosen_gpus: list[int] = []
        current_remote = four_gpu
        for _ in range(4):
            decision = schedule_agent(local, [current_remote], request, now=self.now)
            self.assertTrue(decision.is_placed)
            chosen_gpus.append(decision.gpu_index)
            current_remote = reserve_gpu_capacity(
                current_remote,
                gpu_index=decision.gpu_index,
                amount_mib=request.required_vram_mib,
            )

        final_decision = schedule_agent(local, [current_remote], request, now=self.now)

        self.assertEqual(chosen_gpus, [0, 1, 2, 3])
        self.assertEqual(final_decision.status, "rejected")

    def test_graceful_handling_of_no_valid_nodes(self) -> None:
        local = make_node(
            "local",
            available_until="2026-04-03T00:00:00Z",
            gpus=[make_gpu(0, 20000)],
        )
        remote = make_node("remote", gpus=[make_gpu(0, 10000)])
        request = AgentRequest(
            agent_id="no-capacity",
            model_id="qwen3-coder:30b",
            required_vram_mib=45000,
        )

        decision = schedule_agent(local, [remote], request, now=self.now)

        self.assertEqual(decision.status, "rejected")
        self.assertIn("No non-expired single GPU", decision.reason)


if __name__ == "__main__":
    unittest.main()
