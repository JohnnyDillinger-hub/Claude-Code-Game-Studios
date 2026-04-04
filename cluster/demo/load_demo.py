from __future__ import annotations

from cluster.demo import LOCAL_NODE_PATH, REMOTE_NODES_PATH
from cluster.models import NodeInventory, load_node_inventory_file


def load_demo_local_node() -> NodeInventory:
    return load_node_inventory_file(LOCAL_NODE_PATH)[0]


def load_demo_remote_nodes() -> list[NodeInventory]:
    return load_node_inventory_file(REMOTE_NODES_PATH)
