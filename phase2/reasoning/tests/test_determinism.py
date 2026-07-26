"""Tests for determinism in the Reasoning Engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from phase2.reasoning.cache import ReasoningCache
from phase2.reasoning.config import ReasoningConfig
from phase2.reasoning.store import ReasoningStore


class MockNode:
    def __init__(self, node_id: str):
        self.node_id = node_id


class MockEdge:
    def __init__(self, edge_id: str):
        self.edge_id = edge_id


class MockGraph:
    def __init__(self):
        self._nodes = {"n1": MockNode("n1"), "n2": MockNode("n2")}
        self._edges = {"e1": MockEdge("e1")}

    def nodes(self):
        return list(self._nodes.values())

    def edges(self):
        return list(self._edges.values())

    def node_count(self):
        return len(self._nodes)

    def edge_count(self):
        return len(self._edges)

    def get_node(self, node_id):
        return self._nodes.get(node_id)

    def get_edge(self, edge_id):
        return self._edges.get(edge_id)

    def neighbors(self, node_id, **kwargs):
        return []

    def predecessors(self, node_id, **kwargs):
        return []

    def successors(self, node_id, **kwargs):
        return []

    def nodes_by_type(self, ntype):
        return []

    def edges_by_type(self, etype):
        return []

    def out_degree(self, node_id):
        return 0

    def in_degree(self, node_id):
        return 0

    def degree(self, node_id):
        return 0

    def subgraph(self, node_ids):
        return self

    def metadata(self, run_id):
        pass


class TestDeterminism:
    def test_cache_hash_identical_graphs(self, tmp_path: Path) -> None:
        store1 = ReasoningStore(tmp_path / "a")
        store2 = ReasoningStore(tmp_path / "b")
        config = ReasoningConfig(
            output_dir=tmp_path / "a",
            cache_enabled=True,
            version="1.0",
            reasoning_version="1.0",
        )
        cache1 = ReasoningCache(store1, config)
        cache2 = ReasoningCache(store2, config)

        graph = MockGraph()
        hash1 = cache1.hash_graph(graph)
        hash2 = cache2.hash_graph(graph)
        assert hash1 == hash2

    def test_cache_hash_different_configs(self, tmp_path: Path) -> None:
        store1 = ReasoningStore(tmp_path / "a")
        store2 = ReasoningStore(tmp_path / "b")
        config1 = ReasoningConfig(output_dir=tmp_path / "a", version="1.0", reasoning_version="1.0")
        config2 = ReasoningConfig(output_dir=tmp_path / "b", version="2.0", reasoning_version="1.0")
        cache1 = ReasoningCache(store1, config1)
        cache2 = ReasoningCache(store2, config2)

        graph = MockGraph()
        hash1 = cache1.hash_graph(graph)
        hash2 = cache2.hash_graph(graph)
        assert hash1 != hash2
