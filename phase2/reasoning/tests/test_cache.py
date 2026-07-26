"""Tests for ReasoningCache."""

from __future__ import annotations

from pathlib import Path

from phase2.reasoning.cache import ReasoningCache
from phase2.reasoning.config import ReasoningConfig
from phase2.reasoning.schema import InferenceOutput, ReasoningMetadata
from phase2.reasoning.store import ReasoningStore


class MockNode:
    def __init__(self, node_id: str):
        self.node_id = node_id


class MockEdge:
    def __init__(self, edge_id: str):
        self.edge_id = edge_id


class MockGraph:
    def __init__(self, node_ids: list[str], edge_ids: list[str]):
        self._nodes = [MockNode(nid) for nid in node_ids]
        self._edges = [MockEdge(eid) for eid in edge_ids]

    def nodes(self):
        return list(self._nodes)

    def edges(self):
        return list(self._edges)

    def node_count(self):
        return len(self._nodes)

    def edge_count(self):
        return len(self._edges)

    def get_node(self, node_id):
        for n in self._nodes:
            if n.node_id == node_id:
                return n
        return None

    def get_edge(self, edge_id):
        for e in self._edges:
            if e.edge_id == edge_id:
                return e
        return None

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


class TestReasoningCache:
    def test_hash_changes_with_nodes(self, tmp_path: Path) -> None:
        store = ReasoningStore(tmp_path)
        config = ReasoningConfig(output_dir=tmp_path, cache_enabled=True)
        cache = ReasoningCache(store, config)

        graph1 = MockGraph(["a", "b"], ["e1"])
        graph2 = MockGraph(["a", "b", "c"], ["e1"])

        hash1 = cache.hash_graph(graph1)
        hash2 = cache.hash_graph(graph2)
        assert hash1 != hash2

    def test_hash_changes_with_edges(self, tmp_path: Path) -> None:
        store = ReasoningStore(tmp_path)
        config = ReasoningConfig(output_dir=tmp_path, cache_enabled=True)
        cache = ReasoningCache(store, config)

        graph1 = MockGraph(["a", "b"], ["e1"])
        graph2 = MockGraph(["a", "b"], ["e2"])

        hash1 = cache.hash_graph(graph1)
        hash2 = cache.hash_graph(graph2)
        assert hash1 != hash2

    def test_is_valid_returns_false_for_empty_cache(self, tmp_path: Path) -> None:
        store = ReasoningStore(tmp_path)
        config = ReasoningConfig(output_dir=tmp_path, cache_enabled=True)
        cache = ReasoningCache(store, config)

        graph = MockGraph(["a"], ["e1"])
        assert cache.is_valid(graph) is False

    def test_save_and_load_cycle(self, tmp_path: Path) -> None:
        store = ReasoningStore(tmp_path)
        config = ReasoningConfig(output_dir=tmp_path, cache_enabled=True)
        cache = ReasoningCache(store, config)

        graph = MockGraph(["a", "b"], ["e1"])
        output = InferenceOutput(
            metadata=ReasoningMetadata(run_id="test-run"),
        )
        cache.save(graph, output)
        assert cache.is_valid(graph) is True

        loaded = cache.load()
        assert loaded is not None

    def test_invalidate(self, tmp_path: Path) -> None:
        store = ReasoningStore(tmp_path)
        config = ReasoningConfig(output_dir=tmp_path, cache_enabled=True)
        cache = ReasoningCache(store, config)

        graph = MockGraph(["a"], ["e1"])
        output = InferenceOutput(metadata=ReasoningMetadata(run_id="test"))
        cache.save(graph, output)
        cache.invalidate()
        assert cache.is_valid(graph) is False

    def test_cache_disabled(self, tmp_path: Path) -> None:
        store = ReasoningStore(tmp_path)
        config = ReasoningConfig(output_dir=tmp_path, cache_enabled=False)
        cache = ReasoningCache(store, config)

        graph = MockGraph(["a"], ["e1"])
        assert cache.is_valid(graph) is False
