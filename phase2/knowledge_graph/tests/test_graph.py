"""Tests for CustomGraph."""

from __future__ import annotations

import pytest

from phase2.knowledge_graph.graph import CustomGraph
from phase2.knowledge_graph.schema import EdgeType, GraphEdge, GraphNode, NodeType


def _make_node(nid: str, ntype: NodeType = NodeType.OBSERVATION) -> GraphNode:
    return GraphNode(
        node_id=nid,
        node_type=ntype,
        label=f"Node {nid}",
        source_asset="test.parquet",
        source_id=nid,
        confidence=1.0,
        pipeline_version="1.0",
        schema_version="1.0",
    )


def _make_edge(eid: str, src: str, tgt: str, etype: EdgeType = EdgeType.SIMILAR_TO) -> GraphEdge:
    return GraphEdge(
        edge_id=eid,
        source_node_id=src,
        target_node_id=tgt,
        edge_type=etype,
        weight=0.8,
        confidence=0.9,
        source_asset="test.parquet",
        pipeline_version="1.0",
        schema_version="1.0",
    )


@pytest.fixture
def empty_graph() -> CustomGraph:
    return CustomGraph()


@pytest.fixture
def small_graph() -> CustomGraph:
    g = CustomGraph()
    for nid in ["a", "b", "c"]:
        g.add_node(_make_node(nid))
    g.add_edge(_make_edge("e1", "a", "b"))
    g.add_edge(_make_edge("e2", "b", "c"))
    return g


class TestCustomGraphMutation:
    def test_add_node(self, empty_graph):
        node = _make_node("a")
        empty_graph.add_node(node)
        assert empty_graph.has_node("a") is True
        assert empty_graph.node_count() == 1

    def test_add_edge(self, small_graph):
        assert small_graph.has_edge("e1") is True
        assert small_graph.edge_count() == 2

    def test_remove_node(self, small_graph):
        small_graph.remove_node("a")
        assert small_graph.has_node("a") is False
        assert small_graph.edge_count() == 1

    def test_remove_edge(self, small_graph):
        small_graph.remove_edge("e1")
        assert small_graph.has_edge("e1") is False
        assert small_graph.edge_count() == 1

    def test_clear(self, small_graph):
        small_graph.clear()
        assert small_graph.node_count() == 0
        assert small_graph.edge_count() == 0

    def test_add_node_deduplication(self, empty_graph):
        node = _make_node("a")
        empty_graph.add_node(node)
        empty_graph.add_node(node)
        assert empty_graph.node_count() == 1


class TestCustomGraphAccessors:
    def test_get_node(self, small_graph):
        node = small_graph.get_node("a")
        assert node is not None
        assert node.node_id == "a"

    def test_get_node_missing(self, small_graph):
        assert small_graph.get_node("missing") is None

    def test_get_edge(self, small_graph):
        edge = small_graph.get_edge("e1")
        assert edge is not None
        assert edge.edge_id == "e1"

    def test_get_edge_missing(self, small_graph):
        assert small_graph.get_edge("missing") is None

    def test_nodes(self, small_graph):
        nodes = small_graph.nodes()
        assert len(nodes) == 3

    def test_edges(self, small_graph):
        edges = small_graph.edges()
        assert len(edges) == 2

    def test_nodes_by_type(self):
        g = CustomGraph()
        g.add_node(_make_node("a", NodeType.OBSERVATION))
        g.add_node(_make_node("b", NodeType.EVIDENCE))
        obs = g.nodes_by_type(NodeType.OBSERVATION)
        assert len(obs) == 1
        assert obs[0].node_id == "a"

    def test_edges_by_type(self):
        g = CustomGraph()
        g.add_node(_make_node("a"))
        g.add_node(_make_node("b"))
        g.add_edge(_make_edge("e1", "a", "b", EdgeType.SIMILAR_TO))
        g.add_edge(_make_edge("e2", "b", "a", EdgeType.CAUSES))
        assert len(g.edges_by_type(EdgeType.SIMILAR_TO)) == 1

    def test_node_type_counts(self, small_graph):
        counts = small_graph.node_type_counts()
        assert counts.get("observation") == 3

    def test_edge_type_counts(self, small_graph):
        counts = small_graph.edge_type_counts()
        assert counts.get("similar_to") == 2


class TestCustomGraphTraversal:
    def test_neighbors(self, small_graph):
        neighbors = small_graph.neighbors("a")
        assert len(neighbors) == 1
        assert neighbors[0] == "b"

    def test_predecessors(self, small_graph):
        preds = small_graph.predecessors("b")
        assert len(preds) == 1
        assert preds[0] == "a"

    def test_successors(self, small_graph):
        succs = small_graph.successors("b")
        assert len(succs) == 1
        assert succs[0] == "c"

    def test_out_degree(self, small_graph):
        assert small_graph.out_degree("a") == 1
        assert small_graph.out_degree("c") == 0

    def test_in_degree(self, small_graph):
        assert small_graph.in_degree("a") == 0
        assert small_graph.in_degree("b") == 1

    def test_degree(self, small_graph):
        assert small_graph.degree("a") == 1
        assert small_graph.degree("c") == 1


class TestCustomGraphSubgraph:
    def test_subgraph(self, small_graph):
        sub = small_graph.subgraph({"a", "b"})
        assert sub.node_count() == 2
        assert sub.edge_count() == 1

    def test_subgraph_no_edges(self, small_graph):
        sub = small_graph.subgraph({"a", "c"})
        assert sub.node_count() == 2
        assert sub.edge_count() == 0


class TestCustomGraphMetadata:
    def test_metadata(self, small_graph):
        meta = small_graph.metadata("run1")
        assert meta.node_count == 3
        assert meta.edge_count == 2
        assert meta.run_id == "run1"
        assert meta.connected_components == 1
