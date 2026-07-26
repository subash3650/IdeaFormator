"""Tests for GraphSearch."""

from __future__ import annotations

import pytest

from phase2.knowledge_graph.graph import CustomGraph
from phase2.knowledge_graph.schema import EdgeType, GraphEdge, GraphNode, NodeType
from phase2.knowledge_graph.search import GraphSearch


def _make_node(nid: str, ntype: NodeType = NodeType.OBSERVATION, label: str | None = None) -> GraphNode:
    return GraphNode(
        node_id=nid,
        node_type=ntype,
        label=label or f"Node {nid}",
        properties={"prop_key": nid},
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
def searchable_graph() -> GraphSearch:
    g = CustomGraph()

    # Create diverse node types
    g.add_node(_make_node("obs1", NodeType.OBSERVATION, "Product crash"))
    g.add_node(_make_node("obs2", NodeType.OBSERVATION, "UI bug"))
    g.add_node(_make_node("ev1", NodeType.EVIDENCE, "Multiple crash reports"))
    g.add_node(_make_node("sig1", NodeType.PROBLEM_SIGNAL, "Stability issues"))
    g.add_node(_make_node("cl1", NodeType.CLUSTER, "Crash cluster"))
    g.add_node(_make_node("company1", NodeType.COMPANY, "Acme Corp"))
    g.add_node(_make_node("product1", NodeType.PRODUCT, "Widget Pro"))

    # Create diverse edges
    g.add_edge(_make_edge("e1", "obs1", "ev1", EdgeType.DERIVED_FROM))
    g.add_edge(_make_edge("e2", "ev1", "sig1", EdgeType.CAUSES))
    g.add_edge(_make_edge("e3", "obs1", "cl1", EdgeType.MEMBER_OF_CLUSTER))
    g.add_edge(_make_edge("e4", "obs1", "obs2", EdgeType.SIMILAR_TO))

    return GraphSearch(g)


class TestGraphSearch:
    def test_find_node_exact(self, searchable_graph):
        node = searchable_graph.find_node("obs1")
        assert node is not None
        assert node.node_id == "obs1"

    def test_find_node_missing(self, searchable_graph):
        assert searchable_graph.find_node("missing") is None

    def test_find_nodes_by_type(self, searchable_graph):
        nodes = searchable_graph.find_nodes_by_type(NodeType.OBSERVATION)
        assert len(nodes) == 2

    def test_find_by_label_fuzzy(self, searchable_graph):
        nodes = searchable_graph.find_by_label("crash", fuzzy=True)
        assert len(nodes) >= 1

    def test_find_by_label_exact(self, searchable_graph):
        nodes = searchable_graph.find_by_label("Product crash", fuzzy=False)
        assert len(nodes) == 1

    def test_find_by_property(self, searchable_graph):
        nodes = searchable_graph.find_by_property("prop_key", "obs1")
        assert len(nodes) == 1

    def test_find_observations(self, searchable_graph):
        nodes = searchable_graph.find_observations()
        assert len(nodes) == 2

    def test_find_evidence(self, searchable_graph):
        nodes = searchable_graph.find_evidence()
        assert len(nodes) == 1

    def test_find_problem_signals(self, searchable_graph):
        nodes = searchable_graph.find_problem_signals()
        assert len(nodes) == 1

    def test_find_clusters(self, searchable_graph):
        nodes = searchable_graph.find_clusters()
        assert len(nodes) == 1

    def test_find_similar(self, searchable_graph):
        nodes = searchable_graph.find_similar("obs1")
        assert len(nodes) >= 1

    def test_find_neighbors_depth1(self, searchable_graph):
        neighbors = searchable_graph.find_neighbors("obs1", depth=1)
        assert len(neighbors) >= 2  # ev1, obs2, cl1

    def test_find_neighbors_depth0(self, searchable_graph):
        neighbors = searchable_graph.find_neighbors("obs1", depth=0)
        assert len(neighbors) == 1
        assert neighbors[0].node_id == "obs1"

    def test_find_common_neighbors(self, searchable_graph):
        # obs1 and obs2 don't have common neighbors in this setup
        common = searchable_graph.find_common_neighbors("obs1", "ev1")
        assert isinstance(common, list)

    def test_find_predecessors(self, searchable_graph):
        preds = searchable_graph.find_predecessors("ev1")
        assert len(preds) == 1
        assert preds[0].node_id == "obs1"

    def test_find_path(self, searchable_graph):
        path = searchable_graph.find_path("obs1", "sig1")
        assert path is not None
        assert "obs1" in path
        assert "sig1" in path

    def test_find_path_unreachable(self, searchable_graph):
        path = searchable_graph.find_path("company1", "sig1")
        # These nodes are disconnected
        assert path is None or len(path) > 3

    def test_find_cluster_members(self, searchable_graph):
        members = searchable_graph.find_cluster("cl1")
        assert len(members) == 1
        assert members[0].node_id == "obs1"

    def test_find_central_nodes(self, searchable_graph):
        central = searchable_graph.find_central_nodes(metric="degree", top_k=3)
        assert len(central) >= 1

    def test_find_by_confidence(self, searchable_graph):
        nodes = searchable_graph.find_by_confidence(0.9)
        assert len(nodes) > 0

    def test_find_by_weight(self, searchable_graph):
        edges = searchable_graph.find_by_weight(0.5)
        assert len(edges) > 0

    def test_empty_graph(self):
        g = CustomGraph()
        search = GraphSearch(g)
        assert search.find_node("a") is None
        assert search.find_observations() == []
        assert search.find_path("a", "b") is None

    def test_find_entities(self, searchable_graph):
        nodes = searchable_graph.find_entities()
        assert isinstance(nodes, list)

    def test_find_companies(self, searchable_graph):
        nodes = searchable_graph.find_companies()
        assert len(nodes) == 1
        assert nodes[0].node_id == "company1"
