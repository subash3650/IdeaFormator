"""Tests for graph algorithms."""

from __future__ import annotations

import pytest

from phase2.knowledge_graph.algorithms import (
    bfs,
    betweenness_centrality,
    common_neighbors,
    connected_components,
    degree_centrality,
    dfs,
    has_cycle,
    pagerank,
    shortest_path,
    shortest_paths,
    strongly_connected_components,
    topological_sort,
)
from phase2.knowledge_graph.graph_interface import GraphInterface
from phase2.knowledge_graph.schema import EdgeType, GraphEdge, GraphNode, NodeType


def _make_node(node_id: str) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        node_type=NodeType.OBSERVATION,
        label=f"Node {node_id}",
        source_asset="test.parquet",
        source_id=node_id,
        confidence=1.0,
        pipeline_version="1.0",
        schema_version="1.0",
    )


def _make_edge(eid: str, source: str, target: str, etype: EdgeType = EdgeType.SIMILAR_TO) -> GraphEdge:
    return GraphEdge(
        edge_id=eid,
        source_node_id=source,
        target_node_id=target,
        edge_type=etype,
        weight=1.0,
        confidence=1.0,
        source_asset="test.parquet",
        pipeline_version="1.0",
        schema_version="1.0",
    )


def _linear_graph() -> GraphInterface:
    from phase2.knowledge_graph.graph import CustomGraph
    g = CustomGraph()
    for nid in ["a", "b", "c", "d"]:
        g.add_node(_make_node(nid))
    g.add_edge(_make_edge("e1", "a", "b"))
    g.add_edge(_make_edge("e2", "b", "c"))
    g.add_edge(_make_edge("e3", "c", "d"))
    return g


def _cycle_graph() -> GraphInterface:
    from phase2.knowledge_graph.graph import CustomGraph
    g = CustomGraph()
    for nid in ["a", "b", "c"]:
        g.add_node(_make_node(nid))
    g.add_edge(_make_edge("e1", "a", "b"))
    g.add_edge(_make_edge("e2", "b", "c"))
    g.add_edge(_make_edge("e3", "c", "a"))
    return g


def _disconnected_graph() -> GraphInterface:
    from phase2.knowledge_graph.graph import CustomGraph
    g = CustomGraph()
    for nid in ["a", "b", "c", "x", "y"]:
        g.add_node(_make_node(nid))
    g.add_edge(_make_edge("e1", "a", "b"))
    g.add_edge(_make_edge("e2", "b", "c"))
    g.add_edge(_make_edge("e3", "x", "y"))
    return g


def _diamond_graph() -> GraphInterface:
    from phase2.knowledge_graph.graph import CustomGraph
    g = CustomGraph()
    for nid in ["a", "b", "c", "d"]:
        g.add_node(_make_node(nid))
    g.add_edge(_make_edge("e1", "a", "b"))
    g.add_edge(_make_edge("e2", "a", "c"))
    g.add_edge(_make_edge("e3", "b", "d"))
    g.add_edge(_make_edge("e4", "c", "d"))
    return g


def _star_graph() -> GraphInterface:
    from phase2.knowledge_graph.graph import CustomGraph
    g = CustomGraph()
    for nid in ["center", "l1", "l2", "l3", "l4"]:
        g.add_node(_make_node(nid))
    for lid in ["l1", "l2", "l3", "l4"]:
        g.add_edge(_make_edge(f"e_{lid}", "center", lid))
    return g


class TestBFS:
    def test_linear(self):
        g = _linear_graph()
        result = bfs(g, "a")
        assert result == ["a", "b", "c", "d"]

    def test_with_max_depth(self):
        g = _linear_graph()
        result = bfs(g, "a", max_depth=1)
        assert result == ["a", "b"]

    def test_disconnected(self):
        g = _disconnected_graph()
        result = bfs(g, "a")
        assert "x" not in result

    def test_single_node(self):
        from phase2.knowledge_graph.graph import CustomGraph
        g = CustomGraph()
        g.add_node(_make_node("a"))
        assert bfs(g, "a") == ["a"]


class TestDFS:
    def test_linear(self):
        g = _linear_graph()
        result = dfs(g, "a")
        assert result[0] == "a"
        assert len(result) == 4

    def test_with_max_depth(self):
        g = _linear_graph()
        result = dfs(g, "a", max_depth=1)
        assert len(result) <= 2  # a + 1 neighbor


class TestConnectedComponents:
    def test_single_component(self):
        g = _linear_graph()
        comps = connected_components(g)
        assert len(comps) == 1

    def test_disconnected(self):
        g = _disconnected_graph()
        comps = connected_components(g)
        assert len(comps) == 2

    def test_empty(self):
        from phase2.knowledge_graph.graph import CustomGraph
        g = CustomGraph()
        assert connected_components(g) == []


class TestSCC:
    def test_linear(self):
        g = _linear_graph()
        sccs = strongly_connected_components(g)
        assert len(sccs) == 4  # each node is its own SCC

    def test_cycle(self):
        g = _cycle_graph()
        sccs = strongly_connected_components(g)
        assert len(sccs) == 1  # all nodes in one SCC

    def test_diamond(self):
        g = _diamond_graph()
        sccs = strongly_connected_components(g)
        assert len(sccs) == 4


class TestShortestPath:
    def test_linear(self):
        g = _linear_graph()
        assert shortest_path(g, "a", "d") == ["a", "b", "c", "d"]

    def test_no_path(self):
        g = _disconnected_graph()
        assert shortest_path(g, "a", "x") is None

    def test_same_node(self):
        g = _linear_graph()
        assert shortest_path(g, "a", "a") == ["a"]

    def test_diamond(self):
        g = _diamond_graph()
        path = shortest_path(g, "a", "d")
        assert path is not None
        assert len(path) == 3  # a -> b -> d OR a -> c -> d


class TestShortestPaths:
    def test_linear(self):
        g = _linear_graph()
        paths = shortest_paths(g, "a", max_depth=5)
        assert "b" in paths
        assert "d" in paths
        assert len(paths["d"]) == 4


class TestTopologicalSort:
    def test_linear(self):
        g = _linear_graph()
        result = topological_sort(g)
        assert result is not None
        assert len(result) == 4

    def test_cycle(self):
        g = _cycle_graph()
        result = topological_sort(g)
        assert result is None


class TestHasCycle:
    def test_linear_no_cycle(self):
        g = _linear_graph()
        assert has_cycle(g) is False

    def test_cycle(self):
        g = _cycle_graph()
        assert has_cycle(g) is True

    def test_empty(self):
        from phase2.knowledge_graph.graph import CustomGraph
        g = CustomGraph()
        assert has_cycle(g) is False


class TestDegreeCentrality:
    def test_star(self):
        g = _star_graph()
        scores = degree_centrality(g, top_k=5)
        node_dict = dict(scores)
        assert node_dict["center"] == 1.0  # max degree centrality

    def test_top_k(self):
        g = _star_graph()
        scores = degree_centrality(g, top_k=2)
        assert len(scores) == 2


class TestBetweennessCentrality:
    def test_linear(self):
        g = _linear_graph()
        scores = betweenness_centrality(g, top_k=4)
        node_dict = dict(scores)
        # Middle nodes have higher betweenness
        assert node_dict.get("b", 0) > 0

    def test_star_center(self):
        g = _star_graph()
        scores = betweenness_centrality(g, top_k=5)
        node_dict = dict(scores)
        # In a directed star (center->leaves), center has no betweenness
        # since no paths pass through it between distinct pairs
        assert "center" in node_dict


class TestPageRank:
    def test_linear(self):
        g = _linear_graph()
        scores = pagerank(g, top_k=4)
        assert len(scores) == 4
        # All scores should be positive
        for _, score in scores:
            assert score > 0

    def test_empty(self):
        from phase2.knowledge_graph.graph import CustomGraph
        g = CustomGraph()
        assert pagerank(g) == []

    def test_single_node(self):
        from phase2.knowledge_graph.graph import CustomGraph
        g = CustomGraph()
        g.add_node(_make_node("a"))
        scores = pagerank(g)
        assert len(scores) == 1
        assert scores[0][1] == 1.0


class TestCommonNeighbors:
    def test_diamond(self):
        g = _diamond_graph()
        common = common_neighbors(g, "b", "c")
        assert "a" in common
        assert "d" in common

    def test_no_common(self):
        g = _disconnected_graph()
        common = common_neighbors(g, "a", "x")
        assert common == []
