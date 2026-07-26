"""Tests for adjacency indexes."""

from __future__ import annotations

import pytest

from phase2.knowledge_graph.adjacency import AdjacencyIndex, ReverseAdjacencyIndex
from phase2.knowledge_graph.schema import EdgeType, GraphEdge


def _make_edge(edge_id: str, source: str, target: str, etype: EdgeType = EdgeType.SIMILAR_TO) -> GraphEdge:
    return GraphEdge(
        edge_id=edge_id,
        source_node_id=source,
        target_node_id=target,
        edge_type=etype,
        weight=0.8,
        confidence=0.9,
        source_asset="test.parquet",
        pipeline_version="1.5.0",
        schema_version="1.0.0",
    )


class TestAdjacencyIndex:
    def test_add_and_neighbors(self):
        idx = AdjacencyIndex()
        edge = _make_edge("e1", "a", "b")
        idx.add(edge)
        assert idx.neighbors("a") == ["b"]
        assert idx.neighbors("b") == []

    def test_typed_neighbors(self):
        idx = AdjacencyIndex()
        edge = _make_edge("e1", "a", "b", EdgeType.SIMILAR_TO)
        idx.add(edge)
        assert idx.neighbors("a", EdgeType.SIMILAR_TO) == ["b"]
        assert idx.neighbors("a", EdgeType.CAUSES) == []

    def test_out_degree(self):
        idx = AdjacencyIndex()
        idx.add(_make_edge("e1", "a", "b"))
        idx.add(_make_edge("e2", "a", "c"))
        assert idx.out_degree("a") == 2
        assert idx.out_degree("b") == 0

    def test_remove_edge(self):
        idx = AdjacencyIndex()
        edge = _make_edge("e1", "a", "b")
        idx.add(edge)
        idx.remove("e1", edge)
        assert idx.neighbors("a") == []
        assert idx.has_edge("e1") is False

    def test_remove_node(self):
        idx = AdjacencyIndex()
        idx.add(_make_edge("e1", "a", "b"))
        idx.add(_make_edge("e2", "a", "c"))
        idx.remove_node("a")
        assert idx.neighbors("a") == []
        assert idx.out_degree("a") == 0

    def test_has_edge(self):
        idx = AdjacencyIndex()
        idx.add(_make_edge("e1", "a", "b"))
        assert idx.has_edge("e1") is True
        assert idx.has_edge("missing") is False

    def test_edge_target(self):
        idx = AdjacencyIndex()
        idx.add(_make_edge("e1", "a", "b"))
        assert idx.edge_target("e1") == "b"
        assert idx.edge_target("missing") is None

    def test_clear(self):
        idx = AdjacencyIndex()
        idx.add(_make_edge("e1", "a", "b"))
        idx.clear()
        assert idx.neighbors("a") == []


class TestReverseAdjacencyIndex:
    def test_add_and_predecessors(self):
        idx = ReverseAdjacencyIndex()
        edge = _make_edge("e1", "a", "b")
        idx.add(edge)
        assert idx.predecessors("a") == []
        assert idx.predecessors("b") == ["a"]

    def test_typed_predecessors(self):
        idx = ReverseAdjacencyIndex()
        edge = _make_edge("e1", "a", "b", EdgeType.SIMILAR_TO)
        idx.add(edge)
        assert idx.predecessors("b", EdgeType.SIMILAR_TO) == ["a"]
        assert idx.predecessors("b", EdgeType.CAUSES) == []

    def test_in_degree(self):
        idx = ReverseAdjacencyIndex()
        idx.add(_make_edge("e1", "a", "c"))
        idx.add(_make_edge("e2", "b", "c"))
        assert idx.in_degree("c") == 2
        assert idx.in_degree("a") == 0

    def test_remove_edge(self):
        idx = ReverseAdjacencyIndex()
        edge = _make_edge("e1", "a", "b")
        idx.add(edge)
        idx.remove("e1", edge)
        assert idx.predecessors("b") == []

    def test_remove_node(self):
        idx = ReverseAdjacencyIndex()
        idx.add(_make_edge("e1", "a", "b"))
        idx.add(_make_edge("e2", "b", "a"))
        idx.remove_node("b")
        assert idx.in_degree("b") == 0

    def test_has_edge(self):
        idx = ReverseAdjacencyIndex()
        idx.add(_make_edge("e1", "a", "b"))
        assert idx.has_edge("e1") is True

    def test_edge_source(self):
        idx = ReverseAdjacencyIndex()
        idx.add(_make_edge("e1", "a", "b"))
        assert idx.edge_source("e1") == "a"
