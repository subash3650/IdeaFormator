"""Tests for GraphEvaluator."""

from __future__ import annotations

from phase2.knowledge_graph.graph import CustomGraph
from phase2.knowledge_graph.schema import EdgeType, GraphEdge, GraphNode, NodeType
from phase2.knowledge_graph.evaluator import GraphEvaluator


def _make_node(nid: str, ntype: NodeType = NodeType.OBSERVATION, confidence: float = 1.0) -> GraphNode:
    return GraphNode(
        node_id=nid,
        node_type=ntype,
        label=f"Node {nid}",
        source_asset="test.parquet",
        source_id=nid,
        confidence=confidence,
        pipeline_version="1.0",
        schema_version="1.0",
    )


def _make_edge(eid: str, src: str, tgt: str) -> GraphEdge:
    return GraphEdge(
        edge_id=eid,
        source_node_id=src,
        target_node_id=tgt,
        edge_type=EdgeType.SIMILAR_TO,
        weight=0.8,
        confidence=0.9,
        source_asset="test.parquet",
        pipeline_version="1.0",
        schema_version="1.0",
    )


def _healthy_graph() -> CustomGraph:
    g = CustomGraph()
    for i in range(20):
        g.add_node(_make_node(f"n{i}", confidence=0.9 + (i % 10) * 0.01))
    for i in range(15):
        g.add_edge(_make_edge(f"e{i}", f"n{i}", f"n{i+1}"))
    return g


def _sparse_graph() -> CustomGraph:
    g = CustomGraph()
    g.add_node(_make_node("a", confidence=0.5))
    g.add_node(_make_node("b", confidence=0.5))
    return g


def _empty_graph() -> CustomGraph:
    return CustomGraph()


class TestGraphEvaluator:
    def test_healthy_graph(self):
        evaluator = GraphEvaluator()
        result = evaluator.evaluate(_healthy_graph())
        assert result["health"]["score"] >= 80
        assert result["health"]["status"] == "healthy"

    def test_sparse_graph(self):
        evaluator = GraphEvaluator()
        result = evaluator.evaluate(_sparse_graph())
        assert result["health"]["score"] < 80

    def test_empty_graph(self):
        evaluator = GraphEvaluator()
        result = evaluator.evaluate(_empty_graph())
        assert result["node_count"] == 0
        assert result["health"]["score"] < 50
        assert result["health"]["status"] == "critical"

    def test_type_distribution(self):
        g = CustomGraph()
        g.add_node(_make_node("a", NodeType.OBSERVATION))
        g.add_node(_make_node("b", NodeType.EVIDENCE))
        result = GraphEvaluator().evaluate(g)
        assert "observation" in result["type_distribution"]
        assert "evidence" in result["type_distribution"]

    def test_edge_type_distribution(self):
        g = CustomGraph()
        g.add_node(_make_node("a"))
        g.add_node(_make_node("b"))
        g.add_edge(_make_edge("e1", "a", "b"))
        result = GraphEvaluator().evaluate(g)
        assert "similar_to" in result["edge_type_distribution"]

    def test_evaluate_metadata(self):
        from phase2.knowledge_graph.schema import GraphMetadata
        meta = GraphMetadata(
            graph_id="g1", node_count=50, edge_count=100,
            connected_components=1, largest_component_size=50,
            density=0.1, avg_confidence=0.9, avg_degree=2.0,
            orphan_node_count=0, run_id="r1",
            pipeline_version="1.0", schema_version="1.0",
        )
        health = GraphEvaluator().evaluate_metadata(meta)
        assert health["score"] >= 80

    def test_warnings_healthy(self):
        result = GraphEvaluator().evaluate(_healthy_graph())
        assert len(result["warnings"]) == 0
        assert len(result["recommendations"]) == 0
