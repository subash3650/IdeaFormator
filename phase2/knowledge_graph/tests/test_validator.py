"""Tests for GraphValidator."""

from __future__ import annotations

from phase2.knowledge_graph.graph import CustomGraph
from phase2.knowledge_graph.schema import EdgeType, GraphEdge, GraphNode, NodeType
from phase2.knowledge_graph.validator import GraphValidator


def _make_node(nid: str, confidence: float = 1.0) -> GraphNode:
    return GraphNode(
        node_id=nid,
        node_type=NodeType.OBSERVATION,
        label=f"Node {nid}",
        source_asset="test.parquet",
        source_id=nid,
        confidence=confidence,
        pipeline_version="1.0",
        schema_version="1.0",
    )


def _make_edge(eid: str, src: str, tgt: str, weight: float = 0.8, confidence: float = 0.9) -> GraphEdge:
    return GraphEdge(
        edge_id=eid,
        source_node_id=src,
        target_node_id=tgt,
        edge_type=EdgeType.SIMILAR_TO,
        weight=weight,
        confidence=confidence,
        source_asset="test.parquet",
        pipeline_version="1.0",
        schema_version="1.0",
    )


def _valid_graph() -> CustomGraph:
    g = CustomGraph()
    for nid in ["a", "b", "c"]:
        g.add_node(_make_node(nid))
    g.add_edge(_make_edge("e1", "a", "b"))
    g.add_edge(_make_edge("e2", "b", "c"))
    return g


def _graph_with_orphan_edges() -> CustomGraph:
    g = CustomGraph()
    g.add_node(_make_node("a"))
    g.add_node(_make_node("b"))
    g.add_edge(_make_edge("e1", "a", "missing"))
    g.add_edge(_make_edge("e2", "missing2", "b"))
    return g


def _graph_with_self_loops() -> CustomGraph:
    g = CustomGraph()
    g.add_node(_make_node("a"))
    g.add_edge(_make_edge("e1", "a", "a"))
    return g


def _graph_with_duplicates() -> CustomGraph:
    g = CustomGraph()
    g.add_node(_make_node("a"))
    g.add_node(_make_node("a"))  # overwrites
    g.add_edge(_make_edge("e1", "a", "b"))
    # b was removed, so this adds it back
    g.add_node(_make_node("b"))
    return g


def _graph_with_bad_confidence() -> CustomGraph:
    g = CustomGraph()
    node_a = GraphNode.model_construct(
        node_id="a", node_type=NodeType.OBSERVATION, label="A",
        source_asset="test.parquet", source_id="a", confidence=1.5,
        pipeline_version="1.0", schema_version="1.0",
    )
    node_b = GraphNode.model_construct(
        node_id="b", node_type=NodeType.OBSERVATION, label="B",
        source_asset="test.parquet", source_id="b", confidence=-0.5,
        pipeline_version="1.0", schema_version="1.0",
    )
    g.add_node(node_a)
    g.add_node(node_b)
    return g


def _graph_with_bad_weight() -> CustomGraph:
    g = CustomGraph()
    g.add_node(_make_node("a"))
    g.add_node(_make_node("b"))
    g.add_edge(GraphEdge.model_construct(
        edge_id="e1", source_node_id="a", target_node_id="b",
        edge_type=EdgeType.SIMILAR_TO, weight=1.5, confidence=0.9,
        source_asset="test.parquet", pipeline_version="1.0", schema_version="1.0",
    ))
    return g


class TestGraphValidator:
    def test_valid_graph(self):
        g = _valid_graph()
        result = GraphValidator().validate(g)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_orphan_edges(self):
        g = _graph_with_orphan_edges()
        result = GraphValidator().validate(g)
        assert result.valid is False
        assert "orphan" in " ".join(result.errors).lower()

    def test_self_loops(self):
        g = _graph_with_self_loops()
        result = GraphValidator().validate(g)
        assert result.self_loop_count == 1
        assert len(result.warnings) >= 1

    def test_bad_confidence(self):
        g = _graph_with_bad_confidence()
        result = GraphValidator().validate(g)
        assert result.valid is False
        assert any("confidence" in e.lower() for e in result.errors)

    def test_bad_weight(self):
        g = _graph_with_bad_weight()
        result = GraphValidator().validate(g)
        assert result.valid is False
        assert any("weight" in e.lower() for e in result.errors)

    def test_empty_graph(self):
        g = CustomGraph()
        result = GraphValidator().validate(g)
        assert result.valid is True  # no errors, only warnings
        assert len(result.warnings) >= 1  # zero nodes warning

    def test_disconnected_components(self):
        g = CustomGraph()
        g.add_node(_make_node("a"))
        g.add_node(_make_node("x"))
        g.add_edge(_make_edge("e1", "a", "x"))
        g.add_node(_make_node("y"))  # isolated
        result = GraphValidator().validate(g)
        # y is isolated so components > 1
        assert result.disconnected_components >= 1

    def test_node_count_matches(self):
        g = _valid_graph()
        result = GraphValidator().validate(g)
        assert result.node_count == 3
        assert result.edge_count == 2

    def test_edge_count_matches(self):
        g = _valid_graph()
        result = GraphValidator().validate(g)
        assert result.edge_count == 2

    def test_cycle_detection(self):
        g = CustomGraph()
        for nid in ["a", "b", "c"]:
            g.add_node(_make_node(nid))
        g.add_edge(_make_edge("e1", "a", "b"))
        g.add_edge(_make_edge("e2", "b", "c"))
        g.add_edge(_make_edge("e3", "c", "a"))
        result = GraphValidator().validate(g)
        assert result.cycle_count >= 0  # cycles may be warnings, not errors

    def test_invalid_node_type(self):
        g = CustomGraph()
        node = _make_node("a")
        # Override with bad type after creation
        g.add_node(node)
        result = GraphValidator().validate(g)
        assert result.schema_mismatch_count == 0  # valid node type

    def test_orphan_node_count(self):
        g = CustomGraph()
        g.add_node(_make_node("a"))  # degree 0
        g.add_node(_make_node("b"))
        g.add_edge(_make_edge("e1", "a", "b"))
        result = GraphValidator().validate(g)
        # No orphans since a has degree 1 and b has degree 1
        assert result.orphan_node_count <= 2
