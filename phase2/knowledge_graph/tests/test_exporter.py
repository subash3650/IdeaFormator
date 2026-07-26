"""Tests for GraphExporter."""

from __future__ import annotations

from pathlib import Path

import pytest

from phase2.knowledge_graph.graph import CustomGraph
from phase2.knowledge_graph.schema import EdgeType, GraphEdge, GraphNode, NodeType
from phase2.knowledge_graph.store import KnowledgeGraphStore
from phase2.knowledge_graph.exporter import GraphExporter
from phase2.knowledge_graph.evaluator import GraphEvaluator


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


@pytest.fixture
def graph_with_data() -> CustomGraph:
    g = CustomGraph()
    g.add_node(_make_node("a", NodeType.OBSERVATION))
    g.add_node(_make_node("b", NodeType.EVIDENCE))
    g.add_edge(_make_edge("e1", "a", "b"))
    return g


class TestGraphExporter:
    def test_export_gexf(self, graph_with_data, tmp_path: Path):
        exporter = GraphExporter(graph_with_data, KnowledgeGraphStore(tmp_path))
        path = exporter.export_gexf(tmp_path / "graph.gexf")
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "<gexf" in content
        assert "node_id" in content or "id=" in content

    def test_export_json(self, graph_with_data, tmp_path: Path):
        exporter = GraphExporter(graph_with_data, KnowledgeGraphStore(tmp_path))
        path = exporter.export_json(tmp_path / "graph.json")
        assert path.exists()
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["node_count"] == 2
        assert data["edge_count"] == 1

    def test_export_csv(self, graph_with_data, tmp_path: Path):
        exporter = GraphExporter(graph_with_data, KnowledgeGraphStore(tmp_path))
        paths = exporter.export_csv(tmp_path)
        assert paths["nodes"].exists()
        assert paths["edges"].exists()
        nodes_content = paths["nodes"].read_text(encoding="utf-8")
        assert "node_id" in nodes_content
        assert "a" in nodes_content

    def test_export_statistics(self, graph_with_data, tmp_path: Path):
        exporter = GraphExporter(graph_with_data, KnowledgeGraphStore(tmp_path))
        path = exporter.export_statistics(tmp_path / "statistics.json")
        assert path.exists()
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "node_count" in data
        assert "edge_count" in data

    def test_export_dashboard(self, graph_with_data, tmp_path: Path):
        exporter = GraphExporter(graph_with_data, KnowledgeGraphStore(tmp_path), GraphEvaluator())
        paths = exporter.export_dashboard(tmp_path / "dashboard")
        assert "json" in paths
        assert "txt" in paths
        txt = paths["txt"].read_text(encoding="utf-8")
        assert "KNOWLEDGE GRAPH DASHBOARD" in txt

    def test_export_summary(self, graph_with_data, tmp_path: Path):
        exporter = GraphExporter(graph_with_data, KnowledgeGraphStore(tmp_path), GraphEvaluator())
        path = exporter.export_summary(tmp_path / "summary.md")
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Knowledge Graph Summary" in content
        assert "Node" in content

    def test_export_report(self, graph_with_data, tmp_path: Path):
        exporter = GraphExporter(graph_with_data, KnowledgeGraphStore(tmp_path), GraphEvaluator())
        path = exporter.export_report(tmp_path / "report.json")
        assert path.exists()
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "node_count" in data
        assert "edge_count" in data

    def test_export_empty_graph(self, tmp_path: Path):
        g = CustomGraph()
        exporter = GraphExporter(g, KnowledgeGraphStore(tmp_path), GraphEvaluator())
        path = exporter.export_json(tmp_path / "empty.json")
        data = __import__("json").loads(path.read_text(encoding="utf-8"))
        assert data["node_count"] == 0
