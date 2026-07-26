"""Tests for KnowledgeGraphStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from phase2.knowledge_graph.schema import EdgeType, GraphEdge, GraphMetadata, GraphNode, NodeType
from phase2.knowledge_graph.store import KnowledgeGraphStore


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
def store(tmp_path: Path) -> KnowledgeGraphStore:
    return KnowledgeGraphStore(tmp_path)


@pytest.fixture
def sample_nodes() -> list[GraphNode]:
    return [_make_node("a"), _make_node("b"), _make_node("c")]


@pytest.fixture
def sample_edges() -> list[GraphEdge]:
    return [_make_edge("e1", "a", "b"), _make_edge("e2", "b", "c")]


class TestKnowledgeGraphStore:
    def test_save_nodes(self, store, sample_nodes):
        path = store.save_nodes(sample_nodes, "run1")
        assert path.exists()
        assert store.nodes_path.exists()

    def test_load_nodes(self, store, sample_nodes):
        store.save_nodes(sample_nodes, "run1")
        loaded = store.load_nodes()
        assert len(loaded) == 3
        assert loaded[0].node_id == "a"

    def test_save_edges(self, store, sample_edges):
        path = store.save_edges(sample_edges, "run1")
        assert path.exists()

    def test_load_edges(self, store, sample_edges):
        store.save_edges(sample_edges, "run1")
        loaded = store.load_edges()
        assert len(loaded) == 2
        assert loaded[0].edge_id == "e1"

    def test_overwrite_nodes(self, store):
        store.save_nodes([_make_node("a")], "run1")
        store.save_nodes([_make_node("b")], "run2")
        loaded = store.load_nodes()
        assert len(loaded) == 1
        assert loaded[0].node_id == "b"

    def test_empty_nodes(self, store):
        store.save_nodes([], "run1")
        loaded = store.load_nodes()
        assert loaded == []

    def test_empty_edges(self, store):
        store.save_edges([], "run1")
        loaded = store.load_edges()
        assert loaded == []

    def test_metadata_embedding(self, store, sample_nodes):
        store.save_nodes(sample_nodes, "run1", input_checksum="abc123")
        meta = store.run_id()
        assert meta == "run1"

    def test_save_load_metadata(self, store):
        meta = GraphMetadata(
            graph_id="g1", node_count=5, edge_count=10,
            connected_components=1, largest_component_size=5,
            density=0.5, avg_confidence=0.9, avg_degree=2.0,
            orphan_node_count=1, run_id="run1",
            pipeline_version="1.0", schema_version="1.0",
        )
        store.save_metadata(meta)
        loaded = store.load_metadata()
        assert loaded is not None
        assert loaded.node_count == 5
        assert loaded.edge_count == 10

    def test_exists(self, store, sample_nodes):
        assert store.exists() is False
        store.save_nodes(sample_nodes, "run1")
        assert store.exists() is False  # edges not saved yet
        store.save_edges([_make_edge("e1", "a", "b")], "run1")
        assert store.exists() is True

    def test_count(self, store, sample_nodes, sample_edges):
        store.save_nodes(sample_nodes, "run1")
        store.save_edges(sample_edges, "run1")
        counts = store.count()
        assert counts["nodes"] == 3
        assert counts["edges"] == 2

    def test_checksums(self, store, sample_nodes):
        store.save_nodes(sample_nodes, "run1")
        checksums = store.checksums()
        assert "kg_nodes.parquet" in checksums
        assert len(checksums["kg_nodes.parquet"]) == 16

    def test_manifest_io(self, store):
        manifest = {"run_id": "run1", "node_count": 5}
        store.save_manifest(manifest)
        loaded = store.load_manifest()
        assert loaded["run_id"] == "run1"

    def test_missing_metadata(self, store):
        assert store.load_metadata() is None

    def test_load_df(self, store, sample_nodes):
        store.save_nodes(sample_nodes, "run1")
        df = store.load_nodes_df()
        assert df.height == 3
        assert df.width == 12  # all columns
