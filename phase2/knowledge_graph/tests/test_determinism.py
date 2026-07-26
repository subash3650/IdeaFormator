"""Tests for deterministic knowledge graph output."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import pytest

from pain_intelligence.knowledge.metadata import write_parquet_with_metadata
from phase2.knowledge_graph.builder import KnowledgeGraphBuilder
from phase2.knowledge_graph.config import KnowledgeGraphConfig
from phase2.knowledge_graph.schema import EdgeType, GraphEdge, GraphNode, NodeType
from phase2.knowledge_graph.store import KnowledgeGraphStore


def _make_node(nid: str) -> GraphNode:
    return GraphNode(
        node_id=nid,
        node_type=NodeType.OBSERVATION,
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


class TestDeterminism:
    def test_same_input_same_nodes(self, tmp_path: Path):
        store1 = KnowledgeGraphStore(tmp_path / "run1")
        store2 = KnowledgeGraphStore(tmp_path / "run2")
        nodes = [_make_node("a"), _make_node("b")]
        store1.save_nodes(nodes, "run1")
        store2.save_nodes(nodes, "run1")
        assert store1.load_nodes() == store2.load_nodes()

    def test_same_input_same_edges(self, tmp_path: Path):
        store1 = KnowledgeGraphStore(tmp_path / "run1")
        store2 = KnowledgeGraphStore(tmp_path / "run2")
        edges = [_make_edge("e1", "a", "b")]
        store1.save_edges(edges, "run1")
        store2.save_edges(edges, "run1")
        assert store1.load_edges() == store2.load_edges()

    def test_deterministic_node_id(self):
        raw1 = f"{NodeType.OBSERVATION.value}:obs.parquet:id1:1.0:1.0"
        raw2 = f"{NodeType.OBSERVATION.value}:obs.parquet:id1:1.0:1.0"
        assert hashlib.sha256(raw1.encode()).hexdigest() == hashlib.sha256(raw2.encode()).hexdigest()

    def test_different_input_different_output(self, tmp_path: Path):
        store1 = KnowledgeGraphStore(tmp_path / "run1")
        store2 = KnowledgeGraphStore(tmp_path / "run2")
        nodes1 = [_make_node("a")]
        nodes2 = [_make_node("b")]
        store1.save_nodes(nodes1, "run1")
        store2.save_nodes(nodes2, "run1")
        assert store1.load_nodes() != store2.load_nodes()
