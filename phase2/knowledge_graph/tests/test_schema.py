"""Tests for knowledge graph schema models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from phase2.knowledge_graph.schema import (
    EdgeType,
    GraphEdge,
    GraphMetadata,
    GraphNode,
    NodeType,
    ValidationResult,
)


def make_node(overrides: dict | None = None) -> dict:
    base = {
        "node_id": "abc123",
        "node_type": NodeType.OBSERVATION,
        "label": "Test observation",
        "source_asset": "observations.parquet",
        "source_id": "123",
        "confidence": 0.95,
        "pipeline_version": "1.5.0",
        "schema_version": "1.0.0",
    }
    if overrides:
        base.update(overrides)
    return base


def make_edge(overrides: dict | None = None) -> dict:
    base = {
        "edge_id": "def456",
        "source_node_id": "abc123",
        "target_node_id": "xyz789",
        "edge_type": EdgeType.SIMILAR_TO,
        "weight": 0.85,
        "confidence": 0.9,
        "source_asset": "semantic_relationships.parquet",
        "pipeline_version": "1.5.0",
        "schema_version": "1.0.0",
    }
    if overrides:
        base.update(overrides)
    return base


class TestNodeType:
    def test_enum_values(self):
        assert NodeType.OBSERVATION.value == "observation"
        assert NodeType.EVIDENCE.value == "evidence"
        assert NodeType.PROBLEM_SIGNAL.value == "problem_signal"
        assert NodeType.CLUSTER.value == "cluster"

    def test_all_values_unique(self):
        values = [t.value for t in NodeType]
        assert len(values) == len(set(values))

    def test_total_types(self):
        assert len(NodeType) == 17


class TestEdgeType:
    def test_enum_values(self):
        assert EdgeType.SIMILAR_TO.value == "similar_to"
        assert EdgeType.CAUSES.value == "causes"
        assert EdgeType.MEMBER_OF_CLUSTER.value == "member_of_cluster"

    def test_all_values_unique(self):
        values = [t.value for t in EdgeType]
        assert len(values) == len(set(values))

    def test_total_types(self):
        assert len(EdgeType) == 17


class TestGraphNode:
    def test_create_valid(self):
        node = GraphNode(**make_node())
        assert node.node_id == "abc123"
        assert node.node_type == NodeType.OBSERVATION
        assert node.confidence == 0.95

    def test_frozen(self):
        node = GraphNode(**make_node())
        with pytest.raises(ValidationError):
            node.confidence = 0.5

    def test_forbidden_extra(self):
        with pytest.raises(ValidationError):
            GraphNode(**make_node({"extra_field": "nope"}))

    def test_confidence_range(self):
        with pytest.raises(ValidationError):
            GraphNode(**make_node({"confidence": 1.5}))
        with pytest.raises(ValidationError):
            GraphNode(**make_node({"confidence": -0.1}))

    def test_defaults(self):
        node = GraphNode(**make_node())
        assert isinstance(node.created_at, str)
        assert node.properties == {}
        assert node.metadata == {}
        assert node.attributes == {}

    def test_serialization_roundtrip(self):
        node = GraphNode(**make_node({"properties": {"key": "val"}, "metadata": {"src": "test"}}))
        data = node.model_dump(mode="json")
        restored = GraphNode(**data)
        assert restored.node_id == node.node_id
        assert restored.properties == node.properties
        assert restored.metadata == node.metadata


class TestGraphEdge:
    def test_create_valid(self):
        edge = GraphEdge(**make_edge())
        assert edge.edge_id == "def456"
        assert edge.edge_type == EdgeType.SIMILAR_TO

    def test_frozen(self):
        edge = GraphEdge(**make_edge())
        with pytest.raises(ValidationError):
            edge.weight = 0.5

    def test_forbidden_extra(self):
        with pytest.raises(ValidationError):
            GraphEdge(**make_edge({"extra_fields": "nope"}))

    def test_weight_range(self):
        with pytest.raises(ValidationError):
            GraphEdge(**make_edge({"weight": 1.5}))
        with pytest.raises(ValidationError):
            GraphEdge(**make_edge({"weight": -0.1}))

    def test_no_self_loop_required(self):
        edge = GraphEdge(**make_edge({"source_node_id": "same", "target_node_id": "same"}))
        assert edge.source_node_id == edge.target_node_id

    def test_serialization_roundtrip(self):
        edge = GraphEdge(**make_edge({"properties": {"sim": 0.9}}))
        data = edge.model_dump(mode="json")
        restored = GraphEdge(**data)
        assert restored.edge_id == edge.edge_id
        assert restored.properties == edge.properties


class TestGraphMetadata:
    def test_create_valid(self):
        meta = GraphMetadata(graph_id="g1", run_id="r1", pipeline_version="1.0", schema_version="1.0")
        assert meta.graph_id == "g1"
        assert meta.node_count == 0

    def test_frozen(self):
        meta = GraphMetadata(graph_id="g1", run_id="r1", pipeline_version="1.0", schema_version="1.0")
        with pytest.raises(ValidationError):
            meta.node_count = 10

    def test_node_count_default(self):
        meta = GraphMetadata(graph_id="g1", run_id="r1", pipeline_version="1.0", schema_version="1.0")
        assert meta.node_count == 0
        assert meta.edge_count == 0
        assert meta.connected_components == 0

    def test_node_count_non_negative(self):
        with pytest.raises(ValidationError):
            GraphMetadata(
                graph_id="g1", node_count=-1, run_id="r1",
                pipeline_version="1.0", schema_version="1.0",
            )


class TestValidationResult:
    def test_create_valid(self):
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_frozen(self):
        result = ValidationResult(valid=True)
        with pytest.raises(ValidationError):
            result.valid = False

    def test_default_zero_counts(self):
        result = ValidationResult(valid=True)
        assert result.node_count == 0
        assert result.edge_count == 0
        assert result.duplicate_node_count == 0
