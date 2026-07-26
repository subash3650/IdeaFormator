"""Tests for knowledge graph configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from phase2.knowledge_graph.config import KnowledgeGraphConfig, load_knowledge_graph_config
from phase2.knowledge_graph.schema import EdgeType, NodeType


class TestKnowledgeGraphConfig:
    def test_defaults(self):
        cfg = KnowledgeGraphConfig(output_dir=Path("/tmp"))
        assert cfg.minimum_confidence == 0.5
        assert cfg.minimum_weight == 0.1
        assert cfg.include_isolated_nodes is True
        assert cfg.deterministic is True
        assert cfg.version == "1.0"

    def test_default_node_types(self):
        cfg = KnowledgeGraphConfig(output_dir=Path("/tmp"))
        assert NodeType.DOCUMENT in cfg.node_types
        assert NodeType.OBSERVATION in cfg.node_types
        assert NodeType.CLUSTER in cfg.node_types

    def test_default_edge_types(self):
        cfg = KnowledgeGraphConfig(output_dir=Path("/tmp"))
        assert EdgeType.SIMILAR_TO in cfg.edge_types
        assert EdgeType.CAUSES in cfg.edge_types

    def test_frozen(self):
        cfg = KnowledgeGraphConfig(output_dir=Path("/tmp"))
        with pytest.raises(ValidationError):
            cfg.minimum_confidence = 0.8

    def test_forbidden_extra(self):
        with pytest.raises(ValidationError):
            KnowledgeGraphConfig(output_dir=Path("/tmp"), unknown_field="nope")

    def test_graph_dir_property(self):
        cfg = KnowledgeGraphConfig(output_dir=Path("/tmp"))
        assert cfg.graph_dir == Path("/tmp/knowledge_graph")

    def test_graph_dir_with_knowledge_dir(self):
        cfg = KnowledgeGraphConfig(output_dir=Path("/tmp"), knowledge_dir=Path("/custom"))
        assert cfg.graph_dir == Path("/custom/knowledge_graph")

    def test_confidence_range(self):
        with pytest.raises(ValidationError):
            KnowledgeGraphConfig(output_dir=Path("/tmp"), minimum_confidence=1.5)
        with pytest.raises(ValidationError):
            KnowledgeGraphConfig(output_dir=Path("/tmp"), minimum_confidence=-0.1)

    def test_custom_node_types(self):
        cfg = KnowledgeGraphConfig(
            output_dir=Path("/tmp"),
            node_types=[NodeType.OBSERVATION, NodeType.PROBLEM_SIGNAL],
        )
        assert len(cfg.node_types) == 2

    def test_load_from_nonexistent_file(self):
        cfg = load_knowledge_graph_config("/nonexistent/path.yaml")
        assert isinstance(cfg, KnowledgeGraphConfig)
        assert cfg.minimum_confidence == 0.5
