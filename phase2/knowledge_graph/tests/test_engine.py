"""Tests for KnowledgeGraphEngine."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pain_intelligence.knowledge.metadata import write_parquet_with_metadata
from phase2.knowledge_graph.config import KnowledgeGraphConfig
from phase2.knowledge_graph.engine import KnowledgeGraphEngine
from phase2.knowledge_graph.schema import NodeType


@pytest.fixture
def pipeline_assets(tmp_path: Path) -> Path:
    base = tmp_path / "assets"
    base.mkdir(parents=True, exist_ok=True)
    obs_data = [
        {"observation_id": "obs1", "text": "Product crashes", "confidence": 0.9},
        {"observation_id": "obs2", "text": "UI confusion", "confidence": 0.8},
    ]
    df = pl.DataFrame(obs_data)
    write_parquet_with_metadata(df, str(base / "observations.parquet"), {"run_id": "test-run"})
    ev_data = [
        {"evidence_id": "ev1", "text": "Crash reports", "confidence": 0.85, "observation_ids": ["obs1"]},
    ]
    df = pl.DataFrame(ev_data)
    write_parquet_with_metadata(df, str(base / "evidence.parquet"), {"run_id": "test-run"})
    sig_data = [
        {"signal_id": "sig1", "label": "Stability issues", "confidence": 0.9, "evidence_ids": ["ev1"]},
    ]
    df = pl.DataFrame(sig_data)
    write_parquet_with_metadata(df, str(base / "problem_signals.parquet"), {"run_id": "test-run"})
    clus_data = [
        {"cluster_id": "cl1", "representative_id": "obs1", "member_ids": ["obs1", "obs2"], "quality_score": 0.9},
    ]
    df = pl.DataFrame(clus_data)
    write_parquet_with_metadata(df, str(base / "semantic_clusters.parquet"), {"run_id": "test-run"})
    rel_data = [
        {"source_id": "obs1", "target_id": "obs2", "similarity_score": 0.9, "confidence": 0.85},
    ]
    df = pl.DataFrame(rel_data)
    write_parquet_with_metadata(df, str(base / "semantic_relationships.parquet"), {"run_id": "test-run"})
    return base


class TestKnowledgeGraphEngine:
    def test_build(self, pipeline_assets):
        config = KnowledgeGraphConfig(
            output_dir=pipeline_assets,
            node_types=[NodeType.OBSERVATION, NodeType.EVIDENCE],
            edge_types=[],
        )
        engine = KnowledgeGraphEngine(config)
        result = engine.build()
        assert result["node_count"] >= 2
        assert result["edge_count"] >= 0
        assert result["valid"] is not None

    def test_build_with_edges(self, pipeline_assets):
        config = KnowledgeGraphConfig(
            output_dir=pipeline_assets,
            node_types=[NodeType.OBSERVATION],
            edge_types=[],
        )
        engine = KnowledgeGraphEngine(config)
        result = engine.build()
        assert "health_score" in result

    def test_build_empty(self, tmp_path: Path):
        config = KnowledgeGraphConfig(output_dir=tmp_path)
        engine = KnowledgeGraphEngine(config)
        result = engine.build()
        assert isinstance(result, dict)

    def test_stats(self, pipeline_assets):
        config = KnowledgeGraphConfig(
            output_dir=pipeline_assets,
            node_types=[NodeType.OBSERVATION],
            edge_types=[],
        )
        engine = KnowledgeGraphEngine(config)
        engine.build()
        stats = engine.stats()
        assert "node_count" in stats
        assert "edge_count" in stats

    def test_verify(self, pipeline_assets):
        config = KnowledgeGraphConfig(
            output_dir=pipeline_assets,
            node_types=[NodeType.OBSERVATION],
            edge_types=[],
        )
        engine = KnowledgeGraphEngine(config)
        engine.build()
        result = engine.verify()
        assert "valid" in result
        assert "node_count" in result

    def test_verify_without_build(self, tmp_path: Path):
        config = KnowledgeGraphConfig(output_dir=tmp_path)
        engine = KnowledgeGraphEngine(config)
        result = engine.verify()
        assert result["node_count"] == 0

    def test_store_exists(self, pipeline_assets):
        config = KnowledgeGraphConfig(
            output_dir=pipeline_assets,
            node_types=[NodeType.OBSERVATION],
            edge_types=[],
        )
        engine = KnowledgeGraphEngine(config)
        engine.build()
        assert engine.store.exists()

    def test_search_by_id(self, pipeline_assets):
        config = KnowledgeGraphConfig(
            output_dir=pipeline_assets,
            node_types=[NodeType.OBSERVATION],
            edge_types=[],
        )
        engine = KnowledgeGraphEngine(config)
        engine.build()
        results = engine.search("obs1")
        assert len(results) >= 1

    def test_search_by_label(self, pipeline_assets):
        config = KnowledgeGraphConfig(
            output_dir=pipeline_assets,
            node_types=[NodeType.OBSERVATION],
            edge_types=[],
        )
        engine = KnowledgeGraphEngine(config)
        engine.build()
        results = engine.search("crashes")
        assert len(results) >= 1

    def test_export_all(self, pipeline_assets):
        config = KnowledgeGraphConfig(
            output_dir=pipeline_assets,
            node_types=[NodeType.OBSERVATION],
            edge_types=[],
        )
        engine = KnowledgeGraphEngine(config)
        engine.build()
        result = engine.export(format="all")
        assert "report" in result
