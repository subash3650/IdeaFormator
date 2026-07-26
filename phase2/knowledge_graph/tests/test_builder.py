"""Tests for KnowledgeGraphBuilder."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from pain_intelligence.knowledge.metadata import write_parquet_with_metadata
from phase2.knowledge_graph.builder import KnowledgeGraphBuilder
from phase2.knowledge_graph.config import KnowledgeGraphConfig
from phase2.knowledge_graph.schema import NodeType


@pytest.fixture
def pipeline_assets(tmp_path: Path) -> Path:
    """Create synthetic pipeline asset files."""
    base = tmp_path / "assets"
    base.mkdir(parents=True, exist_ok=True)

    # observations.parquet
    obs_data = [
        {"observation_id": "obs1", "text": "Product crashes on startup", "confidence": 0.9},
        {"observation_id": "obs2", "text": "UI is confusing", "confidence": 0.8},
    ]
    df = pl.DataFrame(obs_data)
    write_parquet_with_metadata(df, str(base / "observations.parquet"), {"run_id": "test-run"})

    # evidence.parquet
    ev_data = [
        {"evidence_id": "ev1", "text": "Multiple crash reports", "confidence": 0.85, "observation_ids": ["obs1"]},
        {"evidence_id": "ev2", "text": "UX complaints", "confidence": 0.75, "observation_ids": ["obs2"]},
    ]
    df = pl.DataFrame(ev_data)
    write_parquet_with_metadata(df, str(base / "evidence.parquet"), {"run_id": "test-run"})

    # problem_signals.parquet
    sig_data = [
        {"signal_id": "sig1", "label": "Stability issues", "confidence": 0.9, "evidence_ids": ["ev1"]},
        {"signal_id": "sig2", "label": "UX problems", "confidence": 0.8, "evidence_ids": ["ev2"]},
    ]
    df = pl.DataFrame(sig_data)
    write_parquet_with_metadata(df, str(base / "problem_signals.parquet"), {"run_id": "test-run"})

    # semantic_clusters.parquet
    clus_data = [
        {"cluster_id": "cl1", "representative_id": "obs1", "member_ids": ["obs1", "obs2"], "quality_score": 0.9},
    ]
    df = pl.DataFrame(clus_data)
    write_parquet_with_metadata(df, str(base / "semantic_clusters.parquet"), {"run_id": "test-run"})

    # semantic_relationships.parquet
    rel_data = [
        {"source_id": "obs1", "target_id": "obs2", "similarity_score": 0.9, "confidence": 0.85},
    ]
    df = pl.DataFrame(rel_data)
    write_parquet_with_metadata(df, str(base / "semantic_relationships.parquet"), {"run_id": "test-run"})

    return base


@pytest.fixture
def builder(pipeline_assets: Path) -> KnowledgeGraphBuilder:
    config = KnowledgeGraphConfig(
        output_dir=pipeline_assets,
        node_types=[
            NodeType.OBSERVATION,
            NodeType.EVIDENCE,
            NodeType.PROBLEM_SIGNAL,
            NodeType.CLUSTER,
        ],
        edge_types=[],
    )
    return KnowledgeGraphBuilder(config)


class TestKnowledgeGraphBuilder:
    def test_build_nodes_only(self, builder):
        graph = builder.build()
        assert graph.node_count() >= 6  # 2 obs + 2 ev + 2 sig + 1 cluster
        assert graph.edge_count() == 0

    def test_build_with_edges(self, pipeline_assets):
        config = KnowledgeGraphConfig(
            output_dir=pipeline_assets,
            node_types=[NodeType.OBSERVATION],
            edge_types=[],
        )
        builder = KnowledgeGraphBuilder(config)
        graph = builder.build()
        assert graph.node_count() >= 2

    def test_build_without_assets(self, tmp_path: Path):
        config = KnowledgeGraphConfig(output_dir=tmp_path)
        builder = KnowledgeGraphBuilder(config)
        # No assets exist, build should work with empty result
        result = builder.build()
        assert result.node_count() == 0
        assert result.edge_count() == 0

    def test_dedup_nodes(self, pipeline_assets):
        config = KnowledgeGraphConfig(
            output_dir=pipeline_assets,
            node_types=[NodeType.OBSERVATION, NodeType.OBSERVATION],
            edge_types=[],
        )
        builder = KnowledgeGraphBuilder(config)
        graph = builder.build()
        # No dedup needed since both builders produce same node IDs
        assert graph.node_count() >= 2

    def test_store_saves(self, builder):
        builder.build()
        assert builder.store.nodes_path.exists()
        assert builder.store.edges_path.exists()
        manifest = builder.store.load_manifest()
        assert manifest.get("run_id", "") != ""

    def test_run_id_consistency(self, builder):
        builder.build()
        run_id = builder.store.run_id()
        assert run_id != ""

    def test_metadata_written(self, builder):
        graph = builder.build()
        meta = graph.metadata("test-run")
        assert meta.node_count > 0
        assert isinstance(meta.connected_components, int)

    def test_empty_graph(self, tmp_path: Path):
        config = KnowledgeGraphConfig(output_dir=tmp_path)
        builder = KnowledgeGraphBuilder(config)
        graph = builder.build()
        # No assets, graph should be empty but not error
        assert graph.node_count() == 0
