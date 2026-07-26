"""Tests that graph builder never modifies pipeline assets."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import pytest

from pain_intelligence.knowledge.metadata import write_parquet_with_metadata
from phase2.knowledge_graph.config import KnowledgeGraphConfig
from phase2.knowledge_graph.engine import KnowledgeGraphEngine
from phase2.knowledge_graph.schema import NodeType


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


@pytest.fixture
def protected_assets(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Create pipeline asset files and record their checksums."""
    base = tmp_path / "assets"
    base.mkdir(parents=True, exist_ok=True)

    obs_data = [{"observation_id": "obs1", "text": "Test observation", "confidence": 0.9}]
    df = pl.DataFrame(obs_data)
    write_parquet_with_metadata(df, str(base / "observations.parquet"), {"run_id": "test-run"})

    checksums = {
        "observations.parquet": _checksum(base / "observations.parquet"),
    }
    return base, checksums


class TestNoModification:
    def test_build_does_not_modify_assets(self, protected_assets):
        base, checksums = protected_assets
        config = KnowledgeGraphConfig(
            output_dir=base,
            node_types=[NodeType.OBSERVATION],
            edge_types=[],
        )
        engine = KnowledgeGraphEngine(config)
        engine.build(force=True)
        # Check asset unchanged
        assert _checksum(base / "observations.parquet") == checksums["observations.parquet"]

    def test_no_new_files_in_asset_dir(self, protected_assets):
        base, _ = protected_assets
        before = set(base.iterdir())
        config = KnowledgeGraphConfig(
            output_dir=base,
            node_types=[NodeType.OBSERVATION],
            edge_types=[],
        )
        engine = KnowledgeGraphEngine(config)
        engine.build(force=True)
        after = set(base.iterdir())
        # New files only in knowledge_graph/ subdirectory
        new_files = after - before
        # The knowledge_graph/ directory itself is new
        assert any(p.is_dir() and p.name == "knowledge_graph" for p in new_files)

    def test_graph_output_separate_directory(self, protected_assets):
        base, _ = protected_assets
        config = KnowledgeGraphConfig(
            output_dir=base,
            node_types=[NodeType.OBSERVATION],
            edge_types=[],
        )
        engine = KnowledgeGraphEngine(config)
        engine.build(force=True)
        # Graph files should be in knowledge_graph/ subdirectory
        graph_dir = engine.store.get_graph_dir()
        assert graph_dir.exists()
        assert (graph_dir / "kg_nodes.parquet").exists()
        assert (graph_dir / "kg_edges.parquet").exists()
