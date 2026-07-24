"""Tests for configuration models."""

from __future__ import annotations

from pathlib import Path

from phase2.embeddings.schema import SourceType
from phase2.similarity.config import SimilarityEngineConfig, load_similarity_config


class TestSimilarityEngineConfig:
    def test_defaults(self) -> None:
        cfg = SimilarityEngineConfig()
        assert cfg.metric == "cosine"
        assert cfg.similarity_threshold == 0.82
        assert cfg.top_k == 20
        assert cfg.batch_size == 1024
        assert cfg.normalize_scores is True
        assert cfg.minimum_confidence == 0.0
        assert cfg.store_bidirectional is False
        assert cfg.version == "1.0"

    def test_model_fingerprint(self) -> None:
        cfg = SimilarityEngineConfig()
        fp = cfg.model_fingerprint
        assert fp == "sentence_transformers/all-MiniLM-L6-v2@384d"

    def test_forbids_extra_fields(self) -> None:
        try:
            SimilarityEngineConfig(unknown_field="bad")  # type: ignore
            assert False, "Should have raised"
        except Exception:
            pass

    def test_frozen(self) -> None:
        cfg = SimilarityEngineConfig()
        try:
            cfg.metric = "dot_product"  # type: ignore
            assert False, "Should have raised"
        except Exception:
            pass

    def test_allowed_relationships_default(self) -> None:
        cfg = SimilarityEngineConfig()
        assert SourceType.observation in cfg.allowed_relationships
        assert SourceType.observation in cfg.allowed_relationships[SourceType.observation]


class TestLoadSimilarityConfig:
    def test_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        cfg = load_similarity_config(tmp_path / "nonexistent.yaml")
        assert cfg.metric == "cosine"

    def test_loads_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
similarity:
  metric: dot_product
  similarity_threshold: 0.90
  top_k: 10
"""
        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml_content, encoding="utf-8")
        cfg = load_similarity_config(config_path)
        assert cfg.metric == "dot_product"
        assert cfg.similarity_threshold == 0.90
        assert cfg.top_k == 10

    def test_loads_with_source_paths(self, tmp_path: Path) -> None:
        yaml_content = """
similarity:
  source_paths:
    observation: "/path/to/obs.parquet"
"""
        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml_content, encoding="utf-8")
        cfg = load_similarity_config(config_path)
        assert cfg.source_paths[SourceType.observation] == Path("/path/to/obs.parquet")

    def test_loads_with_allowed_relationships(self, tmp_path: Path) -> None:
        yaml_content = """
similarity:
  allowed_relationships:
    observation:
      - observation
    evidence:
      - evidence
      - problem_signal
"""
        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml_content, encoding="utf-8")
        cfg = load_similarity_config(config_path)
        assert cfg.allowed_relationships[SourceType.observation] == [SourceType.observation]
        assert cfg.allowed_relationships[SourceType.evidence] == [
            SourceType.evidence,
            SourceType.problem_signal,
        ]
