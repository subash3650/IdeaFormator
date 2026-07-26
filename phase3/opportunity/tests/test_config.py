"""Tests for OpportunityConfig and YAML loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from phase3.opportunity.config import OpportunityConfig, load_opportunity_config
from phase3.opportunity.schema import RankingStrategy, ScoreWeights


class TestOpportunityConfig:
    def test_default_output_dir_required(self) -> None:
        with pytest.raises(ValidationError):
            OpportunityConfig()

    def test_defaults(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        assert cfg.scoring_provider == "weighted"
        assert cfg.top_k == 20
        assert cfg.ranking_strategy == RankingStrategy.COMPOSITE
        assert cfg.cache_enabled is True
        assert cfg.strong_pursue_threshold == 0.75
        assert cfg.version == "1.0"

    def test_opportunity_dir(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        assert cfg.opportunity_dir == tmp_path / "opportunity"

    def test_opportunity_dir_with_knowledge(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(
            output_dir=tmp_path,
            knowledge_dir=Path("/custom/knowledge"),
        )
        assert cfg.opportunity_dir == Path("/custom/knowledge") / "opportunity"

    def test_frozen(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        with pytest.raises(ValidationError):
            cfg.top_k = 50

    def test_extra_forbid(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            OpportunityConfig(output_dir=tmp_path, unknown_field=True)

    def test_score_weights(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        assert isinstance(cfg.score_weights, ScoreWeights)
        assert cfg.score_weights.pain_severity == 0.20

    def test_custom_score_weights(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(
            output_dir=tmp_path,
            score_weights=ScoreWeights(pain_severity=0.5, frequency=0.5),
        )
        assert cfg.score_weights.pain_severity == 0.5

    def test_bounds(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            OpportunityConfig(output_dir=tmp_path, top_k=0)
        with pytest.raises(ValidationError):
            OpportunityConfig(output_dir=tmp_path, top_k=501)

    def test_enabled_providers(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(
            output_dir=tmp_path,
            enabled_scoring_providers=["weighted"],
        )
        assert "weighted" in cfg.enabled_scoring_providers


class TestLoadOpportunityConfig:
    def test_missing_file_uses_defaults(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        cfg = load_opportunity_config(missing)
        assert cfg.output_dir == Path("pain_intelligence/knowledge/assets/phase3")

    def test_with_yaml(self, tmp_path: Path) -> None:
        yml = tmp_path / "config.yaml"
        yml.write_text(
            "opportunity:\n"
            "  output_dir: /tmp/opp\n"
            "  top_k: 50\n"
            "  strong_pursue_threshold: 0.80\n",
            encoding="utf-8",
        )
        cfg = load_opportunity_config(yml)
        assert cfg.top_k == 50
        assert cfg.strong_pursue_threshold == 0.80

    def test_empty_yaml(self, tmp_path: Path) -> None:
        yml = tmp_path / "empty.yaml"
        yml.write_text("", encoding="utf-8")
        cfg = load_opportunity_config(yml)
        assert cfg.scoring_provider == "weighted"

    def test_enum_coercion(self, tmp_path: Path) -> None:
        yml = tmp_path / "config.yaml"
        yml.write_text(
            "opportunity:\n"
            "  output_dir: /tmp/opp\n"
            "  ranking_strategy: pain_severity\n",
            encoding="utf-8",
        )
        cfg = load_opportunity_config(yml)
        assert cfg.ranking_strategy == RankingStrategy.PAIN_SEVERITY
