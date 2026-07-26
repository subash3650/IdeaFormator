"""Tests for TrendConfig and YAML loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from phase3.trend.config import TrendConfig, load_trend_config


class TestTrendConfig:
    def test_default_output_dir_required(self) -> None:
        with pytest.raises(ValidationError):
            TrendConfig()

    def test_defaults(self, tmp_path: Path) -> None:
        cfg = TrendConfig(output_dir=tmp_path)
        assert cfg.min_growth_pct == 5.0
        assert cfg.min_confidence == 0.3
        assert cfg.min_snapshots == 2
        assert cfg.top_k == 20
        assert cfg.cache_enabled is True
        assert cfg.comparison_window == 1
        assert cfg.version == "1.0"

    def test_trend_dir(self, tmp_path: Path) -> None:
        cfg = TrendConfig(output_dir=tmp_path)
        assert cfg.trend_dir == tmp_path / "trend"

    def test_trend_dir_with_knowledge(self, tmp_path: Path) -> None:
        cfg = TrendConfig(output_dir=tmp_path, knowledge_dir=Path("/custom/knowledge"))
        assert cfg.trend_dir == Path("/custom/knowledge") / "trend"

    def test_frozen(self, tmp_path: Path) -> None:
        cfg = TrendConfig(output_dir=tmp_path)
        with pytest.raises(ValidationError):
            cfg.top_k = 50

    def test_extra_forbid(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            TrendConfig(output_dir=tmp_path, unknown_field=True)

    def test_score_weights(self, tmp_path: Path) -> None:
        cfg = TrendConfig(output_dir=tmp_path)
        from phase3.trend.schema import TrendScoreWeights
        assert isinstance(cfg.score_weights, TrendScoreWeights)
        assert cfg.score_weights.growth == 0.30

    def test_bounds(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            TrendConfig(output_dir=tmp_path, top_k=0)
        with pytest.raises(ValidationError):
            TrendConfig(output_dir=tmp_path, top_k=501)
        with pytest.raises(ValidationError):
            TrendConfig(output_dir=tmp_path, min_snapshots=0)

    def test_enabled_providers(self, tmp_path: Path) -> None:
        cfg = TrendConfig(output_dir=tmp_path, enabled_scoring_providers=["growth"])
        assert "growth" in cfg.enabled_scoring_providers


class TestLoadTrendConfig:
    def test_missing_file_uses_defaults(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        cfg = load_trend_config(missing)
        assert cfg.output_dir == Path("pain_intelligence/knowledge/assets/phase3")

    def test_with_yaml(self, tmp_path: Path) -> None:
        yml = tmp_path / "config.yaml"
        yml.write_text(
            "trend:\n"
            "  output_dir: /tmp/trend\n"
            "  top_k: 50\n"
            "  min_growth_pct: 10.0\n",
            encoding="utf-8",
        )
        cfg = load_trend_config(yml)
        assert cfg.top_k == 50
        assert cfg.min_growth_pct == 10.0

    def test_empty_yaml(self, tmp_path: Path) -> None:
        yml = tmp_path / "empty.yaml"
        yml.write_text("", encoding="utf-8")
        cfg = load_trend_config(yml)
        assert cfg.top_k == 20

    def test_without_trend_section(self, tmp_path: Path) -> None:
        yml = tmp_path / "config.yaml"
        yml.write_text("other:\n  key: value\n", encoding="utf-8")
        cfg = load_trend_config(yml)
        assert cfg.top_k == 20

    def test_comparison_window(self, tmp_path: Path) -> None:
        cfg = TrendConfig(output_dir=tmp_path, comparison_window=3)
        assert cfg.comparison_window == 3
