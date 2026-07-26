"""Tests for ReasoningConfig."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from phase2.reasoning.config import ReasoningConfig, load_reasoning_config
from phase2.reasoning.schema import ExplanationFormat, PropagationStrategy, RootCauseRanking


class TestReasoningConfig:
    def test_default_output_dir_required(self) -> None:
        cfg = ReasoningConfig(output_dir=Path("/tmp/test"))
        assert cfg.output_dir == Path("/tmp/test")
        assert cfg.enabled_rules == ["transitive_closure", "causal_chain", "evidence_convergence"]
        assert cfg.max_chain_length == 8
        assert cfg.max_inferences_per_run == 10000
        assert cfg.max_rule_iterations == 5
        assert cfg.min_confidence == 0.15

    def test_defaults(self) -> None:
        cfg = ReasoningConfig(output_dir=Path("/tmp/test"))
        assert cfg.propagation_strategy == PropagationStrategy.MULTIPLICATIVE
        assert cfg.root_cause_ranking == RootCauseRanking.TRANSITIVE_IMPACT
        assert cfg.explanation_format == ExplanationFormat.TEMPLATE
        assert cfg.collapse_chains_longer_than == 4
        assert cfg.cache_enabled is True

    def test_reasoning_dir(self) -> None:
        cfg = ReasoningConfig(output_dir=Path("/tmp/test"))
        assert cfg.reasoning_dir == Path("/tmp/test") / "reasoning"

    def test_reasoning_dir_with_knowledge(self) -> None:
        cfg = ReasoningConfig(
            output_dir=Path("/tmp/test"),
            knowledge_dir=Path("/tmp/kb"),
        )
        assert cfg.reasoning_dir == Path("/tmp/kb") / "reasoning"

    def test_frozen(self) -> None:
        cfg = ReasoningConfig(output_dir=Path("/tmp/test"))
        with pytest.raises(ValidationError):
            cfg.output_dir = Path("/other")

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            ReasoningConfig(output_dir=Path("/tmp/test"), unknown="x")

    def test_min_confidence_range(self) -> None:
        with pytest.raises(ValidationError):
            ReasoningConfig(output_dir=Path("/tmp/test"), min_confidence=1.5)

    def test_max_inferences_positive(self) -> None:
        with pytest.raises(ValidationError):
            ReasoningConfig(output_dir=Path("/tmp/test"), max_inferences_per_run=0)

    def test_max_chain_length_range(self) -> None:
        with pytest.raises(ValidationError):
            ReasoningConfig(output_dir=Path("/tmp/test"), max_chain_length=0)


class TestLoadReasoningConfig:
    def test_missing_file_uses_defaults(self) -> None:
        cfg = load_reasoning_config("/nonexistent/config.yaml")
        assert cfg.output_dir == Path("pain_intelligence/knowledge/assets/phase2")
        assert cfg.enabled_rules == ["transitive_closure", "causal_chain", "evidence_convergence"]

    def test_with_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_data = {
            "reasoning": {
                "output_dir": str(tmp_path / "output"),
                "propagation_strategy": "minimum",
                "max_chain_length": 5,
                "min_confidence": 0.2,
            }
        }
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")
        cfg = load_reasoning_config(str(config_file))
        assert cfg.output_dir == tmp_path / "output"
        assert cfg.propagation_strategy == PropagationStrategy.MINIMUM
        assert cfg.max_chain_length == 5
        assert cfg.min_confidence == 0.2

    def test_empty_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("{}", encoding="utf-8")
        cfg = load_reasoning_config(str(config_file))
        assert cfg.output_dir == Path("pain_intelligence/knowledge/assets/phase2")

    def test_enum_coercion(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_data = {
            "reasoning": {
                "output_dir": str(tmp_path),
                "root_cause_ranking": "confidence",
                "explanation_format": "markdown",
            }
        }
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")
        cfg = load_reasoning_config(str(config_file))
        assert cfg.root_cause_ranking == RootCauseRanking.CONFIDENCE
        assert cfg.explanation_format == ExplanationFormat.MARKDOWN
