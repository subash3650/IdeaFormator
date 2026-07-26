"""Integration tests for ReasoningEngine."""

from __future__ import annotations

from pathlib import Path

import pytest

from phase2.reasoning.config import ReasoningConfig
from phase2.reasoning.engine import ReasoningEngine
from phase2.reasoning.provenance_id import generate_run_id


class TestReasoningEngine:
    def test_reason_with_empty_graph(self, tmp_path: Path) -> None:
        cfg = ReasoningConfig(
            output_dir=tmp_path,
            enabled_rules=["transitive_closure"],
            max_inferences_per_run=100,
            max_rule_iterations=2,
            min_confidence=0.1,
            generate_explanations=False,
            cache_enabled=False,
        )
        run_id = generate_run_id()
        engine = ReasoningEngine(cfg, run_id=run_id)
        result = engine.reason(force=True)
        assert "inference_count" in result
        assert result["run_id"] == run_id

    def test_reason_return_keys(self, tmp_path: Path) -> None:
        cfg = ReasoningConfig(
            output_dir=tmp_path,
            cache_enabled=False,
            generate_explanations=False,
        )
        engine = ReasoningEngine(cfg)
        result = engine.reason(force=True)
        expected_keys = [
            "run_id", "inference_count", "chain_count",
            "root_cause_count", "evidence_aggregation_count",
            "explanation_count", "rules_applied",
            "rule_firing_counts", "elapsed_seconds", "cache_hit",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_stats(self, tmp_path: Path) -> None:
        cfg = ReasoningConfig(output_dir=tmp_path, cache_enabled=False)
        engine = ReasoningEngine(cfg)
        stats = engine.stats()
        assert "graph_nodes" in stats
        assert "inferences" in stats
        assert "cache_valid" in stats

    def test_clear_cache(self, tmp_path: Path) -> None:
        cfg = ReasoningConfig(output_dir=tmp_path, cache_enabled=True)
        engine = ReasoningEngine(cfg)
        engine.reason(force=True)
        engine.clear_cache()
        stats = engine.stats()
        assert stats["cache_valid"] is False

    def test_config_property(self, tmp_path: Path) -> None:
        cfg = ReasoningConfig(output_dir=tmp_path)
        engine = ReasoningEngine(cfg)
        assert engine.config.output_dir == tmp_path

    def test_store_property(self, tmp_path: Path) -> None:
        cfg = ReasoningConfig(output_dir=tmp_path)
        engine = ReasoningEngine(cfg)
        assert engine.store is not None
