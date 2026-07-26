"""Tests for OpportunityEngine."""

from __future__ import annotations

from pathlib import Path

from phase3.opportunity.config import OpportunityConfig


class TestOpportunityEngine:
    def test_stats_empty(self, tmp_path: Path) -> None:
        from phase3.opportunity.engine import OpportunityEngine
        cfg = OpportunityConfig(output_dir=tmp_path, min_confidence_threshold=0.0)
        engine = OpportunityEngine(cfg)
        stats = engine.stats()
        assert isinstance(stats, dict)
        assert stats["total_opportunities"] == 0

    def test_stats_after_add(self, tmp_path: Path) -> None:
        from phase3.opportunity.engine import OpportunityEngine
        from phase3.opportunity.schema import Opportunity
        cfg = OpportunityConfig(output_dir=tmp_path, min_confidence_threshold=0.0)
        engine = OpportunityEngine(cfg)
        engine.store.save_opportunities([
            Opportunity(opportunity_id="o1", title="T", summary="S", root_problem="p"),
        ], "run1")
        stats = engine.stats()
        assert stats["total_opportunities"] == 1

    def test_search(self, tmp_path: Path) -> None:
        from phase3.opportunity.engine import OpportunityEngine
        from phase3.opportunity.schema import Opportunity
        cfg = OpportunityConfig(output_dir=tmp_path, min_confidence_threshold=0.0)
        engine = OpportunityEngine(cfg)
        engine.store.save_opportunities([
            Opportunity(opportunity_id="o1", title="Test Opp", summary="S", root_problem="p"),
        ], "run1")
        results = engine.search("Test")
        assert len(results) > 0

    def test_search_no_match(self, tmp_path: Path) -> None:
        from phase3.opportunity.engine import OpportunityEngine
        cfg = OpportunityConfig(output_dir=tmp_path, min_confidence_threshold=0.0)
        engine = OpportunityEngine(cfg)
        results = engine.search("nonexistent")
        assert results == []

    def test_clear_cache(self, tmp_path: Path) -> None:
        from phase3.opportunity.engine import OpportunityEngine
        cfg = OpportunityConfig(output_dir=tmp_path, min_confidence_threshold=0.0)
        engine = OpportunityEngine(cfg)
        engine.clear_cache()
        assert True
