"""Tests for RecommendationEngine."""

from __future__ import annotations

from pathlib import Path

from phase3.opportunity.config import OpportunityConfig
from phase3.opportunity.recommendation import RecommendationEngine
from phase3.opportunity.schema import (
    ConfidenceBreakdown,
    Opportunity,
    RecommendationType,
    ScoringBreakdown,
)


class TestRecommendationEngine:
    def test_recommend_empty(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        engine = RecommendationEngine(cfg)
        result = engine.recommend([])
        assert result == []

    def test_strong_pursue(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, strong_pursue_threshold=0.75)
        engine = RecommendationEngine(cfg)
        opp = Opportunity(
            opportunity_id="o1", title="Test", summary="S", root_problem="p",
            opportunity_score=0.85,
            scoring_breakdown=ScoringBreakdown(),
            confidence=ConfidenceBreakdown(final_confidence=0.8),
        )
        result = engine.recommend([opp])
        assert result[0].recommendation_type == RecommendationType.STRONG_PURSUE

    def test_worth_exploring(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, worth_exploring_threshold=0.50)
        engine = RecommendationEngine(cfg)
        opp = Opportunity(
            opportunity_id="o1", title="Test", summary="S", root_problem="p",
            opportunity_score=0.55,
            scoring_breakdown=ScoringBreakdown(),
            confidence=ConfidenceBreakdown(final_confidence=0.5),
        )
        result = engine.recommend([opp])
        assert result[0].recommendation_type == RecommendationType.WORTH_EXPLORING

    def test_niche_potential(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, niche_potential_threshold=0.30)
        engine = RecommendationEngine(cfg)
        opp = Opportunity(
            opportunity_id="o1", title="Test", summary="S", root_problem="p",
            opportunity_score=0.35,
            scoring_breakdown=ScoringBreakdown(),
            confidence=ConfidenceBreakdown(final_confidence=0.3),
        )
        result = engine.recommend([opp])
        assert result[0].recommendation_type == RecommendationType.NICHE_POTENTIAL

    def test_monitor(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, monitor_threshold=0.10)
        engine = RecommendationEngine(cfg)
        opp = Opportunity(
            opportunity_id="o1", title="Test", summary="S", root_problem="p",
            opportunity_score=0.15,
            scoring_breakdown=ScoringBreakdown(),
            confidence=ConfidenceBreakdown(final_confidence=0.1),
        )
        result = engine.recommend([opp])
        assert result[0].recommendation_type == RecommendationType.MONITOR

    def test_insufficient_data(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        engine = RecommendationEngine(cfg)
        opp = Opportunity(
            opportunity_id="o1", title="Test", summary="S", root_problem="p",
            opportunity_score=0.05,
            scoring_breakdown=ScoringBreakdown(),
            confidence=ConfidenceBreakdown(final_confidence=0.0),
        )
        result = engine.recommend([opp])
        assert result[0].recommendation_type == RecommendationType.INSUFFICIENT_DATA

    def test_suggested_solution_generated(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        engine = RecommendationEngine(cfg)
        opp = Opportunity(
            opportunity_id="o1", title="Test", summary="S", root_problem="p",
            opportunity_score=0.85,
            scoring_breakdown=ScoringBreakdown(),
            confidence=ConfidenceBreakdown(final_confidence=0.8),
            affected_products=["ProductA"],
        )
        result = engine.recommend([opp])
        assert len(result[0].suggested_solution) > 0
        assert result[0].suggested_solution[0].upper() == "A"

    def test_sets_recommended_status(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        engine = RecommendationEngine(cfg)
        opp = Opportunity(
            opportunity_id="o1", title="Test", summary="S", root_problem="p",
            opportunity_score=0.85,
            scoring_breakdown=ScoringBreakdown(),
            confidence=ConfidenceBreakdown(final_confidence=0.8),
        )
        from phase3.opportunity.schema import OpportunityStatus
        result = engine.recommend([opp])
        assert result[0].status == OpportunityStatus.RECOMMENDED

    def test_business_model_selected(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        engine = RecommendationEngine(cfg)
        opp = Opportunity(
            opportunity_id="o1", title="AI Automation Platform",
            summary="Intelligent agent for business", root_problem="p",
            opportunity_score=0.85,
            scoring_breakdown=ScoringBreakdown(),
            confidence=ConfidenceBreakdown(final_confidence=0.8),
        )
        result = engine.recommend([opp])
        from phase3.opportunity.schema import OpportunityType
        assert isinstance(result[0].suggested_business_model, OpportunityType)
