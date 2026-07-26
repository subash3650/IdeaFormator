"""Tests for OpportunityScorer."""

from __future__ import annotations

from pathlib import Path

import pytest

from phase3.opportunity.config import OpportunityConfig
from phase3.opportunity.scoring import OpportunityScorer
from phase3.opportunity.schema import OpportunityStatus, RecommendationType


class TestOpportunityScorer:
    def test_score_empty_candidates(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        scorer = OpportunityScorer(cfg)
        result = scorer.score([], {}, "run1")
        assert result == []

    def test_score_single_candidate(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        scorer = OpportunityScorer(cfg)
        candidates = [{
            "root_problem": "p1",
            "title": "Slow app performance",
            "summary": "Users report slow loading times",
            "evidence_ids": ["ev1", "ev2"],
            "reasoning_chain_ids": [],
            "cluster_ids": [],
            "kg_node_ids": [],
            "affected_products": ["ProductX"],
            "affected_companies": [],
            "affected_technologies": [],
            "pain_severity": 0.8,
            "frequency_score": 0.7,
            "trend_score": 0.6,
            "evidence_count": 5,
            "reasoning_confidence": 0.75,
            "product_count": 1,
            "company_count": 0,
            "platform_count": 2,
            "cluster_density": 0.6,
            "competition_score": 0.4,
            "feasibility_score": 0.8,
            "novelty_score": 0.6,
            "recent_evidence_ratio": 0.7,
            "evidence_growth_rate": 0.3,
            "estimated_market_size": "unknown",
        }]
        context = {
            "max_evidence_count": 10,
            "max_product_count": 5,
            "total_platforms": 3,
            "total_products": 10,
            "total_companies": 5,
        }
        opportunities = scorer.score(candidates, context, "run1")
        assert len(opportunities) == 1
        opp = opportunities[0]
        assert opp.root_problem == "p1"
        assert opp.status == OpportunityStatus.SCORED
        assert opp.recommendation_type == RecommendationType.INSUFFICIENT_DATA
        assert 0.0 <= opp.opportunity_score <= 1.0
        assert opp.scoring_breakdown.pain_severity == 0.8
        assert opp.confidence.final_confidence > 0.0

    def test_providers_used(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        scorer = OpportunityScorer(cfg)
        assert "weighted" in scorer.providers_used

    def test_deterministic_ids(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        scorer = OpportunityScorer(cfg)
        candidates = [{
            "root_problem": "p1", "title": "Test", "summary": "S",
            "evidence_ids": [], "reasoning_chain_ids": [], "cluster_ids": [],
            "kg_node_ids": [], "affected_products": [], "affected_companies": [],
            "affected_technologies": [], "pain_severity": 0.5, "frequency_score": 0.5,
            "trend_score": 0.5, "evidence_count": 1, "reasoning_confidence": 0.5,
            "product_count": 0, "company_count": 0, "platform_count": 1,
            "cluster_density": 0.5, "competition_score": 0.5, "feasibility_score": 0.5,
            "novelty_score": 0.5, "recent_evidence_ratio": 0.5, "evidence_growth_rate": 0.0,
            "estimated_market_size": "unknown",
        }]
        context = {"max_evidence_count": 10, "max_product_count": 5, "total_platforms": 3, "total_products": 10, "total_companies": 5}
        opps1 = scorer.score(candidates, context, "run1")
        opps2 = scorer.score(candidates, context, "run1")
        assert opps1[0].opportunity_id == opps2[0].opportunity_id

    def test_different_run_different_ids(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        scorer = OpportunityScorer(cfg)
        candidates = [{
            "root_problem": "p1", "title": "Test", "summary": "S",
            "evidence_ids": [], "reasoning_chain_ids": [], "cluster_ids": [],
            "kg_node_ids": [], "affected_products": [], "affected_companies": [],
            "affected_technologies": [], "pain_severity": 0.5, "frequency_score": 0.5,
            "trend_score": 0.5, "evidence_count": 1, "reasoning_confidence": 0.5,
            "product_count": 0, "company_count": 0, "platform_count": 1,
            "cluster_density": 0.5, "competition_score": 0.5, "feasibility_score": 0.5,
            "novelty_score": 0.5, "recent_evidence_ratio": 0.5, "evidence_growth_rate": 0.0,
            "estimated_market_size": "unknown",
        }]
        context = {"max_evidence_count": 10, "max_product_count": 5, "total_platforms": 3, "total_products": 10, "total_companies": 5}
        opps1 = scorer.score(candidates, context, "run1")
        opps2 = scorer.score(candidates, context, "run2")
        assert opps1[0].opportunity_id != opps2[0].opportunity_id

    def test_market_size_estimation(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        scorer = OpportunityScorer(cfg)
        candidates = [{
            "root_problem": "p1", "title": "T", "summary": "S",
            "evidence_ids": [str(i) for i in range(50)],
            "reasoning_chain_ids": [], "cluster_ids": [],
            "kg_node_ids": [],
            "affected_products": [f"Product{i}" for i in range(10)],
            "affected_companies": [f"Company{i}" for i in range(10)],
            "affected_technologies": [],
            "pain_severity": 0.5, "frequency_score": 0.5, "trend_score": 0.5,
            "evidence_count": 50, "reasoning_confidence": 0.5,
            "product_count": 10, "company_count": 10, "platform_count": 5,
            "cluster_density": 0.5, "competition_score": 0.5, "feasibility_score": 0.5,
            "novelty_score": 0.5, "recent_evidence_ratio": 0.5, "evidence_growth_rate": 0.0,
            "estimated_market_size": "unknown",
        }]
        context = {"max_evidence_count": 100, "max_product_count": 20, "total_platforms": 5, "total_products": 50, "total_companies": 30}
        opps = scorer.score(candidates, context, "run1")
        from phase3.opportunity.schema import MarketSize
        assert opps[0].estimated_market_size in (MarketSize.LARGE, MarketSize.MEDIUM, MarketSize.SMALL, MarketSize.UNKNOWN)

    def test_score_range(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        scorer = OpportunityScorer(cfg)
        candidates = [{
            "root_problem": f"p{i}", "title": f"T{i}", "summary": "S",
            "evidence_ids": [], "reasoning_chain_ids": [], "cluster_ids": [],
            "kg_node_ids": [], "affected_products": [], "affected_companies": [],
            "affected_technologies": [], "pain_severity": 0.1 * i,
            "frequency_score": 0.1 * i, "trend_score": 0.1 * i, "evidence_count": i,
            "reasoning_confidence": 0.1 * i, "product_count": 0, "company_count": 0,
            "platform_count": 1, "cluster_density": 0.1 * i,
            "competition_score": 0.1 * i, "feasibility_score": 0.1 * i,
            "novelty_score": 0.1 * i, "recent_evidence_ratio": 0.5,
            "evidence_growth_rate": 0.0, "estimated_market_size": "unknown",
        } for i in range(1, 6)]
        context = {"max_evidence_count": 10, "max_product_count": 5, "total_platforms": 3, "total_products": 10, "total_companies": 5}
        opps = scorer.score(candidates, context, "run1")
        for opp in opps:
            assert 0.0 <= opp.opportunity_score <= 1.0
