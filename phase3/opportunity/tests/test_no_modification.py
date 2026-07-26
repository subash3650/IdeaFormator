"""Tests that the Opportunity Engine does not mutate upstream data."""

from __future__ import annotations

from pathlib import Path

from phase3.opportunity.config import OpportunityConfig
from phase3.opportunity.extractor import OpportunityExtractor
from phase3.opportunity.scoring import OpportunityScorer


class MockNode:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.node_type = type("NT", (), {"value": "product"})()
        self.label = node_id
        self.source_asset = "ios"


class MockEdge:
    def __init__(self, src: str, tgt: str):
        self.source_node_id = src
        self.target_node_id = tgt
        self.weight = 0.5


class MockRC:
    def __init__(self, cause: str, effect: str):
        self.cause_node_id = cause
        self.cause_label = cause
        self.effect_node_id = effect
        self.effect_label = effect
        self.path = [cause, effect]
        self.path_length = 2
        self.propagated_confidence = 0.8
        self.transitive_impact_count = 2
        self.evidence_count = 1


class MockEA:
    def __init__(self):
        self.conclusion_node_id = "e1"
        self.conclusion_label = "e1"
        self.evidence_node_ids = ["ev1"]
        self.evidence_count = 1


class TestNoModification:
    def test_extract_does_not_mutate(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, min_confidence_threshold=0.0)
        extractor = OpportunityExtractor(cfg)
        rc = MockRC("c1", "e1")
        ev = MockEA()
        rc_before = rc.cause_node_id
        result = extractor.extract([rc], [ev], [], [], [], [], [])
        assert rc.cause_node_id == rc_before
        assert ev.conclusion_node_id == "e1"

    def test_score_does_not_mutate_candidates(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        scorer = OpportunityScorer(cfg)
        candidate = {
            "root_problem": "p1", "title": "Test", "summary": "S",
            "evidence_ids": [], "reasoning_chain_ids": [], "cluster_ids": [],
            "kg_node_ids": [], "affected_products": [], "affected_companies": [],
            "affected_technologies": [], "pain_severity": 0.5, "frequency_score": 0.5,
            "trend_score": 0.5, "evidence_count": 1, "reasoning_confidence": 0.5,
            "product_count": 0, "company_count": 0, "platform_count": 1,
            "cluster_density": 0.5, "competition_score": 0.5, "feasibility_score": 0.5,
            "novelty_score": 0.5, "recent_evidence_ratio": 0.5, "evidence_growth_rate": 0.0,
            "estimated_market_size": "unknown",
        }
        context = {"max_evidence_count": 10, "max_product_count": 5, "total_platforms": 3, "total_products": 10, "total_companies": 5}
        orig_problem = candidate["root_problem"]
        scorer.score([candidate], context, "run1")
        assert candidate["root_problem"] == orig_problem

    def test_load_does_not_mutate_stored_opportunities(self, tmp_path: Path) -> None:
        from phase3.opportunity.schema import Opportunity
        from phase3.opportunity.store import OpportunityStore
        store = OpportunityStore(tmp_path)
        opp = Opportunity(opportunity_id="o1", title="Test", summary="S", root_problem="p")
        store.save_opportunities([opp], "run1")
        loaded = store.load_opportunities()
        assert loaded[0].opportunity_id == "o1"
        assert loaded[0].title == "Test"
