"""Tests for OpportunityBuilder."""

from __future__ import annotations

from pathlib import Path

from phase3.opportunity.builder import OpportunityBuilder
from phase3.opportunity.config import OpportunityConfig


class MockRC:
    def __init__(self, cause: str, effect: str, conf: float = 0.8):
        self.cause_node_id = cause
        self.cause_label = cause
        self.effect_node_id = effect
        self.effect_label = effect
        self.path = [cause, effect]
        self.path_length = 2
        self.propagated_confidence = conf
        self.transitive_impact_count = 1
        self.evidence_count = 1


class MockEA:
    def __init__(self, conclusion: str = "e1", ev: list[str] | None = None):
        self.conclusion_node_id = conclusion
        self.conclusion_label = conclusion
        self.evidence_node_ids = ev or []
        self.evidence_count = len(self.evidence_node_ids)
        self.aggregated_confidence = 0.5


class MockCluster:
    def __init__(self, cid: str, rep: str, members: list[str] | None = None):
        self.cluster_id = cid
        self.representative_id = rep
        self.member_ids = members or [rep]
        self.member_count = len(self.member_ids)
        self.density = 0.5
        self.quality_score = 0.5


class TestOpportunityBuilder:
    def test_build_empty(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, min_confidence_threshold=0.0)
        builder = OpportunityBuilder(cfg)
        result = builder.build([], [], [], [], [], [], [], "run1")
        assert result["total_opportunities"] == 0

    def test_build_single_opportunity(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, min_confidence_threshold=0.0)
        builder = OpportunityBuilder(cfg)
        rc = MockRC("c1", "e1")
        builder.build([rc], [], [], [], [], [], [], "run1")
        loaded = builder.store.load_opportunities()
        assert len(loaded) > 0
        assert loaded[0].root_problem == "c1"

    def test_build_persists_to_store(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, min_confidence_threshold=0.0)
        builder = OpportunityBuilder(cfg)
        rc = MockRC("c1", "e1")
        builder.build([rc], [], [], [], [], [], [], "run1")
        loaded = builder.store.load_opportunities()
        assert len(loaded) > 0

    def test_build_with_cache(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, min_confidence_threshold=0.0)
        builder = OpportunityBuilder(cfg)
        rc = MockRC("c1", "e1")
        result1 = builder.build([rc], [], [], [], [], [], [], "run1")
        result2 = builder.build([rc], [], [], [], [], [], [], "run1")
        assert result1["cache_hit"] is False
        assert result2["cache_hit"] is True

    def test_build_opportunity_has_recommendation(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, min_confidence_threshold=0.0)
        builder = OpportunityBuilder(cfg)
        rc = MockRC("c1", "e1", conf=0.9)
        ev = MockEA(conclusion="e1", ev=["ev1", "ev2"])
        builder.build([rc], [ev], [], [], [], [], [], "run1")
        loaded = builder.store.load_opportunities()
        assert len(loaded) > 0
        assert loaded[0].recommendation_type is not None
        assert loaded[0].suggested_solution != ""

    def test_build_opportunity_has_rank(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, min_confidence_threshold=0.0)
        builder = OpportunityBuilder(cfg)
        rcs = [MockRC("c1", "e1", conf=0.9), MockRC("c2", "e2", conf=0.5)]
        builder.build(rcs, [], [], [], [], [], [], "run1")
        loaded = builder.store.load_opportunities()
        assert len(loaded) >= 1
        for opp in loaded:
            assert opp.rank >= 1
