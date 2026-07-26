"""Tests for OpportunityExtractor."""

from __future__ import annotations

from pathlib import Path

from phase3.opportunity.config import OpportunityConfig
from phase3.opportunity.extractor import OpportunityExtractor


class MockNode:
    def __init__(self, node_id: str, node_type=None, label: str = "", source_asset: str = ""):
        self.node_id = node_id
        self.node_type = node_type
        self.label = label
        self.source_asset = source_asset
        self.properties = {}
        self.metadata = {}
        self.attributes = {}
        self.confidence = 0.5


class MockEdge:
    def __init__(self, source_node_id: str, target_node_id: str, edge_type=None):
        self.source_node_id = source_node_id
        self.target_node_id = target_node_id
        self.edge_type = edge_type
        self.weight = 0.5


class MockRC:
    def __init__(self, cause: str, effect: str, path=None, conf: float = 0.5):
        self.cause_node_id = cause
        self.cause_label = cause
        self.effect_node_id = effect
        self.effect_label = effect
        self.path = path or [cause, effect]
        self.path_length = len(self.path)
        self.propagated_confidence = conf
        self.transitive_impact_count = 1
        self.evidence_count = 1


class MockEA:
    def __init__(self, conclusion: str, evidence_ids: list[str] | None = None):
        self.conclusion_node_id = conclusion
        self.conclusion_label = conclusion
        self.evidence_node_ids = evidence_ids or []
        self.evidence_count = len(self.evidence_node_ids)
        self.aggregated_confidence = 0.5


class MockCluster:
    def __init__(self, cluster_id: str, rep: str, members: list[str] | None = None, density: float = 0.5):
        self.cluster_id = cluster_id
        self.representative_id = rep
        self.member_ids = members or [rep]
        self.member_count = len(self.member_ids)
        self.density = density
        self.quality_score = 0.5


class TestOpportunityExtractor:
    def test_extract_empty(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        extractor = OpportunityExtractor(cfg)
        result = extractor.extract([], [], [], [], [], [], [])
        assert result == []

    def test_extract_from_root_causes(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        extractor = OpportunityExtractor(cfg)
        root_causes = [MockRC(cause="c1", effect="e1", conf=0.8)]
        result = extractor.extract(root_causes, [], [], [], [], [], [])
        assert len(result) > 0
        assert result[0]["root_problem"] == "c1"

    def test_extract_reasoning_confidence_threshold(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, min_confidence_threshold=0.5)
        extractor = OpportunityExtractor(cfg)
        root_causes = [MockRC(cause="c1", effect="e1", conf=0.3)]
        result = extractor.extract(root_causes, [], [], [], [], [], [])
        assert len(result) == 0

    def test_extract_from_evidence_aggregations(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, min_confidence_threshold=0.0)
        extractor = OpportunityExtractor(cfg)
        ev = MockEA(conclusion="e1", evidence_ids=["ev1", "ev2"])
        result = extractor.extract([], [ev], [], [], [], [], [])
        assert len(result) > 0
        assert result[0]["root_problem"] == "e1"

    def test_extract_from_clusters(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, min_confidence_threshold=0.0)
        extractor = OpportunityExtractor(cfg)
        cluster = MockCluster(cluster_id="cl1", rep="rep1", members=["rep1", "m1", "m2"], density=0.7)
        result = extractor.extract([], [], [], [], [], [], [cluster])
        assert len(result) > 0
        assert result[0]["root_problem"] == "rep1"

    def test_extract_merges_entities(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, min_confidence_threshold=0.0)
        extractor = OpportunityExtractor(cfg)
        class MockNodeType:
            value = "product"
        node = MockNode(node_id="p1", node_type=MockNodeType(), label="AwesomeApp")
        rc = MockRC(cause="c1", effect="e1", path=["c1", "n1", "e1"])
        result = extractor.extract([rc], [], [], [], [node], [], [])
        assert len(result) > 0

    def test_merge_multiple_intelligence_sources(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, min_confidence_threshold=0.0)
        extractor = OpportunityExtractor(cfg)
        rc = MockRC(cause="c1", effect="e1", conf=0.8)
        ev = MockEA(conclusion="e1", evidence_ids=["ev1"])
        result = extractor.extract([rc], [ev], [], [], [], [], [])
        assert len(result) >= 1

    def test_deduplication_prevents_duplicates(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, min_confidence_threshold=0.0)
        extractor = OpportunityExtractor(cfg)
        rc = MockRC(cause="c1", effect="e1")
        ev = MockEA(conclusion="c1")
        result = extractor.extract([rc], [ev], [], [], [], [], [])
        # Same root_problem shouldn't appear twice
        problems = [c["root_problem"] for c in result]
        assert len(problems) == len(set(problems))

    def test_candidate_has_evidence_ids(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, min_confidence_threshold=0.0)
        extractor = OpportunityExtractor(cfg)
        ev = MockEA(conclusion="e1", evidence_ids=["ev1", "ev2"])
        result = extractor.extract([], [ev], [], [], [], [], [])
        assert len(result) > 0
        assert "evidence_ids" in result[0]
