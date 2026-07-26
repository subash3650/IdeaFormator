"""Tests for OpportunityStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from phase3.opportunity.schema import (
    ConfidenceBreakdown,
    Opportunity,
    OpportunityMetadata,
    ScoringBreakdown,
)
from phase3.opportunity.store import OpportunityStore


class TestOpportunityStore:
    def test_save_and_load_opportunities(self, tmp_path: Path) -> None:
        store = OpportunityStore(tmp_path)
        opps = [
            Opportunity(opportunity_id="o1", title="Opp 1", summary="S1", root_problem="p1"),
            Opportunity(opportunity_id="o2", title="Opp 2", summary="S2", root_problem="p2"),
        ]
        store.save_opportunities(opps, "run1")
        loaded = store.load_opportunities()
        assert len(loaded) == 2
        assert loaded[0].opportunity_id in ("o1", "o2")

    def test_save_and_load_empty(self, tmp_path: Path) -> None:
        store = OpportunityStore(tmp_path)
        store.save_opportunities([], "run1")
        loaded = store.load_opportunities()
        assert loaded == []

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        store = OpportunityStore(tmp_path / "nonexistent")
        loaded = store.load_opportunities()
        assert loaded == []

    def test_save_and_load_metadata(self, tmp_path: Path) -> None:
        store = OpportunityStore(tmp_path)
        meta = OpportunityMetadata(run_id="run1", total_opportunities=5)
        store.save_metadata(meta)
        loaded = store.load_metadata()
        assert loaded is not None
        assert loaded.run_id == "run1"
        assert loaded.total_opportunities == 5

    def test_load_metadata_nonexistent(self, tmp_path: Path) -> None:
        store = OpportunityStore(tmp_path)
        loaded = store.load_metadata()
        assert loaded is None

    def test_save_manifest(self, tmp_path: Path) -> None:
        store = OpportunityStore(tmp_path)
        manifest = {"run_id": "run1", "total": 10}
        path = store.save_manifest(manifest)
        assert path.exists()
        import json
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["run_id"] == "run1"

    def test_checksums(self, tmp_path: Path) -> None:
        store = OpportunityStore(tmp_path)
        store.save_opportunities([
            Opportunity(opportunity_id="o1", title="T", summary="S", root_problem="p"),
        ], "run1")
        cs = store.checksums()
        assert "opportunities.parquet" in cs
        assert len(cs["opportunities.parquet"]) == 16

    def test_checksums_empty(self, tmp_path: Path) -> None:
        store = OpportunityStore(tmp_path)
        cs = store.checksums()
        assert cs == {}

    def test_file_structure(self, tmp_path: Path) -> None:
        store = OpportunityStore(tmp_path)
        store.save_opportunities([
            Opportunity(opportunity_id="o1", title="T", summary="S", root_problem="p"),
        ], "run1")
        store.save_metadata(OpportunityMetadata(run_id="run1"))
        store.save_manifest({"run_id": "run1"})
        opp_dir = store.opportunity_dir
        assert (opp_dir / "opportunities.parquet").exists()
        assert (opp_dir / "opportunity_metadata.json").exists()
        assert (opp_dir / "opportunity_manifest.json").exists()

    def test_roundtrip_with_scores(self, tmp_path: Path) -> None:
        sb = ScoringBreakdown(pain_severity=0.8, frequency=0.7, trend=0.6)
        cb = ConfidenceBreakdown(final_confidence=0.85)
        opp = Opportunity(
            opportunity_id="o1", title="Test", summary="S", root_problem="p",
            opportunity_score=0.85,
            scoring_breakdown=sb,
            confidence=cb,
            affected_products=["ProductA"],
            affected_companies=["CompanyB"],
        )
        store = OpportunityStore(tmp_path)
        store.save_opportunities([opp], "run1")
        loaded = store.load_opportunities()
        assert len(loaded) == 1
        assert loaded[0].opportunity_score == 0.85
        assert loaded[0].scoring_breakdown.pain_severity == 0.8
        assert loaded[0].confidence.final_confidence == 0.85
        assert "ProductA" in loaded[0].affected_products
        assert "CompanyB" in loaded[0].affected_companies

    def test_properties(self, tmp_path: Path) -> None:
        store = OpportunityStore(tmp_path)
        assert store.base_path == tmp_path
        assert store.opportunity_dir == tmp_path / "opportunity"
        assert store.opportunities_path == tmp_path / "opportunity" / "opportunities.parquet"
        assert store.metadata_path == tmp_path / "opportunity" / "opportunity_metadata.json"
        assert store.manifest_path == tmp_path / "opportunity" / "opportunity_manifest.json"
