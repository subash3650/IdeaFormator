"""Tests for OpportunityExporter."""

from __future__ import annotations

from pathlib import Path

from phase3.opportunity.exporter import OpportunityExporter
from phase3.opportunity.schema import (
    ConfidenceBreakdown,
    Opportunity,
    OpportunityMetadata,
    ScoringBreakdown,
)
from phase3.opportunity.store import OpportunityStore


class TestOpportunityExporter:
    def test_export_report(self, tmp_path: Path) -> None:
        store = OpportunityStore(tmp_path)
        store.save_opportunities([
            Opportunity(opportunity_id="o1", title="T1", summary="S", root_problem="p"),
        ], "run1")
        store.save_metadata(OpportunityMetadata(run_id="run1"))
        exporter = OpportunityExporter(store)
        path = exporter.export_report()
        assert path.exists()
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "total_opportunities" in data

    def test_export_statistics(self, tmp_path: Path) -> None:
        store = OpportunityStore(tmp_path)
        store.save_opportunities([
            Opportunity(opportunity_id="o1", title="T1", summary="S", root_problem="p",
                        opportunity_score=0.85),
        ], "run1")
        exporter = OpportunityExporter(store)
        path = exporter.export_statistics()
        assert path.exists()
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "opportunity_count" in data

    def test_export_dashboard(self, tmp_path: Path) -> None:
        store = OpportunityStore(tmp_path)
        store.save_opportunities([
            Opportunity(opportunity_id="o1", title="T1", summary="S", root_problem="p"),
        ], "run1")
        store.save_metadata(OpportunityMetadata(run_id="run1"))
        exporter = OpportunityExporter(store)
        path = exporter.export_dashboard()
        assert path.exists()

    def test_export_dashboard_text(self, tmp_path: Path) -> None:
        store = OpportunityStore(tmp_path)
        store.save_opportunities([
            Opportunity(opportunity_id="o1", title="T1", summary="S", root_problem="p"),
        ], "run1")
        store.save_metadata(OpportunityMetadata(run_id="run1"))
        exporter = OpportunityExporter(store)
        path = exporter.export_dashboard_text()
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "OPPORTUNITY DISCOVERY" in content

    def test_export_summary(self, tmp_path: Path) -> None:
        store = OpportunityStore(tmp_path)
        store.save_opportunities([
            Opportunity(opportunity_id="o1", title="T1", summary="S", root_problem="p"),
        ], "run1")
        store.save_metadata(OpportunityMetadata(run_id="run1"))
        exporter = OpportunityExporter(store)
        path = exporter.export_summary()
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Opportunity Discovery Summary" in content

    def test_export_csv(self, tmp_path: Path) -> None:
        store = OpportunityStore(tmp_path)
        store.save_opportunities([
            Opportunity(opportunity_id="o1", title="T1", summary="S", root_problem="p"),
        ], "run1")
        exporter = OpportunityExporter(store)
        path = exporter.export_csv()
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "opportunity_id" in content

    def test_export_empty(self, tmp_path: Path) -> None:
        store = OpportunityStore(tmp_path)
        exporter = OpportunityExporter(store)
        path = exporter.export_report()
        assert path.exists()
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["total_opportunities"] == 0

    def test_all_exports_return_paths(self, tmp_path: Path) -> None:
        store = OpportunityStore(tmp_path)
        store.save_opportunities([
            Opportunity(opportunity_id="o1", title="T", summary="S", root_problem="p"),
        ], "run1")
        store.save_metadata(OpportunityMetadata(run_id="run1"))
        exporter = OpportunityExporter(store)
        assert exporter.export_report().exists()
        assert exporter.export_statistics().exists()
        assert exporter.export_dashboard().exists()
        assert exporter.export_dashboard_text().exists()
        assert exporter.export_summary().exists()
        assert exporter.export_csv().exists()
