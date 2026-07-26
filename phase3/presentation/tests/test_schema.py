from __future__ import annotations

import hashlib
import re

import pytest
from pydantic import ValidationError

from phase3.presentation.schema import (
    ChartSeries,
    ChartSpec,
    ChartType,
    ComparisonChange,
    Highlight,
    PresentationModel,
    ReportAssets,
    ReportComparison,
    ReportFormat,
    ReportIndex,
    ReportIndexEntry,
    ReportOutput,
    ReportSection,
    ReportSummaries,
    ReportType,
    ReportVersion,
    SectionDefinition,
    SectionProvenance,
    SectionType,
    SourceLineage,
    TableSpec,
    TrendDelta,
)


class TestEnums:
    def test_report_type_values(self) -> None:
        assert ReportType.startup_opportunity.value == "startup_opportunity"
        assert ReportType.executive_summary.value == "executive_summary"
        assert len(ReportType) == 10

    def test_report_format_values(self) -> None:
        assert ReportFormat.html.value == "html"
        assert ReportFormat.pdf.value == "pdf"
        assert len(ReportFormat) == 7

    def test_section_type_values(self) -> None:
        assert SectionType.executive_summary.value == "executive_summary"
        assert len(SectionType) == 11

    def test_chart_type_values(self) -> None:
        assert ChartType.bar.value == "bar"
        assert len(ChartType) == 7

    def test_comparison_change_values(self) -> None:
        assert ComparisonChange.added.value == "added"
        assert len(ComparisonChange) == 4


class TestReportVersion:
    def test_default_version(self) -> None:
        v = ReportVersion()
        assert v.report_version == "1.0.0"
        assert v.schema_version == "1.0.0"

    def test_frozen(self) -> None:
        v = ReportVersion()
        with pytest.raises(ValidationError):
            v.report_version = "2.0.0"

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            ReportVersion(unknown_field="x")


class TestSourceLineage:
    def test_default_lineage(self) -> None:
        l = SourceLineage()
        assert l.knowledge_graph_run_id is None
        assert l.pipeline_manifest == {}

    def test_with_values(self) -> None:
        l = SourceLineage(knowledge_graph_run_id="kg123", snapshot_id="snap1")
        assert l.knowledge_graph_run_id == "kg123"
        assert l.snapshot_id == "snap1"

    def test_frozen(self) -> None:
        l = SourceLineage()
        with pytest.raises(ValidationError):
            l.knowledge_graph_run_id = "new"


class TestReportSummaries:
    def test_default_summaries(self) -> None:
        s = ReportSummaries()
        assert s.one_paragraph == ""
        assert s.five_bullets == []
        assert s.json_summary == {}

    def test_with_values(self) -> None:
        s = ReportSummaries(
            one_paragraph="Test paragraph",
            five_bullets=["Bullet 1", "Bullet 2"],
            one_tweet="Short tweet",
        )
        assert s.one_paragraph == "Test paragraph"
        assert len(s.five_bullets) == 2
        assert s.one_tweet == "Short tweet"


class TestSectionProvenance:
    def test_default_provenance(self) -> None:
        p = SectionProvenance()
        assert p.reasoning_chain_ids == []
        assert p.confidence == 0.0

    def test_with_values(self) -> None:
        p = SectionProvenance(
            reasoning_chain_ids=["chain1", "chain2"],
            kg_node_ids=["node1"],
            confidence=0.85,
        )
        assert len(p.reasoning_chain_ids) == 2
        assert len(p.kg_node_ids) == 1
        assert p.confidence == 0.85


class TestSectionDefinition:
    def test_section_definition(self) -> None:
        sd = SectionDefinition(section_type=SectionType.trend_analysis, title="Trends", order=2)
        assert sd.section_type == SectionType.trend_analysis
        assert sd.title == "Trends"
        assert sd.order == 2
        assert sd.renderer_hints == {}

    def test_with_hints(self) -> None:
        sd = SectionDefinition(
            section_type=SectionType.charts,
            title="Charts",
            order=5,
            renderer_hints={"max_charts": 3},
        )
        assert sd.renderer_hints["max_charts"] == 3


class TestChartSeries:
    def test_chart_series(self) -> None:
        cs = ChartSeries(name="Series A", values=[1, 2, 3])
        assert cs.name == "Series A"
        assert cs.values == [1, 2, 3]
        assert cs.color is None

    def test_with_color(self) -> None:
        cs = ChartSeries(name="Series B", values=[4, 5], color="#ff0000")
        assert cs.color == "#ff0000"


class TestChartSpec:
    def test_auto_generates_id(self) -> None:
        cs = ChartSpec(chart_type=ChartType.bar, title="Test Chart")
        assert cs.chart_id != ""
        assert isinstance(cs.chart_id, str)
        assert len(cs.chart_id) == 16
        assert re.match(r"^[a-f0-9]{16}$", cs.chart_id)

    def test_with_explicit_id(self) -> None:
        cs = ChartSpec(
            chart_id="abc123",
            chart_type=ChartType.line,
            title="Line Chart",
        )
        assert cs.chart_id == "abc123"

    def test_with_series(self) -> None:
        series = [ChartSeries(name="A", values=[1, 2, 3])]
        cs = ChartSpec(
            chart_type=ChartType.pie,
            title="Pie",
            series=series,
            labels=["X", "Y", "Z"],
        )
        assert len(cs.series) == 1
        assert cs.series[0].name == "A"
        assert cs.labels == ["X", "Y", "Z"]

    def test_frozen(self) -> None:
        cs = ChartSpec(chart_type=ChartType.bar, title="T")
        with pytest.raises(ValidationError):
            cs.title = "New"


class TestTableSpec:
    def test_auto_generates_id(self) -> None:
        ts = TableSpec(title="Test Table", headers=["A", "B"], rows=[["1", "2"]])
        assert ts.table_id != ""
        assert len(ts.table_id) == 16

    def test_with_explicit_id(self) -> None:
        ts = TableSpec(
            table_id="tab123",
            title="Table",
            headers=["A"],
            rows=[["1"]],
        )
        assert ts.table_id == "tab123"


class TestHighlight:
    def test_highlight(self) -> None:
        h = Highlight(text="Key finding", source="trend", score=0.9, section=SectionType.top_findings)
        assert h.text == "Key finding"
        assert h.source == "trend"
        assert h.score == 0.9


class TestReportSection:
    def test_auto_generates_id(self) -> None:
        rs = ReportSection(section_type=SectionType.executive_summary, title="Executive Summary", order=0)
        assert rs.section_id != ""
        assert len(rs.section_id) == 16

    def test_with_content(self) -> None:
        rs = ReportSection(
            section_type=SectionType.trend_analysis,
            title="Trends",
            order=2,
            content={"growing": 5, "declining": 2},
        )
        assert rs.content["growing"] == 5
        assert rs.provenance.confidence == 0.0

    def test_with_provenance(self) -> None:
        provenance = SectionProvenance(reasoning_chain_ids=["c1"], confidence=0.9)
        rs = ReportSection(
            section_type=SectionType.root_causes,
            title="Root Causes",
            order=3,
            provenance=provenance,
        )
        assert len(rs.provenance.reasoning_chain_ids) == 1
        assert rs.provenance.confidence == 0.9

    def test_default_summaries(self) -> None:
        rs = ReportSection(section_type=SectionType.evidence, title="Evidence", order=4)
        assert rs.summaries.one_paragraph == ""


class TestReportAssets:
    def test_default_assets(self) -> None:
        a = ReportAssets()
        assert a.charts == []
        assert a.tables == []
        assert a.metrics == {}

    def test_with_values(self) -> None:
        chart = ChartSpec(chart_type=ChartType.bar, title="B")
        table = TableSpec(title="T", headers=["A"], rows=[["1"]])
        hl = Highlight(text="H", source="opp", score=0.8, section=SectionType.top_findings)
        a = ReportAssets(
            charts=[chart],
            tables=[table],
            metrics={"total": 42.0},
            highlights=[hl],
            statistics={"avg": 3.5},
        )
        assert len(a.charts) == 1
        assert len(a.tables) == 1
        assert len(a.highlights) == 1
        assert a.metrics["total"] == 42.0


class TestPresentationModel:
    def test_auto_generates_id_and_timestamp(self) -> None:
        pm = PresentationModel(report_type=ReportType.executive_summary, title="Test Report")
        assert pm.report_id != ""
        assert len(pm.report_id) == 16
        assert pm.generated_at != ""

    def test_with_explicit_id(self) -> None:
        pm = PresentationModel(
            report_id="custom123",
            report_type=ReportType.investor,
            title="Investor Report",
        )
        assert pm.report_id == "custom123"

    def test_default_fields(self) -> None:
        pm = PresentationModel(report_type=ReportType.weekly, title="Weekly")
        assert pm.locale == "en-US"
        assert pm.sections == []
        assert pm.tags == []
        assert pm.companies == []
        assert pm.checksum == ""

    def test_with_sections(self) -> None:
        rs = ReportSection(section_type=SectionType.top_findings, title="Findings", order=0)
        pm = PresentationModel(
            report_type=ReportType.market_intelligence,
            title="Market Report",
            sections=[rs],
            tags=["ai", "saas"],
            companies=["Acme"],
        )
        assert len(pm.sections) == 1
        assert "ai" in pm.tags
        assert "Acme" in pm.companies

    def test_with_lineage(self) -> None:
        lineage = SourceLineage(knowledge_graph_run_id="kg_run_1")
        pm = PresentationModel(
            report_type=ReportType.competitor_analysis,
            title="Competitor",
            lineage=lineage,
        )
        assert pm.lineage.knowledge_graph_run_id == "kg_run_1"

    def test_frozen(self) -> None:
        pm = PresentationModel(report_type=ReportType.product, title="Product")
        with pytest.raises(ValidationError):
            pm.title = "New"

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            PresentationModel(report_type=ReportType.company, title="C", unknown="x")


class TestTrendDelta:
    def test_trend_delta(self) -> None:
        td = TrendDelta(trend_id="t1", title="Growing Trend", change=ComparisonChange.changed)
        assert td.trend_id == "t1"
        assert td.change == ComparisonChange.changed
        assert td.growth_pct_before == 0.0

    def test_with_values(self) -> None:
        td = TrendDelta(
            trend_id="t2",
            title="AI Trend",
            change=ComparisonChange.added,
            score_after=0.95,
        )
        assert td.score_after == 0.95


class TestReportComparison:
    def test_auto_generates_id_and_timestamp(self) -> None:
        rc = ReportComparison(report_a_id="r1", report_b_id="r2")
        assert rc.comparison_id != ""
        assert rc.generated_at != ""

    def test_with_changes(self) -> None:
        td = TrendDelta(trend_id="t1", title="T", change=ComparisonChange.unchanged)
        rc = ReportComparison(
            report_a_id="r1",
            report_b_id="r2",
            new_opportunities=["opp1"],
            changed_trends=[td],
            confidence_deltas={"overall": -0.05},
        )
        assert rc.new_opportunities == ["opp1"]
        assert len(rc.changed_trends) == 1
        assert rc.confidence_deltas["overall"] == -0.05


class TestReportIndexEntry:
    def test_minimal_entry(self) -> None:
        e = ReportIndexEntry(
            report_id="r1",
            report_type=ReportType.executive_summary,
            title="Summary",
            generated_at="2024-01-01T00:00:00",
        )
        assert e.tags == []
        assert e.sections == []

    def test_with_tags(self) -> None:
        e = ReportIndexEntry(
            report_id="r2",
            report_type=ReportType.technology_landscape,
            title="Tech",
            generated_at="2024-01-01T00:00:00",
            tags=["ai", "ml"],
            companies=["OpenAI"],
            sections=[SectionType.trend_analysis],
        )
        assert "ai" in e.tags
        assert "OpenAI" in e.companies
        assert SectionType.trend_analysis in e.sections


class TestReportIndex:
    def test_default_index(self) -> None:
        idx = ReportIndex()
        assert idx.entries == {}
        assert idx.by_tag == {}
        assert idx.by_company == {}

    def test_mutable(self) -> None:
        idx = ReportIndex()
        idx.entries["r1"] = ReportIndexEntry(
            report_id="r1",
            report_type=ReportType.executive_summary,
            title="S",
            generated_at="2024-01-01T00:00:00",
        )
        assert "r1" in idx.entries

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            ReportIndex(unknown="x")


class TestReportOutput:
    def test_report_output(self) -> None:
        entry = ReportIndexEntry(
            report_id="r1",
            report_type=ReportType.executive_summary,
            title="S",
            generated_at="2024-01-01T00:00:00",
        )
        ro = ReportOutput(
            report_id="r1",
            report_type=ReportType.executive_summary,
            title="S",
            generated_at="2024-01-01T00:00:00",
            sections_count=3,
            charts_count=2,
            formats=[ReportFormat.json, ReportFormat.html],
            index_entry=entry,
            elapsed_seconds=1.5,
        )
        assert ro.sections_count == 3
        assert ro.charts_count == 2
        assert len(ro.formats) == 2


class TestDeterminism:
    def test_chart_spec_deterministic_id(self) -> None:
        cs1 = ChartSpec(chart_type=ChartType.bar, title="Monthly Trends")
        cs2 = ChartSpec(chart_type=ChartType.bar, title="Monthly Trends")
        assert cs1.chart_type == cs2.chart_type
        assert cs1.title == cs2.title

    def test_presentation_model_deterministic_id(self) -> None:
        pm1 = PresentationModel(report_type=ReportType.weekly, title="Weekly Brief")
        pm2 = PresentationModel(report_type=ReportType.weekly, title="Weekly Brief")
        assert pm1.report_type == pm2.report_type
        assert pm1.title == pm2.title

    def test_round_trip_serialization(self) -> None:
        pm = PresentationModel(report_type=ReportType.executive_summary, title="Round Trip")
        serialized = pm.model_dump(mode="json")
        restored = PresentationModel(**serialized)
        assert restored.report_id == pm.report_id
        assert restored.title == pm.title
        assert restored.generated_at == pm.generated_at

    def test_report_assets_round_trip(self) -> None:
        assets = ReportAssets(
            metrics={"score": 85.0},
            highlights=[Highlight(text="H", source="trend", score=0.9, section=SectionType.top_findings)],
        )
        serialized = assets.model_dump(mode="json")
        restored = ReportAssets(**serialized)
        assert restored.metrics["score"] == 85.0
        assert len(restored.highlights) == 1


class TestEdgeCases:
    def test_empty_chart_spec(self) -> None:
        cs = ChartSpec(chart_type=ChartType.line, title="")
        assert cs.title == ""
        assert cs.series == []
        assert cs.labels == []

    def test_large_series(self) -> None:
        values = list(range(1000))
        cs = ChartSpec(
            chart_type=ChartType.line,
            title="Large",
            series=[ChartSeries(name="Big", values=values)],
        )
        assert len(cs.series[0].values) == 1000

    def test_report_section_with_many_charts(self) -> None:
        charts = [ChartSpec(chart_type=ChartType.bar, title=f"Chart {i}") for i in range(10)]
        rs = ReportSection(
            section_type=SectionType.charts,
            title="All Charts",
            order=0,
            charts=charts,
        )
        assert len(rs.charts) == 10

    def test_presentation_model_with_many_sections(self) -> None:
        sections = [
            ReportSection(section_type=SectionType(t), title=t.value, order=i)
            for i, t in enumerate(SectionType)
        ]
        pm = PresentationModel(
            report_type=ReportType.weekly,
            title="Full Report",
            sections=sections,
        )
        assert len(pm.sections) == len(SectionType)

    def test_report_comparison_empty_deltas(self) -> None:
        rc = ReportComparison(report_a_id="a", report_b_id="b")
        assert rc.confidence_deltas == {}
        assert rc.changed_trends == []
