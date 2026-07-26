from __future__ import annotations

import json

import pytest

from phase3.presentation.config import PresentationConfig
from phase3.presentation.renderers import (
    CSVRenderer,
    HTMLRenderer,
    JSONRenderer,
    MarkdownRenderer,
    available_renderers,
    create_renderer,
    get_renderer_class,
    register_renderer,
)
from phase3.presentation.renderers.base import Renderer
from phase3.presentation.schema import (
    ChartSpec,
    ChartType,
    PresentationModel,
    ReportAssets,
    ReportFormat,
    ReportSection,
    ReportSummaries,
    ReportType,
    SectionProvenance,
    SectionType,
    SourceLineage,
    TableSpec,
)


@pytest.fixture
def config() -> PresentationConfig:
    return PresentationConfig()


def _make_model(**kwargs: object) -> PresentationModel:
    defaults: dict = {
        "report_type": ReportType.executive_summary,
        "title": "Test Report",
        "subtitle": "A test",
    }
    defaults.update(kwargs)
    return PresentationModel(**defaults)


def _make_section(
    section_type: SectionType = SectionType.top_findings,
    title: str = "Findings",
    order: int = 1,
    content: dict | None = None,
    charts: list | None = None,
    summaries: dict | None = None,
    provenance: dict | None = None,
) -> ReportSection:
    return ReportSection(
        section_type=section_type,
        title=title,
        order=order,
        content=content or {},
        charts=charts or [],
        summaries=ReportSummaries(**(summaries or {})),
        provenance=SectionProvenance(**(provenance or {})),
    )


class TestRendererBase:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            Renderer()

    def test_register_and_create(self) -> None:
        class TestRenderer(Renderer):
            @property
            def name(self) -> str:
                return "test_renderer"

            @property
            def format(self) -> ReportFormat:
                return ReportFormat.json

            def render(self, model: PresentationModel, config: PresentationConfig) -> str:
                return "test"

        register_renderer("test_renderer")(TestRenderer)
        r = create_renderer("test_renderer")
        assert isinstance(r, TestRenderer)
        assert r.name == "test_renderer"


class TestRendererRegistry:
    def test_available_renderers(self) -> None:
        available = available_renderers()
        assert "json" in available
        assert "markdown" in available
        assert "html" in available
        assert "csv" in available

    def test_get_renderer_class(self) -> None:
        cls = get_renderer_class("json")
        assert cls is JSONRenderer

    def test_get_renderer_class_unknown(self) -> None:
        with pytest.raises(KeyError, match="Unknown renderer"):
            get_renderer_class("unknown")

    def test_create_renderer(self) -> None:
        r = create_renderer("markdown")
        assert isinstance(r, MarkdownRenderer)
        assert r.name == "markdown"

    def test_format_property(self) -> None:
        assert JSONRenderer().format == ReportFormat.json
        assert MarkdownRenderer().format == ReportFormat.markdown
        assert HTMLRenderer().format == ReportFormat.html
        assert CSVRenderer().format == ReportFormat.csv


class TestJSONRenderer:
    def test_render_minimal(self, config: PresentationConfig) -> None:
        model = _make_model()
        output = JSONRenderer().render(model, config)
        data = json.loads(output)
        assert data["title"] == "Test Report"
        assert data["report_type"] == "executive_summary"

    def test_render_with_sections(self, config: PresentationConfig) -> None:
        section = _make_section(content={"findings": ["Finding 1", "Finding 2"]})
        model = _make_model(sections=[section])
        output = JSONRenderer().render(model, config)
        data = json.loads(output)
        assert len(data["sections"]) == 1
        assert data["sections"][0]["content"]["findings"] == ["Finding 1", "Finding 2"]

    def test_render_with_assets(self, config: PresentationConfig) -> None:
        assets = ReportAssets(metrics={"score": 85.0, "count": 42})
        model = _make_model(assets=assets)
        output = JSONRenderer().render(model, config)
        data = json.loads(output)
        assert data["assets"]["metrics"]["score"] == 85.0

    def test_render_round_trip(self, config: PresentationConfig) -> None:
        model = _make_model(
            tags=["ai"],
            companies=["Acme"],
            summaries=ReportSummaries(one_sentence="Test"),
        )
        output = JSONRenderer().render(model, config)
        data = json.loads(output)
        assert "ai" in data["tags"]
        assert data["summaries"]["one_sentence"] == "Test"

    def test_render_with_charts(self, config: PresentationConfig) -> None:
        chart = ChartSpec(chart_type=ChartType.bar, title="Bar")
        section = _make_section(
            section_type=SectionType.charts,
            title="Charts",
            charts=[chart],
        )
        model = _make_model(sections=[section])
        output = JSONRenderer().render(model, config)
        data = json.loads(output)
        assert len(data["sections"][0]["charts"]) == 1

    def test_valid_json(self, config: PresentationConfig) -> None:
        model = _make_model()
        output = JSONRenderer().render(model, config)
        json.loads(output)


class TestMarkdownRenderer:
    def test_render_minimal(self, config: PresentationConfig) -> None:
        model = _make_model()
        output = MarkdownRenderer().render(model, config)
        assert "# Test Report" in output
        assert "A test" in output
        assert model.report_id in output

    def test_render_with_section(self, config: PresentationConfig) -> None:
        section = _make_section(
            section_type=SectionType.executive_summary,
            title="Executive Summary",
            content={"summary": "This is a summary."},
        )
        model = _make_model(sections=[section])
        output = MarkdownRenderer().render(model, config)
        assert "## Executive Summary" in output
        assert "This is a summary." in output

    def test_render_with_bullets(self, config: PresentationConfig) -> None:
        section = _make_section(
            content={"findings": ["Finding 1", "Finding 2"]},
        )
        model = _make_model(sections=[section])
        output = MarkdownRenderer().render(model, config)
        assert "- Finding 1" in output
        assert "- Finding 2" in output

    def test_render_with_charts(self, config: PresentationConfig) -> None:
        chart = ChartSpec(chart_type=ChartType.line, title="Growth")
        section = _make_section(charts=[chart])
        model = _make_model(sections=[section])
        output = MarkdownRenderer().render(model, config)
        assert "Chart: Growth" in output
        assert "Line" in output

    def test_render_with_highlights(self, config: PresentationConfig) -> None:
        from phase3.presentation.schema import Highlight
        hl = Highlight(text="Key finding", source="trend", score=0.95, section=SectionType.top_findings)
        assets = ReportAssets(highlights=[hl])
        model = _make_model(assets=assets)
        output = MarkdownRenderer().render(model, config)
        assert "Key finding" in output
        assert "0.95" in output

    def test_render_with_tables(self, config: PresentationConfig) -> None:
        table = TableSpec(title="Data", headers=["A", "B"], rows=[["1", "2"]])
        assets = ReportAssets(tables=[table])
        model = _make_model(assets=assets)
        output = MarkdownRenderer().render(model, config)
        assert "| A | B |" in output
        assert "| 1 | 2 |" in output
        assert "| --- | --- |" in output

    def test_render_with_metrics(self, config: PresentationConfig) -> None:
        assets = ReportAssets(metrics={"total": 42.0, "avg": 3.5})
        model = _make_model(assets=assets)
        output = MarkdownRenderer().render(model, config)
        assert "| total | 42.0 |" in output or "| total | 42 |" in output
        assert "avg" in output

    def test_render_with_distribution(self, config: PresentationConfig) -> None:
        section = _make_section(
            section_type=SectionType.trend_analysis,
            content={"distribution": {"growing": 5, "declining": 2}},
        )
        model = _make_model(sections=[section])
        output = MarkdownRenderer().render(model, config)
        assert "| growing | 5 |" in output

    def test_render_with_provenance(self, config: PresentationConfig) -> None:
        section = _make_section(
            provenance={"confidence": 0.85},
        )
        model = _make_model(sections=[section])
        output = MarkdownRenderer().render(model, config)
        assert "85%" in output

    def test_render_with_lineage(self, config: PresentationConfig) -> None:
        lineage = SourceLineage(knowledge_graph_run_id="kg_1", trend_run_id="tr_1")
        model = _make_model(lineage=lineage)
        output = MarkdownRenderer().render(model, config)
        assert "kg_1" in output
        assert "tr_1" in output

    def test_render_with_summaries(self, config: PresentationConfig) -> None:
        summaries = ReportSummaries(
            one_paragraph="Paragraph summary",
            five_bullets=["Bullet 1", "Bullet 2"],
        )
        model = _make_model(summaries=summaries)
        output = MarkdownRenderer().render(model, config)
        assert "Paragraph summary" in output
        assert "- Bullet 1" in output

    def test_render_empty_model(self, config: PresentationConfig) -> None:
        model = _make_model()
        output = MarkdownRenderer().render(model, config)
        assert len(output) > 0


class TestHTMLRenderer:
    def test_render_minimal(self, config: PresentationConfig) -> None:
        model = _make_model()
        output = HTMLRenderer().render(model, config)
        assert "Test Report" in output
        assert "<!DOCTYPE html>" in output

    def test_render_with_section(self, config: PresentationConfig) -> None:
        section = _make_section(title="My Section")
        model = _make_model(sections=[section])
        output = HTMLRenderer().render(model, config)
        assert "My Section" in output

    def test_render_with_chart_placeholder(self, config: PresentationConfig) -> None:
        chart = ChartSpec(chart_type=ChartType.bar, title="Bar Chart")
        section = _make_section(section_type=SectionType.charts, charts=[chart])
        model = _make_model(sections=[section])
        output = HTMLRenderer().render(model, config)
        assert "Bar Chart" in output
        assert "bar" in output.lower()

    def test_render_with_highlights(self, config: PresentationConfig) -> None:
        from phase3.presentation.schema import Highlight
        hl = Highlight(text="Top finding", source="opp", score=0.9, section=SectionType.top_findings)
        assets = ReportAssets(highlights=[hl])
        model = _make_model(assets=assets)
        output = HTMLRenderer().render(model, config)
        assert "Top finding" in output

    def test_render_with_metrics(self, config: PresentationConfig) -> None:
        assets = ReportAssets(metrics={"total": 42.0})
        model = _make_model(assets=assets)
        output = HTMLRenderer().render(model, config)
        assert "42" in output

    def test_render_valid_html(self, config: PresentationConfig) -> None:
        model = _make_model()
        output = HTMLRenderer().render(model, config)
        assert output.startswith("<!DOCTYPE html>")
        assert "</html>" in output

    def test_render_empty_model(self, config: PresentationConfig) -> None:
        model = _make_model()
        output = HTMLRenderer().render(model, config)
        assert len(output) > 0


class TestCSVRenderer:
    def test_render_minimal(self, config: PresentationConfig) -> None:
        model = _make_model()
        output = CSVRenderer().render(model, config)
        assert len(output) > 0
        assert "Report ID" in output

    def test_render_with_findings(self, config: PresentationConfig) -> None:
        section = _make_section(
            content={"findings": ["Finding 1", "Finding 2"]},
        )
        model = _make_model(sections=[section])
        output = CSVRenderer().render(model, config)
        assert "Finding 1" in output
        assert "Finding 2" in output

    def test_render_with_recommendations(self, config: PresentationConfig) -> None:
        section = _make_section(
            content={"recommendations": ["Rec A", "Rec B"]},
        )
        model = _make_model(sections=[section])
        output = CSVRenderer().render(model, config)
        assert "Rec A" in output
        assert "Rec B" in output

    def test_render_with_root_causes(self, config: PresentationConfig) -> None:
        section = _make_section(
            content={"root_causes": ["Cause X"]},
        )
        model = _make_model(sections=[section])
        output = CSVRenderer().render(model, config)
        assert "Cause X" in output

    def test_render_with_structured_items(self, config: PresentationConfig) -> None:
        section = _make_section(
            content={
                "items": [
                    {"title": "Item 1", "score": 0.9, "source": "trend"},
                    {"title": "Item 2", "score": 0.5},
                ]
            },
        )
        model = _make_model(sections=[section])
        output = CSVRenderer().render(model, config)
        assert "Item 1" in output
        assert "Item 2" in output

    def test_render_empty_sections(self, config: PresentationConfig) -> None:
        model = _make_model()
        output = CSVRenderer().render(model, config)
        assert "Report ID" in output
        assert "Generated" in output

    def test_valid_csv(self, config: PresentationConfig) -> None:
        import csv
        import io
        section = _make_section(
            content={"findings": ["F1", "F2"]},
        )
        model = _make_model(sections=[section])
        output = CSVRenderer().render(model, config)
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)
        assert len(rows) >= 2
        assert rows[0][0] == "Report ID"


class TestRendererIntegration:
    def test_all_renderers_creatable(self) -> None:
        for name in available_renderers():
            r = create_renderer(name)
            assert r.name == name
            assert isinstance(r.format, ReportFormat)

    def test_all_renderers_render(self, config: PresentationConfig) -> None:
        model = _make_model()
        for name in available_renderers():
            r = create_renderer(name)
            output = r.render(model, config)
            assert isinstance(output, str)
            assert len(output) > 0

    def test_consistent_report_id(self, config: PresentationConfig) -> None:
        model = _make_model()
        for name in available_renderers():
            r = create_renderer(name)
            output = r.render(model, config)
            assert isinstance(output, str)
            assert len(output) > 0

    def test_all_renderers_with_full_model(self, config: PresentationConfig) -> None:
        chart = ChartSpec(chart_type=ChartType.bar, title="B")
        section = _make_section(
            section_type=SectionType.trend_analysis,
            title="Trends",
            content={"distribution": {"growing": 5}},
            charts=[chart],
            summaries={"one_paragraph": "Trend summary"},
            provenance={"confidence": 0.9},
        )
        from phase3.presentation.schema import Highlight
        hl = Highlight(text="H", source="t", score=0.8, section=SectionType.top_findings)
        assets = ReportAssets(
            metrics={"score": 85.0},
            highlights=[hl],
            tables=[TableSpec(title="T", headers=["A"], rows=[["1"]])],
        )
        model = _make_model(
            sections=[section],
            assets=assets,
            summaries=ReportSummaries(one_paragraph="Global summary"),
            lineage=SourceLineage(knowledge_graph_run_id="kg_1"),
        )

        for name in available_renderers():
            r = create_renderer(name)
            output = r.render(model, config)
            assert isinstance(output, str)
            assert len(output) > 0
