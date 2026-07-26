from __future__ import annotations

import importlib
import json

import pytest

from phase3.presentation.config import PresentationConfig
from phase3.presentation.renderers import PlotlyRenderer, available_renderers, create_renderer
from phase3.presentation.schema import (
    ChartSeries,
    ChartSpec,
    ChartType,
    PresentationModel,
    ReportAssets,
    ReportSection,
    ReportType,
    SectionType,
)

_plotly_available = importlib.util.find_spec("plotly") is not None


@pytest.fixture
def config() -> PresentationConfig:
    return PresentationConfig()


@pytest.mark.skipif(not _plotly_available, reason="plotly not installed")
class TestPlotlyRenderer:
    def test_name(self) -> None:
        assert PlotlyRenderer().name == "plotly"

    def test_format(self) -> None:
        assert PlotlyRenderer().format.value == "json"

    def test_render_empty_model(self, config: PresentationConfig) -> None:
        model = PresentationModel(report_type=ReportType.executive_summary, title="T")
        output = PlotlyRenderer().render(model, config)
        data = json.loads(output)
        assert "charts" in data
        assert data["charts"] == {}

    def test_render_section_chart(self, config: PresentationConfig) -> None:
        chart = ChartSpec(
            chart_type=ChartType.bar,
            title="Bar",
            labels=["A", "B"],
            series=[ChartSeries(name="V", values=[1, 2])],
        )
        section = ReportSection(section_type=SectionType.charts, title="C", order=0, charts=[chart])
        model = PresentationModel(report_type=ReportType.executive_summary, title="T", sections=[section])
        output = PlotlyRenderer().render(model, config)
        data = json.loads(output)
        assert len(data["charts"]) == 1

    def test_render_asset_chart(self, config: PresentationConfig) -> None:
        chart = ChartSpec(
            chart_type=ChartType.pie,
            title="Pie",
            labels=["X", "Y"],
            series=[ChartSeries(name="V", values=[3, 7])],
        )
        model = PresentationModel(
            report_type=ReportType.executive_summary,
            title="T",
            assets=ReportAssets(charts=[chart]),
        )
        output = PlotlyRenderer().render(model, config)
        data = json.loads(output)
        assert len(data["charts"]) == 1

    def test_bar_to_plotly_json(self, config: PresentationConfig) -> None:
        spec = ChartSpec(
            chart_type=ChartType.bar,
            title="Bar",
            labels=["A", "B"],
            series=[ChartSeries(name="V", values=[1, 2])],
        )
        fig = PlotlyRenderer().chart_to_plotly_json(spec, config)
        assert fig is not None
        assert fig["data"][0]["type"] == "bar"

    def test_bar_horizontal(self, config: PresentationConfig) -> None:
        spec = ChartSpec(
            chart_type=ChartType.bar,
            title="HBar",
            labels=["A"],
            series=[ChartSeries(name="V", values=[5])],
            metadata={"orientation": "h"},
        )
        fig = PlotlyRenderer().chart_to_plotly_json(spec, config)
        assert fig is not None
        assert fig["data"][0]["type"] == "bar"
        assert fig["data"][0]["x"] == [5]
        assert fig["data"][0]["y"] == ["A"]

    def test_line_to_plotly_json(self, config: PresentationConfig) -> None:
        spec = ChartSpec(
            chart_type=ChartType.line,
            title="Line",
            labels=["X", "Y", "Z"],
            series=[ChartSeries(name="S", values=[1, 2, 3])],
        )
        fig = PlotlyRenderer().chart_to_plotly_json(spec, config)
        assert fig is not None
        assert fig["data"][0]["type"] == "scatter"

    def test_line_multi_series(self, config: PresentationConfig) -> None:
        spec = ChartSpec(
            chart_type=ChartType.line,
            title="Multi",
            labels=["A", "B"],
            series=[
                ChartSeries(name="S1", values=[1, 2]),
                ChartSeries(name="S2", values=[3, 4]),
            ],
        )
        fig = PlotlyRenderer().chart_to_plotly_json(spec, config)
        assert fig is not None
        assert len(fig["data"]) == 2

    def test_pie_to_plotly_json(self, config: PresentationConfig) -> None:
        spec = ChartSpec(
            chart_type=ChartType.pie,
            title="Pie",
            labels=["A", "B"],
            series=[ChartSeries(name="V", values=[3, 7])],
        )
        fig = PlotlyRenderer().chart_to_plotly_json(spec, config)
        assert fig is not None
        assert fig["data"][0]["type"] == "pie"

    def test_pie_donut(self, config: PresentationConfig) -> None:
        spec = ChartSpec(
            chart_type=ChartType.pie,
            title="Donut",
            labels=["A", "B"],
            series=[ChartSeries(name="V", values=[4, 6])],
            metadata={"hole": 0.4},
        )
        fig = PlotlyRenderer().chart_to_plotly_json(spec, config)
        assert fig is not None
        assert fig["data"][0]["hole"] == 0.4

    def test_timeline_to_plotly_json(self, config: PresentationConfig) -> None:
        spec = ChartSpec(
            chart_type=ChartType.timeline,
            title="Timeline",
            labels=["Event"],
            series=[ChartSeries(name="Event", values=["2024-01", "2024-06"])],
        )
        fig = PlotlyRenderer().chart_to_plotly_json(spec, config)
        assert fig is not None

    def test_heatmap_to_plotly_json(self, config: PresentationConfig) -> None:
        spec = ChartSpec(
            chart_type=ChartType.heatmap,
            title="Heat",
            labels=["X", "Y"],
            series=[
                ChartSeries(name="R1", values=[0.5, 0.8]),
                ChartSeries(name="R2", values=[0.2, 0.9]),
            ],
            metadata={"colorscale": "Viridis"},
        )
        fig = PlotlyRenderer().chart_to_plotly_json(spec, config)
        assert fig is not None
        assert fig["data"][0]["type"] == "heatmap"

    def test_treemap_to_plotly_json(self, config: PresentationConfig) -> None:
        spec = ChartSpec(
            chart_type=ChartType.treemap,
            title="Tree",
            labels=["Root", "A", "B"],
            series=[ChartSeries(name="S", values=[100, 60, 40])],
            metadata={"parents": ["", "Root", "Root"]},
        )
        fig = PlotlyRenderer().chart_to_plotly_json(spec, config)
        assert fig is not None
        assert fig["data"][0]["type"] == "treemap"

    def test_sankey_to_plotly_json(self, config: PresentationConfig) -> None:
        spec = ChartSpec(
            chart_type=ChartType.sankey,
            title="Flow",
            labels=["A", "B", "X", "Y"],
            series=[
                ChartSeries(name="Source", values=["A", "A", "B"]),
                ChartSeries(name="Target", values=["X", "Y", "Y"]),
                ChartSeries(name="Flow", values=[10, 5, 8]),
            ],
        )
        fig = PlotlyRenderer().chart_to_plotly_json(spec, config)
        assert fig is not None
        assert fig["data"][0]["type"] == "sankey"

    def test_chart_to_html(self, config: PresentationConfig) -> None:
        spec = ChartSpec(
            chart_type=ChartType.bar,
            title="Bar",
            labels=["A"],
            series=[ChartSeries(name="V", values=[1])],
        )
        html = PlotlyRenderer().chart_to_html(spec, config)
        assert isinstance(html, str)
        assert len(html) > 0
        assert "<div" in html

    def test_all_chart_types_build(self, config: PresentationConfig) -> None:
        cases: list[tuple[ChartType, ChartSpec]] = [
            (ChartType.bar, ChartSpec(chart_type=ChartType.bar, title="B", labels=["A"], series=[ChartSeries(name="V", values=[1])])),
            (ChartType.line, ChartSpec(chart_type=ChartType.line, title="L", labels=["A"], series=[ChartSeries(name="V", values=[1])])),
            (ChartType.pie, ChartSpec(chart_type=ChartType.pie, title="P", labels=["A"], series=[ChartSeries(name="V", values=[1])])),
            (ChartType.timeline, ChartSpec(chart_type=ChartType.timeline, title="T", labels=["E"], series=[ChartSeries(name="E", values=["2024-01", "2024-06"])])),
            (ChartType.heatmap, ChartSpec(chart_type=ChartType.heatmap, title="H", labels=["X"], series=[ChartSeries(name="R", values=[0.5])])),
            (ChartType.treemap, ChartSpec(chart_type=ChartType.treemap, title="Tr", labels=["R"], series=[ChartSeries(name="S", values=[100])], metadata={"parents": [""]})),
            (ChartType.sankey, ChartSpec(chart_type=ChartType.sankey, title="S", labels=["A", "B"], series=[ChartSeries(name="S", values=["A"]), ChartSeries(name="T", values=["B"]), ChartSeries(name="F", values=[1])])),
        ]

        for chart_type, spec in cases:
            fig = PlotlyRenderer().chart_to_plotly_json(spec, config)
            assert fig is not None, f"Failed to build {chart_type}"

    def test_config_applied_to_layout(self, config: PresentationConfig) -> None:
        spec = ChartSpec(chart_type=ChartType.bar, title="B", labels=["A"], series=[ChartSeries(name="V", values=[1])])
        fig = PlotlyRenderer().chart_to_plotly_json(spec, config)
        assert fig is not None
        assert fig["layout"]["width"] == config.chart_width
        assert fig["layout"]["height"] == config.chart_height


class TestPlotlyInRegistry:
    def test_plotly_available(self) -> None:
        available = available_renderers()
        assert "plotly" in available

    def test_plotly_creatable(self) -> None:
        r = create_renderer("plotly")
        assert isinstance(r, PlotlyRenderer)
