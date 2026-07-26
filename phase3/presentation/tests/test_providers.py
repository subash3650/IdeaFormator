from __future__ import annotations

import pytest

from phase3.presentation.config import PresentationConfig
from phase3.presentation.providers import (
    BarChartProvider,
    HeatmapChartProvider,
    LineChartProvider,
    PieChartProvider,
    SankeyChartProvider,
    TimelineChartProvider,
    TreemapChartProvider,
    available_chart_providers,
    chart_provider_priority,
    create_chart_provider,
    get_chart_provider_class,
    sorted_chart_providers,
)
from phase3.presentation.schema import ChartType


@pytest.fixture
def config() -> PresentationConfig:
    return PresentationConfig()


class TestBarChartProvider:
    def test_name(self) -> None:
        assert BarChartProvider().name == "bar"

    def test_chart_type(self) -> None:
        assert BarChartProvider().chart_type == ChartType.bar

    def test_build_simple(self, config: PresentationConfig) -> None:
        spec = BarChartProvider().build(
            {"labels": ["A", "B", "C"], "values": [10, 20, 15], "title": "Test Bar"},
            config,
        )
        assert spec.chart_type == ChartType.bar
        assert spec.title == "Test Bar"
        assert spec.labels == ["A", "B", "C"]
        assert len(spec.series) == 1
        assert spec.series[0].values == [10, 20, 15]
        assert spec.metadata["orientation"] == "v"

    def test_build_horizontal(self, config: PresentationConfig) -> None:
        spec = BarChartProvider().build(
            {"labels": ["X", "Y"], "values": [5, 8], "orientation": "h"},
            config,
        )
        assert spec.metadata["orientation"] == "h"

    def test_build_stacked(self, config: PresentationConfig) -> None:
        spec = BarChartProvider().build(
            {"labels": ["A", "B"], "values": [1, 2], "stacked": True},
            config,
        )
        assert spec.metadata["stacked"] is True

    def test_build_empty(self, config: PresentationConfig) -> None:
        spec = BarChartProvider().build({}, config)
        assert spec.labels == []
        assert spec.series[0].values == []


class TestLineChartProvider:
    def test_name(self) -> None:
        assert LineChartProvider().name == "line"

    def test_chart_type(self) -> None:
        assert LineChartProvider().chart_type == ChartType.line

    def test_build_single_series(self, config: PresentationConfig) -> None:
        spec = LineChartProvider().build(
            {
                "labels": ["Jan", "Feb", "Mar"],
                "series": [{"name": "Growth", "values": [0.5, 0.7, 0.9]}],
                "title": "Test Line",
            },
            config,
        )
        assert spec.title == "Test Line"
        assert len(spec.series) == 1
        assert spec.series[0].name == "Growth"
        assert spec.series[0].values == [0.5, 0.7, 0.9]
        assert spec.labels == ["Jan", "Feb", "Mar"]

    def test_build_multi_series(self, config: PresentationConfig) -> None:
        spec = LineChartProvider().build(
            {
                "series": [
                    {"name": "Trend A", "values": [1, 2, 3]},
                    {"name": "Trend B", "values": [3, 2, 1]},
                ]
            },
            config,
        )
        assert len(spec.series) == 2
        assert spec.series[0].name == "Trend A"
        assert spec.series[1].name == "Trend B"

    def test_build_empty(self, config: PresentationConfig) -> None:
        spec = LineChartProvider().build({}, config)
        assert spec.series == []
        assert spec.labels == []


class TestPieChartProvider:
    def test_name(self) -> None:
        assert PieChartProvider().name == "pie"

    def test_chart_type(self) -> None:
        assert PieChartProvider().chart_type == ChartType.pie

    def test_build(self, config: PresentationConfig) -> None:
        spec = PieChartProvider().build(
            {
                "labels": ["Large", "Medium", "Small"],
                "values": [10, 5, 3],
                "title": "Market Size",
            },
            config,
        )
        assert spec.title == "Market Size"
        assert spec.labels == ["Large", "Medium", "Small"]
        assert spec.series[0].values == [10, 5, 3]
        assert spec.metadata["hole"] == 0.0
        assert spec.metadata["show_percent"] is True

    def test_build_donut(self, config: PresentationConfig) -> None:
        spec = PieChartProvider().build(
            {"labels": ["A", "B"], "values": [60, 40], "hole": 0.4},
            config,
        )
        assert spec.metadata["hole"] == 0.4

    def test_build_empty(self, config: PresentationConfig) -> None:
        spec = PieChartProvider().build({}, config)
        assert spec.labels == []


class TestTimelineChartProvider:
    def test_name(self) -> None:
        assert TimelineChartProvider().name == "timeline"

    def test_chart_type(self) -> None:
        assert TimelineChartProvider().chart_type == ChartType.timeline

    def test_build(self, config: PresentationConfig) -> None:
        spec = TimelineChartProvider().build(
            {
                "items": [
                    {"label": "AI Growth", "start": "2024-01", "end": "2024-06"},
                    {"label": "Cloud Boom", "start": "2024-03", "end": "2024-09"},
                ],
                "title": "Timeline",
            },
            config,
        )
        assert spec.title == "Timeline"
        assert len(spec.series) == 2
        assert spec.series[0].name == "AI Growth"
        assert spec.series[1].name == "Cloud Boom"
        assert spec.labels == ["AI Growth", "Cloud Boom"]

    def test_build_with_colors(self, config: PresentationConfig) -> None:
        spec = TimelineChartProvider().build(
            {
                "items": [
                    {"label": "Trend", "start": "2024-01", "end": "2024-06", "color": "#ff0000"},
                ]
            },
            config,
        )
        assert spec.series[0].color == "#ff0000"

    def test_build_empty(self, config: PresentationConfig) -> None:
        spec = TimelineChartProvider().build({}, config)
        assert spec.series == []


class TestHeatmapChartProvider:
    def test_name(self) -> None:
        assert HeatmapChartProvider().name == "heatmap"

    def test_chart_type(self) -> None:
        assert HeatmapChartProvider().chart_type == ChartType.heatmap

    def test_build(self, config: PresentationConfig) -> None:
        spec = HeatmapChartProvider().build(
            {
                "x_labels": ["A", "B", "C"],
                "y_labels": ["X", "Y"],
                "values": [[0.8, 0.6, 0.9], [0.5, 0.7, 0.3]],
                "title": "Heatmap",
            },
            config,
        )
        assert spec.title == "Heatmap"
        assert spec.labels == ["A", "B", "C"]
        assert len(spec.series) == 2
        assert spec.series[0].name == "X"
        assert spec.series[0].values == [0.8, 0.6, 0.9]
        assert spec.series[1].name == "Y"
        assert spec.series[1].values == [0.5, 0.7, 0.3]
        assert spec.metadata["colorscale"] == "Viridis"

    def test_build_empty(self, config: PresentationConfig) -> None:
        spec = HeatmapChartProvider().build({}, config)
        assert spec.series == []


class TestTreemapChartProvider:
    def test_name(self) -> None:
        assert TreemapChartProvider().name == "treemap"

    def test_chart_type(self) -> None:
        assert TreemapChartProvider().chart_type == ChartType.treemap

    def test_build(self, config: PresentationConfig) -> None:
        spec = TreemapChartProvider().build(
            {
                "labels": ["Tech", "AI", "ML", "Cloud"],
                "parents": ["", "Tech", "Tech", "Tech"],
                "values": [100, 60, 40, 30],
                "title": "Treemap",
            },
            config,
        )
        assert spec.title == "Treemap"
        assert spec.labels == ["Tech", "AI", "ML", "Cloud"]
        assert spec.metadata["parents"] == ["", "Tech", "Tech", "Tech"]
        assert spec.series[0].values == [100, 60, 40, 30]

    def test_build_empty(self, config: PresentationConfig) -> None:
        spec = TreemapChartProvider().build({}, config)
        assert spec.labels == []
        assert spec.metadata["parents"] == []


class TestSankeyChartProvider:
    def test_name(self) -> None:
        assert SankeyChartProvider().name == "sankey"

    def test_chart_type(self) -> None:
        assert SankeyChartProvider().chart_type == ChartType.sankey

    def test_build(self, config: PresentationConfig) -> None:
        spec = SankeyChartProvider().build(
            {
                "source": ["A", "A", "B"],
                "target": ["X", "Y", "Z"],
                "value": [10, 5, 8],
                "labels": ["A", "B", "X", "Y", "Z"],
                "title": "Sankey",
            },
            config,
        )
        assert spec.title == "Sankey"
        assert spec.labels == ["A", "B", "X", "Y", "Z"]
        assert len(spec.series) == 3
        assert spec.series[0].name == "Source"
        assert spec.series[0].values == ["A", "A", "B"]
        assert spec.series[1].values == ["X", "Y", "Z"]
        assert spec.series[2].values == [10.0, 5.0, 8.0]

    def test_build_empty(self, config: PresentationConfig) -> None:
        spec = SankeyChartProvider().build({}, config)
        assert spec.series[0].values == []


class TestProviderRegistry:
    def test_available_providers(self) -> None:
        available = available_chart_providers()
        assert "bar" in available
        assert "line" in available
        assert "pie" in available
        assert "timeline" in available
        assert "heatmap" in available
        assert "treemap" in available
        assert "sankey" in available
        assert len(available) == 7

    def test_sorted_by_priority(self) -> None:
        sorted_names = sorted_chart_providers()
        assert sorted_names[0] == "bar"  # priority 100
        assert sorted_names[1] == "line"  # priority 90

    def test_get_provider_class(self) -> None:
        cls = get_chart_provider_class("bar")
        assert cls is BarChartProvider

    def test_get_provider_class_unknown(self) -> None:
        with pytest.raises(KeyError, match="Unknown chart provider"):
            get_chart_provider_class("unknown")

    def test_create_provider(self) -> None:
        provider = create_chart_provider("bar")
        assert isinstance(provider, BarChartProvider)
        assert provider.name == "bar"

    def test_create_provider_unknown(self) -> None:
        with pytest.raises(KeyError):
            create_chart_provider("unknown")

    def test_priority(self) -> None:
        assert chart_provider_priority("bar") == 100
        assert chart_provider_priority("sankey") == 40
        assert chart_provider_priority("unknown") == 100

    def test_all_providers_creatable(self) -> None:
        for name in available_chart_providers():
            provider = create_chart_provider(name)
            assert provider.name == name
            assert isinstance(provider.chart_type, ChartType)
            assert provider.priority > 0

    def test_all_providers_build(self, config: PresentationConfig) -> None:
        for name in available_chart_providers():
            provider = create_chart_provider(name)
            spec = provider.build({"title": f"Test {name}"}, config)
            assert spec.chart_type == provider.chart_type
            assert spec.title == f"Test {name}"
