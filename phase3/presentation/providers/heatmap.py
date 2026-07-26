from __future__ import annotations

from typing import Any

from phase3.presentation.config import PresentationConfig
from phase3.presentation.providers.base import ChartProvider
from phase3.presentation.providers.registry import register_chart_provider
from phase3.presentation.schema import ChartSeries, ChartSpec, ChartType


@register_chart_provider(name="heatmap", priority=60)
class HeatmapChartProvider(ChartProvider):
    @property
    def name(self) -> str:
        return "heatmap"

    @property
    def chart_type(self) -> ChartType:
        return ChartType.heatmap

    @property
    def priority(self) -> int:
        return 60

    def build(self, data: dict[str, Any], config: PresentationConfig) -> ChartSpec:
        x_labels: list[str] = data.get("x_labels", [])
        y_labels: list[str] = data.get("y_labels", [])
        values: list[list[float]] = data.get("values", [])
        title: str = data.get("title", "Heatmap")
        description: str = data.get("description", "")

        series: list[ChartSeries] = []
        for i, row in enumerate(values):
            y_label = y_labels[i] if i < len(y_labels) else f"Row {i}"
            series.append(ChartSeries(name=y_label, values=[float(v) for v in row]))

        return ChartSpec(
            chart_type=ChartType.heatmap,
            title=title,
            description=description,
            series=series,
            labels=x_labels,
            metadata={
                "colorscale": data.get("colorscale", "Viridis"),
                "x_labels": x_labels,
                "y_labels": y_labels,
            },
        )
