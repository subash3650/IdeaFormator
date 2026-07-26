from __future__ import annotations

from typing import Any

from phase3.presentation.config import PresentationConfig
from phase3.presentation.providers.base import ChartProvider
from phase3.presentation.providers.registry import register_chart_provider
from phase3.presentation.schema import ChartSeries, ChartSpec, ChartType


@register_chart_provider(name="pie", priority=80)
class PieChartProvider(ChartProvider):
    @property
    def name(self) -> str:
        return "pie"

    @property
    def chart_type(self) -> ChartType:
        return ChartType.pie

    @property
    def priority(self) -> int:
        return 80

    def build(self, data: dict[str, Any], config: PresentationConfig) -> ChartSpec:
        labels: list[str] = data.get("labels", [])
        values: list[float | int | str] = data.get("values", [])
        series_name: str = data.get("series_name", "Values")
        title: str = data.get("title", "Pie Chart")
        description: str = data.get("description", "")

        series = [ChartSeries(name=series_name, values=values)]

        return ChartSpec(
            chart_type=ChartType.pie,
            title=title,
            description=description,
            series=series,
            labels=labels,
            metadata={
                "hole": data.get("hole", 0.0),
                "show_percent": data.get("show_percent", True),
            },
        )
