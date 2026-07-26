from __future__ import annotations

from typing import Any

from phase3.presentation.config import PresentationConfig
from phase3.presentation.providers.base import ChartProvider
from phase3.presentation.providers.registry import register_chart_provider
from phase3.presentation.schema import ChartSeries, ChartSpec, ChartType


@register_chart_provider(name="bar", priority=100)
class BarChartProvider(ChartProvider):
    @property
    def name(self) -> str:
        return "bar"

    @property
    def chart_type(self) -> ChartType:
        return ChartType.bar

    @property
    def priority(self) -> int:
        return 100

    def build(self, data: dict[str, Any], config: PresentationConfig) -> ChartSpec:
        labels: list[str] = data.get("labels", [])
        values: list[float | int | str] = data.get("values", [])
        series_name: str = data.get("series_name", "Values")
        title: str = data.get("title", "Bar Chart")
        description: str = data.get("description", "")
        orientation: str = data.get("orientation", "v")

        series = [ChartSeries(name=series_name, values=values)]

        return ChartSpec(
            chart_type=ChartType.bar,
            title=title,
            description=description,
            series=series,
            labels=labels,
            metadata={"orientation": orientation, "stacked": data.get("stacked", False)},
        )
