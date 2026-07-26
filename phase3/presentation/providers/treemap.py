from __future__ import annotations

from typing import Any

from phase3.presentation.config import PresentationConfig
from phase3.presentation.providers.base import ChartProvider
from phase3.presentation.providers.registry import register_chart_provider
from phase3.presentation.schema import ChartSeries, ChartSpec, ChartType


@register_chart_provider(name="treemap", priority=50)
class TreemapChartProvider(ChartProvider):
    @property
    def name(self) -> str:
        return "treemap"

    @property
    def chart_type(self) -> ChartType:
        return ChartType.treemap

    @property
    def priority(self) -> int:
        return 50

    def build(self, data: dict[str, Any], config: PresentationConfig) -> ChartSpec:
        labels: list[str] = data.get("labels", [])
        parents: list[str] = data.get("parents", [])
        values: list[float | int | str] = data.get("values", [])
        title: str = data.get("title", "Treemap")
        description: str = data.get("description", "")

        series = [ChartSeries(name="Size", values=values)]

        return ChartSpec(
            chart_type=ChartType.treemap,
            title=title,
            description=description,
            series=series,
            labels=labels,
            metadata={
                "parents": parents,
                "branch_values": data.get("branch_values", "total"),
            },
        )
