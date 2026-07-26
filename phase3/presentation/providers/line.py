from __future__ import annotations

from typing import Any

from phase3.presentation.config import PresentationConfig
from phase3.presentation.providers.base import ChartProvider
from phase3.presentation.providers.registry import register_chart_provider
from phase3.presentation.schema import ChartSeries, ChartSpec, ChartType


@register_chart_provider(name="line", priority=90)
class LineChartProvider(ChartProvider):
    @property
    def name(self) -> str:
        return "line"

    @property
    def chart_type(self) -> ChartType:
        return ChartType.line

    @property
    def priority(self) -> int:
        return 90

    def build(self, data: dict[str, Any], config: PresentationConfig) -> ChartSpec:
        labels: list[str] = data.get("labels", [])
        raw_series: list[dict[str, Any]] = data.get("series", [])
        title: str = data.get("title", "Line Chart")
        description: str = data.get("description", "")

        series = [
            ChartSeries(name=s.get("name", f"Series {i}"), values=s.get("values", []))
            for i, s in enumerate(raw_series)
        ]

        return ChartSpec(
            chart_type=ChartType.line,
            title=title,
            description=description,
            series=series,
            labels=labels,
            metadata={
                "show_markers": data.get("show_markers", True),
                "smooth": data.get("smooth", False),
            },
        )
