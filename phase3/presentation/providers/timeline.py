from __future__ import annotations

from typing import Any

from phase3.presentation.config import PresentationConfig
from phase3.presentation.providers.base import ChartProvider
from phase3.presentation.providers.registry import register_chart_provider
from phase3.presentation.schema import ChartSeries, ChartSpec, ChartType


@register_chart_provider(name="timeline", priority=70)
class TimelineChartProvider(ChartProvider):
    @property
    def name(self) -> str:
        return "timeline"

    @property
    def chart_type(self) -> ChartType:
        return ChartType.timeline

    @property
    def priority(self) -> int:
        return 70

    def build(self, data: dict[str, Any], config: PresentationConfig) -> ChartSpec:
        items: list[dict[str, Any]] = data.get("items", [])
        title: str = data.get("title", "Timeline Chart")
        description: str = data.get("description", "")

        series = [
            ChartSeries(
                name=item.get("label", f"Item {i}"),
                values=[item.get("start", ""), item.get("end", "")],
                color=item.get("color"),
            )
            for i, item in enumerate(items)
        ]

        labels = [item.get("label", f"Item {i}") for i, item in enumerate(items)]

        return ChartSpec(
            chart_type=ChartType.timeline,
            title=title,
            description=description,
            series=series,
            labels=labels,
            metadata={"date_format": data.get("date_format", "%Y-%m-%d")},
        )
