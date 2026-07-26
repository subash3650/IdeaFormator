from __future__ import annotations

from typing import Any

from phase3.presentation.config import PresentationConfig
from phase3.presentation.providers.base import ChartProvider
from phase3.presentation.providers.registry import register_chart_provider
from phase3.presentation.schema import ChartSeries, ChartSpec, ChartType


@register_chart_provider(name="sankey", priority=40)
class SankeyChartProvider(ChartProvider):
    @property
    def name(self) -> str:
        return "sankey"

    @property
    def chart_type(self) -> ChartType:
        return ChartType.sankey

    @property
    def priority(self) -> int:
        return 40

    def build(self, data: dict[str, Any], config: PresentationConfig) -> ChartSpec:
        source: list[str] = data.get("source", [])
        target: list[str] = data.get("target", [])
        value: list[float | int | str] = data.get("value", [])
        labels: list[str] = data.get("labels", [])
        title: str = data.get("title", "Sankey Diagram")
        description: str = data.get("description", "")

        series = [
            ChartSeries(name="Source", values=source),
            ChartSeries(name="Target", values=target),
            ChartSeries(name="Flow", values=[float(v) for v in value]),
        ]

        return ChartSpec(
            chart_type=ChartType.sankey,
            title=title,
            description=description,
            series=series,
            labels=labels,
            metadata={"arrangement": data.get("arrangement", "snap")},
        )
