from __future__ import annotations

from typing import Any

from phase3.presentation.schema import ChartType, SectionDefinition, SectionType
from phase3.presentation.templates.base import BaseTemplate
from phase3.presentation.templates.registry import register_template


@register_template(name="business")
class BusinessTemplate(BaseTemplate):
    @property
    def name(self) -> str:
        return "business"

    @property
    def description(self) -> str:
        return "Business"

    @property
    def report_type(self) -> str:
        return "business"

    def sections(self) -> list[SectionDefinition]:
        return [
            SectionDefinition(section_type=SectionType.executive_summary, title="Executive Summary", order=0),
            SectionDefinition(section_type=SectionType.top_findings, title="Top Findings", order=1),
            SectionDefinition(section_type=SectionType.recommendations, title="Recommendations", order=7),
            SectionDefinition(section_type=SectionType.appendix, title="Appendix", order=10),
        ]

    def chart_types(self) -> list[ChartType]:
        return [ChartType.bar, ChartType.pie]

    def title(self, data: dict[str, Any]) -> str:
        return data.get("title", "Business Report")

    def subtitle(self, data: dict[str, Any]) -> str:
        return data.get("subtitle", "Business Intelligence Report")
