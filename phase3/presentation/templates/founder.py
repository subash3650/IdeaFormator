from __future__ import annotations

from typing import Any

from phase3.presentation.schema import ChartType, SectionDefinition, SectionType
from phase3.presentation.templates.business import BusinessTemplate
from phase3.presentation.templates.registry import register_template


@register_template(name="founder")
class FounderTemplate(BusinessTemplate):
    @property
    def name(self) -> str:
        return "founder"

    @property
    def description(self) -> str:
        return "Startup Opportunity"

    @property
    def report_type(self) -> str:
        return "startup_opportunity"

    def sections(self) -> list[SectionDefinition]:
        return [
            SectionDefinition(section_type=SectionType.executive_summary, title="Executive Summary", order=0),
            SectionDefinition(section_type=SectionType.top_findings, title="Top Findings", order=1),
            SectionDefinition(section_type=SectionType.opportunity_analysis, title="Opportunity Analysis", order=2),
            SectionDefinition(section_type=SectionType.root_causes, title="Root Causes", order=3),
            SectionDefinition(section_type=SectionType.evidence, title="Evidence", order=4),
            SectionDefinition(section_type=SectionType.reasoning_summary, title="Reasoning Summary", order=5),
            SectionDefinition(section_type=SectionType.recommendations, title="Recommendations", order=7),
            SectionDefinition(section_type=SectionType.appendix, title="Appendix", order=10),
        ]

    def chart_types(self) -> list[ChartType]:
        return [ChartType.bar, ChartType.line, ChartType.pie]

    def title(self, data: dict[str, Any]) -> str:
        return data.get("title", "Startup Opportunity Report")

    def subtitle(self, data: dict[str, Any]) -> str:
        return data.get("subtitle", "Validated problem-solution fits and market opportunities")
