from __future__ import annotations

from typing import Any

from phase3.presentation.config import PresentationConfig
from phase3.presentation.renderers.base import Renderer
from phase3.presentation.renderers.registry import register_renderer
from phase3.presentation.schema import PresentationModel, ReportFormat, SectionType


@register_renderer(name="markdown")
class MarkdownRenderer(Renderer):
    @property
    def name(self) -> str:
        return "markdown"

    @property
    def format(self) -> ReportFormat:
        return ReportFormat.markdown

    def render(self, model: PresentationModel, config: PresentationConfig) -> str:
        lines: list[str] = []

        lines.append(f"# {model.title}")
        if model.subtitle:
            lines.append(f"*{model.subtitle}*")
        lines.append("")
        lines.append(f"**Report ID:** {model.report_id}")
        lines.append(f"**Generated:** {model.generated_at}")
        lines.append(f"**Type:** {model.report_type.value}")
        lines.append("")

        if model.summaries.one_paragraph:
            lines.append(model.summaries.one_paragraph)
            lines.append("")

        if model.summaries.five_bullets:
            for bullet in model.summaries.five_bullets:
                lines.append(f"- {bullet}")
            lines.append("")

        for section in model.sections:
            lines.append(f"## {section.title}")
            lines.append("")

            if section.summaries.one_paragraph:
                lines.append(section.summaries.one_paragraph)
                lines.append("")

            if section.summaries.five_bullets:
                for bullet in section.summaries.five_bullets:
                    lines.append(f"- {bullet}")
                lines.append("")

            self._render_content(lines, section.content, section.section_type)

            if section.charts:
                lines.append("### Charts")
                for chart in section.charts:
                    types_display = chart.chart_type.value.title()
                    lines.append(f"- *Chart: {chart.title} ({types_display})*")
                lines.append("")

            if section.provenance.confidence > 0:
                lines.append(f"*Confidence: {section.provenance.confidence:.0%}*")
                lines.append("")

        assets = model.assets
        if assets.metrics:
            lines.append("## Key Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for key, val in sorted(assets.metrics.items()):
                lines.append(f"| {key} | {val} |")
            lines.append("")

        if assets.highlights:
            lines.append("## Highlights")
            lines.append("")
            for h in assets.highlights:
                lines.append(f"- **{h.text}** (source: {h.source}, score: {h.score:.2f})")
            lines.append("")

        if assets.tables:
            for table in assets.tables:
                lines.append(f"### {table.title}")
                lines.append("")
                header = "| " + " | ".join(table.headers) + " |"
                sep = "| " + " | ".join("---" for _ in table.headers) + " |"
                lines.append(header)
                lines.append(sep)
                for row in table.rows:
                    str_row = [str(c) for c in row]
                    lines.append("| " + " | ".join(str_row) + " |")
                lines.append("")

        if model.versions:
            lines.append("---")
            lines.append("")
            lines.append("*Versions:*")
            lines.append(f"- Report: {model.versions.report_version}")
            lines.append(f"- Schema: {model.versions.schema_version}")
            lines.append(f"- Pipeline: {model.versions.pipeline_version}")
            lines.append(f"- Template: {model.versions.template_version}")
            lines.append("")

        if model.lineage.knowledge_graph_run_id:
            lines.append("*Source Lineage:*")
            if model.lineage.knowledge_graph_run_id:
                lines.append(f"- KG Run: {model.lineage.knowledge_graph_run_id}")
            if model.lineage.reasoning_run_id:
                lines.append(f"- Reasoning Run: {model.lineage.reasoning_run_id}")
            if model.lineage.opportunity_run_id:
                lines.append(f"- Opportunity Run: {model.lineage.opportunity_run_id}")
            if model.lineage.trend_run_id:
                lines.append(f"- Trend Run: {model.lineage.trend_run_id}")
            if model.lineage.snapshot_id:
                lines.append(f"- Snapshot: {model.lineage.snapshot_id}")
            lines.append("")

        return "\n".join(lines)

    def _render_content(
        self, lines: list[str], content: dict[str, Any], section_type: SectionType
    ) -> None:
        if not content:
            return

        summary_keys = {"summary", "text", "description"}
        for key in summary_keys:
            if key in content and isinstance(content[key], str) and content[key]:
                lines.append(content[key])
                lines.append("")
                break

        list_keys = {"items", "findings", "highlights", "recommendations", "root_causes"}
        for key in list_keys:
            if key in content and isinstance(content[key], list):
                for item in content[key]:
                    if isinstance(item, str):
                        lines.append(f"- {item}")
                    elif isinstance(item, dict):
                        label = item.get("title") or item.get("name") or item.get("text", "")
                        score = item.get("score")
                        if score is not None:
                            lines.append(f"- **{label}** (score: {score:.2f})")
                        else:
                            lines.append(f"- {label}")
                lines.append("")
                break

        table_keys = {"distribution", "counts", "table"}
        for key in table_keys:
            if key in content and isinstance(content[key], dict):
                d = content[key]
                lines.append("| Category | Count |")
                lines.append("|----------|-------|")
                for k, v in sorted(d.items()):
                    lines.append(f"| {k} | {v} |")
                lines.append("")
                break

        metrics = content.get("metrics", content.get("stats", {}))
        if isinstance(metrics, dict) and metrics:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in sorted(metrics.items()):
                lines.append(f"| {k} | {v} |")
            lines.append("")
