from __future__ import annotations

from typing import Any

from phase3.presentation.config import PresentationConfig
from phase3.presentation.renderers.base import Renderer
from phase3.presentation.renderers.registry import register_renderer
from phase3.presentation.schema import PresentationModel, ReportFormat, SectionType

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #1a1a2e; background: #f8f9fa; padding: 40px; }
.container { max-width: 900px; margin: 0 auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); padding: 48px; }
h1 { font-size: 28px; color: #1a1a2e; margin-bottom: 4px; }
.subtitle { font-size: 16px; color: #6c757d; margin-bottom: 24px; }
.meta { font-size: 13px; color: #6c757d; margin-bottom: 32px; padding-bottom: 16px; border-bottom: 1px solid #e9ecef; }
.meta span { margin-right: 20px; }
.summary { font-size: 16px; color: #495057; padding: 16px 20px; background: #f0f4ff; border-left: 4px solid #4a6cf7; border-radius: 4px; margin-bottom: 32px; }
.section { margin-bottom: 40px; }
.section h2 { font-size: 22px; color: #1a1a2e; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 2px solid #e9ecef; }
.section p { margin-bottom: 12px; color: #495057; }
.bullets { padding-left: 20px; margin-bottom: 16px; }
.bullets li { margin-bottom: 8px; color: #495057; }
table { width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 14px; }
th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid #e9ecef; }
th { background: #f8f9fa; font-weight: 600; color: #1a1a2e; }
tr:hover td { background: #f8f9fa; }
.chart-placeholder { padding: 20px; background: #f8f9fa; border: 2px dashed #dee2e6; border-radius: 8px; text-align: center; color: #6c757d; margin-bottom: 16px; font-size: 14px; }
.confidence { font-size: 13px; color: #6c757d; margin-top: 8px; }
.highlights { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }
.highlight { padding: 12px 16px; background: #f0f4ff; border-radius: 8px; font-size: 14px; flex: 1; min-width: 200px; }
.highlight strong { color: #1a1a2e; }
.highlight .meta { font-size: 12px; color: #6c757d; margin-top: 4px; }
.versions { font-size: 12px; color: #adb5bd; margin-top: 40px; padding-top: 16px; border-top: 1px solid #e9ecef; }
.lineage { font-size: 12px; color: #adb5bd; margin-top: 8px; }
</style>
</head>
<body>
<div class="container">

<h1>{{ title }}</h1>
{% if subtitle %}<p class="subtitle">{{ subtitle }}</p>{% endif %}
<div class="meta">
<span>ID: {{ report_id }}</span>
<span>Generated: {{ generated_at }}</span>
<span>Type: {{ report_type }}</span>
</div>

{% if summary_paragraph %}
<div class="summary">{{ summary_paragraph }}</div>
{% endif %}

{% for section in sections %}
<div class="section section-{{ section.section_type }}">
<h2>{{ section.title }}</h2>

{% if section.summary_paragraph %}
<p>{{ section.summary_paragraph }}</p>
{% endif %}

{% if section.five_bullets %}
<ul class="bullets">
{% for bullet in section.five_bullets %}
<li>{{ bullet }}</li>
{% endfor %}
</ul>
{% endif %}

{% if section.content_summary %}
<p>{{ section.content_summary }}</p>
{% endif %}

{% if section.list_items %}
<ul class="bullets">
{% for item in section.list_items %}
<li>{{ item }}</li>
{% endfor %}
</ul>
{% endif %}

{% if section.table_data %}
<table>
<thead>
<tr>
{% for col in section.table_data.headers %}
<th>{{ col }}</th>
{% endfor %}
</tr>
</thead>
<tbody>
{% for row in section.table_data.rows %}
<tr>
{% for cell in row %}
<td>{{ cell }}</td>
{% endfor %}
</tr>
{% endfor %}
</tbody>
</table>
{% endif %}

{% if section.metrics_table %}
<table>
<thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>
{% for key, val in section.metrics_table.items() %}
<tr><td>{{ key }}</td><td>{{ val }}</td></tr>
{% endfor %}
</tbody>
</table>
{% endif %}

{% for chart in section.charts %}
<div class="chart-placeholder">Chart: {{ chart.title }} ({{ chart.chart_type }})</div>
{% endfor %}

{% if section.confidence %}
<p class="confidence">Confidence: {{ "%.0f"|format(section.confidence * 100) }}%</p>
{% endif %}
</div>
{% endfor %}

{% if highlights %}
<h2>Highlights</h2>
<div class="highlights">
{% for h in highlights %}
<div class="highlight">
<strong>{{ h.text }}</strong>
<div class="meta">Source: {{ h.source }} &middot; Score: {{ "%.2f"|format(h.score) }}</div>
</div>
{% endfor %}
</div>
{% endif %}

{% if metrics_table %}
<h2>Key Metrics</h2>
<table>
<thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>
{% for key, val in metrics_table.items() %}
<tr><td>{{ key }}</td><td>{{ val }}</td></tr>
{% endfor %}
</tbody>
</table>
{% endif %}

<div class="versions">
<p>Report v{{ versions.report_version }} | Schema v{{ versions.schema_version }} | Pipeline v{{ versions.pipeline_version }} | Template v{{ versions.template_version }}</p>
{% if lineage_kg or lineage_reasoning %}
<p class="lineage">
{% if lineage_kg %}KG: {{ lineage_kg }} | {% endif %}
{% if lineage_reasoning %}Reasoning: {{ lineage_reasoning }} | {% endif %}
{% if lineage_opportunity %}Opportunity: {{ lineage_opportunity }} | {% endif %}
{% if lineage_trend %}Trend: {{ lineage_trend }}{% endif %}
</p>
{% endif %}
</div>

</div>
</body>
</html>"""


def _extract_section_content(
    section_type: SectionType, content: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key in ("summary", "text", "description"):
        if key in content and isinstance(content[key], str) and content[key]:
            result["content_summary"] = content[key]
            break

    list_keys = {"items", "findings", "recommendations", "root_causes"}
    for key in list_keys:
        if key in content and isinstance(content[key], list) and content[key]:
            items = []
            for item in content[key]:
                if isinstance(item, str):
                    items.append(item)
                elif isinstance(item, dict):
                    label = item.get("title") or item.get("name") or item.get("text", "")
                    score = item.get("score")
                    if score is not None:
                        items.append(f"{label} (score: {score:.2f})")
                    else:
                        items.append(label)
            result["list_items"] = items
            break

    table_keys = {"distribution", "counts", "table"}
    for key in table_keys:
        if key in content and isinstance(content[key], dict):
            d = content[key]
            result["table_data"] = {
                "headers": ["Category", "Count"],
                "rows": [[k, str(v)] for k, v in sorted(d.items())],
            }
            break

    metrics = content.get("metrics", content.get("stats", {}))
    if isinstance(metrics, dict) and metrics:
        result["metrics_table"] = {str(k): str(v) for k, v in sorted(metrics.items())}

    return result


@register_renderer(name="html")
class HTMLRenderer(Renderer):
    @property
    def name(self) -> str:
        return "html"

    @property
    def format(self) -> ReportFormat:
        return ReportFormat.html

    def render(self, model: PresentationModel, config: PresentationConfig) -> str:
        try:
            from jinja2 import Template
        except ImportError:
            return self._render_fallback(model)

        template = Template(_HTML_TEMPLATE)

        sections_data = []
        for section in model.sections:
            sec = {"title": section.title, "section_type": section.section_type.value}
            sec["summary_paragraph"] = section.summaries.one_paragraph
            sec["five_bullets"] = section.summaries.five_bullets
            sec["confidence"] = section.provenance.confidence

            content_data = _extract_section_content(section.section_type, section.content)
            sec.update(content_data)

            sec["charts"] = [
                {"title": c.title, "chart_type": c.chart_type.value}
                for c in section.charts
            ]
            sections_data.append(sec)

        highlights = [
            {"text": h.text, "source": h.source, "score": h.score}
            for h in model.assets.highlights
        ]

        metrics_table = {str(k): str(v) for k, v in sorted(model.assets.metrics.items())}

        return template.render(
            title=model.title,
            subtitle=model.subtitle,
            report_id=model.report_id,
            generated_at=model.generated_at,
            report_type=model.report_type.value,
            summary_paragraph=model.summaries.one_paragraph,
            sections=sections_data,
            highlights=highlights,
            metrics_table=metrics_table,
            versions=model.versions,
            lineage_kg=model.lineage.knowledge_graph_run_id,
            lineage_reasoning=model.lineage.reasoning_run_id,
            lineage_opportunity=model.lineage.opportunity_run_id,
            lineage_trend=model.lineage.trend_run_id,
        )

    def _render_fallback(self, model: PresentationModel) -> str:
        parts = [f"<h1>{model.title}</h1>"]
        if model.subtitle:
            parts.append(f"<p><em>{model.subtitle}</em></p>")
        parts.append(f"<p>ID: {model.report_id} | Generated: {model.generated_at}</p>")
        for section in model.sections:
            parts.append(f"<h2>{section.title}</h2>")
            if section.summaries.one_paragraph:
                parts.append(f"<p>{section.summaries.one_paragraph}</p>")
        parts.append("</html>")
        return "\n".join(parts)
