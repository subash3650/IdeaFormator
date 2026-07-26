from __future__ import annotations

import csv
import io
from typing import Any

from phase3.presentation.config import PresentationConfig
from phase3.presentation.renderers.base import Renderer
from phase3.presentation.renderers.registry import register_renderer
from phase3.presentation.schema import PresentationModel, ReportFormat


@register_renderer(name="csv")
class CSVRenderer(Renderer):
    @property
    def name(self) -> str:
        return "csv"

    @property
    def format(self) -> ReportFormat:
        return ReportFormat.csv

    def render(self, model: PresentationModel, config: PresentationConfig) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)

        writer.writerow([
            "Report ID", "Report Type", "Generated",
            "Section", "Section Title", "Item Type",
            "Title", "Score", "Source",
        ])

        seen: set[tuple[str, str, str]] = set()

        for section in model.sections:
            items = self._extract_items(section.content)
            for item_type, title, score, source in items:
                key = (section.section_id, str(item_type), str(title))
                if key not in seen:
                    seen.add(key)
                    writer.writerow([
                        model.report_id,
                        model.report_type.value,
                        model.generated_at,
                        section.section_type.value,
                        section.title,
                        item_type,
                        title,
                        f"{score:.4f}" if isinstance(score, (int, float)) else "",
                        source,
                    ])

        return buf.getvalue()

    def _extract_items(
        self, content: dict[str, Any]
    ) -> list[tuple[str, str, float | str, str]]:
        items: list[tuple[str, str, float | str, str]] = []

        list_fields = {
            "findings": "finding",
            "recommendations": "recommendation",
            "root_causes": "root_cause",
            "items": "item",
            "opportunities": "opportunity",
            "trends": "trend",
            "growing": "trend",
            "declining": "trend",
            "emerging": "trend",
        }

        for key, item_type in list_fields.items():
            raw = content.get(key, [])
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, str):
                        items.append((item_type, item, 0.0, key))
                    elif isinstance(item, dict):
                        title = (
                            item.get("title")
                            or item.get("name")
                            or item.get("text", "")
                        )
                        score = item.get("score", item.get("value", 0.0))
                        source = item.get("source", key)
                        items.append((item_type, title, score, source))

        return items
