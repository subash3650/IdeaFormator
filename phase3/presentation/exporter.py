from __future__ import annotations

from pathlib import Path
from typing import Any

from phase3.presentation.config import PresentationConfig
from phase3.presentation.renderers import available_renderers, create_renderer
from phase3.presentation.schema import PresentationModel, ReportFormat
from phase3.presentation.store import PresentationStore


class PresentationExporter:
    def __init__(self, config: PresentationConfig) -> None:
        self._config = config
        self._store = PresentationStore(config.output_dir)

    def export(
        self,
        report_id: str,
        output_dir: str | Path | None = None,
        formats: list[ReportFormat] | None = None,
    ) -> dict[str, str]:
        model = self._store.load_content(report_id)
        if model is None:
            return {"error": f"Report not found: {report_id}"}

        out_base = Path(output_dir) if output_dir else self._config.report_dir / report_id
        out_base.mkdir(parents=True, exist_ok=True)

        if formats is None:
            formats = self._config.enabled_formats

        exported: dict[str, str] = {}

        for fmt in formats:
            renderer_name = fmt.value
            if renderer_name not in available_renderers():
                exported[fmt.value] = ""
                continue

            try:
                renderer = create_renderer(renderer_name)
                content = renderer.render(model, self._config)

                ext = _format_extension(fmt)
                out_path = out_base / f"{report_id}.{ext}"
                out_path.write_text(content, encoding="utf-8")
                exported[fmt.value] = str(out_path)
            except Exception:
                exported[fmt.value] = ""

        return exported

    def export_all(
        self,
        output_dir: str | Path | None = None,
        formats: list[ReportFormat] | None = None,
    ) -> dict[str, dict[str, str]]:
        results: dict[str, dict[str, str]] = {}
        for output in self._store.load_all():
            rid = output.report_id
            result = self.export(rid, output_dir=output_dir, formats=formats)
            results[rid] = result
        return results

    def export_to_format(
        self,
        report_id: str,
        fmt: ReportFormat,
        output_path: str | Path | None = None,
    ) -> str:
        model = self._store.load_content(report_id)
        if model is None:
            return ""

        renderer_name = fmt.value
        if renderer_name not in available_renderers():
            return ""

        try:
            renderer = create_renderer(renderer_name)
            content = renderer.render(model, self._config)

            if output_path:
                Path(output_path).write_text(content, encoding="utf-8")
            return content
        except Exception:
            return ""

    def export_summary(self, report_id: str, output_path: str | Path | None = None) -> str:
        model = self._store.load_content(report_id)
        if model is None:
            return ""

        lines = [
            f"# {model.title}",
            f"Type: {model.report_type.value}",
            f"Generated: {model.generated_at}",
            "",
            "## Sections",
        ]

        for s in model.sections:
            lines.append(f"- {s.title} ({s.section_type.value})")

        if model.summaries.one_paragraph:
            lines.extend(["", "## Summary", model.summaries.one_paragraph])

        if model.summaries.five_bullets:
            lines.extend(["", "## Key Points"])
            lines.extend(f"- {b}" for b in model.summaries.five_bullets)

        text = "\n".join(lines)
        if output_path:
            Path(output_path).write_text(text, encoding="utf-8")
        return text


def _format_extension(fmt: ReportFormat) -> str:
    ext_map: dict[ReportFormat, str] = {
        ReportFormat.json: "json",
        ReportFormat.markdown: "md",
        ReportFormat.html: "html",
        ReportFormat.csv: "csv",
        ReportFormat.pdf: "pdf",
        ReportFormat.docx: "docx",
        ReportFormat.pptx: "pptx",
    }
    return ext_map.get(fmt, fmt.value)
