from __future__ import annotations

from typing import Any

from phase4.copilot.schema import Citation, ToolResult


class MarkdownFormatter:
    def format(self, content: str, tool_results: list[ToolResult], citations: list[Citation]) -> str:
        parts = [content]

        citations_md = self._format_citations(citations)
        if citations_md:
            parts.append(citations_md)

        trace_md = self._format_trace(tool_results)
        if trace_md:
            parts.append(trace_md)

        return "\n\n".join(parts)

    def _format_citations(self, citations: list[Citation]) -> str:
        if not citations:
            return ""
        lines = ["---", "### Sources"]
        for i, c in enumerate(citations, 1):
            conf = f" (confidence: {c.confidence:.2f})" if c.confidence else ""
            snippet = f": {c.snippet[:100]}" if c.snippet else ""
            lines.append(f"  [{i}] **{c.source_title}**{conf} [{c.source_module.value}]{snippet}")
        return "\n".join(lines)

    def _format_trace(self, results: list[ToolResult]) -> str:
        if not results:
            return ""
        lines = ["---", "### Tools Used"]
        for r in results:
            status = "✅" if r.success else "❌"
            elapsed = f"{r.elapsed_ms:.0f}ms"
            lines.append(f"  {status} **{r.tool_name}** ({elapsed})")
        return "\n".join(lines)
