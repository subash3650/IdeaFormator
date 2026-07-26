from __future__ import annotations

from typing import Any

from phase4.copilot.schema import Citation, ToolResult


class JSONFormatter:
    def format(self, content: str, tool_results: list[ToolResult], citations: list[Citation]) -> str:
        import json

        payload: dict[str, Any] = {
            "answer": content,
            "citations": [
                {
                    "id": c.citation_id,
                    "source": c.source_module.value,
                    "source_id": c.source_id,
                    "title": c.source_title,
                    "confidence": c.confidence,
                    "snippet": c.snippet[:200] if c.snippet else "",
                }
                for c in citations
            ],
            "data": [
                {
                    "tool": r.tool_name,
                    "success": r.success,
                    "data": r.data if isinstance(r.data, dict) else {},
                    "elapsed_ms": r.elapsed_ms,
                }
                for r in tool_results
            ],
        }
        return json.dumps(payload, indent=2, default=str)
