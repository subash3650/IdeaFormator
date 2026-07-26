from __future__ import annotations

from typing import Any

from phase4.copilot.schema import Citation, CitationSource, ToolResult


class CitationBuilder:
    def __init__(self, min_confidence: float = 0.2, max_citations: int = 10) -> None:
        self._min_confidence = min_confidence
        self._max_citations = max_citations

    def from_tool_result(self, result: ToolResult) -> list[Citation]:
        return list(result.citations)

    def from_results(self, results: list[ToolResult]) -> list[Citation]:
        all_citations: list[Citation] = []
        seen: set[str] = set()
        for r in results:
            for c in r.citations:
                if c.citation_id not in seen and c.confidence >= self._min_confidence:
                    seen.add(c.citation_id)
                    all_citations.append(c)
        return all_citations[: self._max_citations]

    def filter_citations(self, citations: list[Citation]) -> list[Citation]:
        filtered = [c for c in citations if c.confidence >= self._min_confidence]
        seen: set[str] = set()
        unique: list[Citation] = []
        for c in filtered:
            key = f"{c.source_module}:{c.source_id}"
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique[: self._max_citations]

    def format_markdown(self, citations: list[Citation]) -> str:
        if not citations:
            return ""
        lines = ["\n\n**Sources:**"]
        for i, c in enumerate(citations, 1):
            snippet = c.snippet[:120] if c.snippet else ""
            conf = f" (confidence: {c.confidence:.2f})" if c.confidence else ""
            lines.append(f"  [{i}] {c.source_title}{conf}")
            if snippet:
                lines.append(f"      _{snippet}_")
        return "\n".join(lines)

    def format_json(self, citations: list[Citation]) -> list[dict[str, Any]]:
        return [
            {
                "id": c.citation_id,
                "source": c.source_module.value,
                "source_id": c.source_id,
                "title": c.source_title,
                "confidence": c.confidence,
                "snippet": c.snippet[:200] if c.snippet else "",
            }
            for c in citations
            if c.confidence >= self._min_confidence
        ][: self._max_citations]
