from __future__ import annotations

import time
from typing import Any, Generator

from phase4.copilot.llm.base import BaseLLMProvider
from phase4.copilot.schema import LLMProviderResponse


class MockProvider(BaseLLMProvider):
    def __init__(self, model: str = "mock") -> None:
        self._model = model

    @property
    def name(self) -> str:
        return "mock"

    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> LLMProviderResponse:
        start = time.perf_counter()
        content = self._build_response(prompt, context)
        elapsed = (time.perf_counter() - start) * 1000
        return LLMProviderResponse(
            content=content,
            provider="mock",
            model=self._model,
            tokens_in=len(prompt.split()),
            tokens_out=len(content.split()),
            elapsed_ms=round(elapsed, 1),
        )

    def generate_stream(self, prompt: str, context: dict[str, Any] | None = None) -> Generator[str, None, None]:
        content = self._build_response(prompt, context)
        words = content.split()
        for word in words:
            yield word + " "
            time.sleep(0.01)

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _build_response(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        ctx = context or {}
        tool_results = ctx.get("tool_results", [])
        intent = ctx.get("intent", "unknown")

        preamble = f"I analyzed your query using the IdeaFormator intelligence platform.\n\n"

        if not tool_results:
            return preamble + "No data was retrieved. Try rephrasing your question."

        sections = []
        for result in tool_results:
            data = getattr(result, "data", {}) if hasattr(result, "data") else {}
            name = getattr(result, "tool_name", "tool") if hasattr(result, "tool_name") else "tool"

            if name == "opportunity":
                results = data.get("results", data.get("top", []))
                if results:
                    sections.append("### Top Opportunities")
                    for item in results[:5]:
                        title = item.get("title", item.get("name", ""))
                        score = item.get("opportunity_score", item.get("score", 0))
                        rec = item.get("recommendation_type", "")
                        sections.append(f"- **{title}** (Score: {score:.2f}, Recommendation: {rec})")

            elif name == "trend":
                results = data.get("results", [])
                if results:
                    sections.append("### Trends")
                    for item in results[:5]:
                        title = item.get("title", "")
                        score = item.get("score", 0)
                        ttype = item.get("trend_type", "")
                        sections.append(f"- **{title}** ({ttype}, Score: {score:.2f})")
                else:
                    dist = data.get("type_distribution", {})
                    if dist:
                        sections.append("### Trend Distribution")
                        for ttype, count in dist.items():
                            sections.append(f"- {ttype}: {count}")

            elif name == "knowledge_graph":
                results = data.get("results", [])
                if results:
                    sections.append("### Knowledge Graph Results")
                    for item in results[:5]:
                        label = item.get("label", item.get("name", ""))
                        ntype = item.get("node_type", "")
                        sections.append(f"- **{label}** ({ntype})")

            elif name == "reasoning":
                causes = data.get("root_causes", [])
                if causes:
                    sections.append("### Root Causes")
                    for item in causes[:5]:
                        label = item.get("cause_label", "")
                        score = item.get("ranking_score", 0)
                        sections.append(f"- {label} (Score: {score:.2f})")

            elif name == "evidence":
                aggregations = data.get("evidence_aggregations", [])
                if aggregations:
                    sections.append("### Evidence")
                    for item in aggregations[:5]:
                        label = item.get("conclusion_label", "")
                        count = item.get("evidence_count", 0)
                        conf = item.get("aggregated_confidence", 0)
                        sections.append(f"- {label}: {count} items (Confidence: {conf:.2f})")

            elif name == "comparison":
                comparison = data.get("comparison", [])
                if comparison:
                    sections.append("### Comparison")
                    sections.append("| Field | Entity A | Entity B |")
                    sections.append("|-------|----------|----------|")
                    for item in comparison[:10]:
                        field = item.get("field", "")
                        a_val = item.get("a", "")
                        b_val = item.get("b", "")
                        sections.append(f"| {field} | {a_val} | {b_val} |")

            elif name == "search":
                for module in ["knowledge_graph", "opportunities", "trends", "reasoning"]:
                    mod_data = data.get(module, {})
                    matches = mod_data.get("results", [])
                    if matches:
                        sections.append(f"### {module.title()}")
                        for m in matches[:3]:
                            sections.append(f"- {m}")

            elif name == "presentation":
                reports = data.get("reports", [])
                if reports:
                    sections.append("### Available Reports")
                    for r in reports[:5]:
                        sections.append(f"- **{r.get('title', '')}** ({r.get('type', '')})")

        if sections:
            return preamble + "\n".join(sections)

        return preamble + "Analysis complete. See structured data above for details."
