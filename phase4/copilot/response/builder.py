from __future__ import annotations

from typing import Any

from phase4.copilot.citations.citation_system import CitationBuilder
from phase4.copilot.config import CopilotConfig
from phase4.copilot.llm.mock import MockProvider
from phase4.copilot.schema import (
    Citation,
    ConversationState,
    CopilotResponse,
    ExecutionPlan,
    Intent,
    ReasoningTrace,
    ReasoningStep,
    ResponseFormat,
    ToolCall,
    ToolResult,
)
from phase4.copilot.response.markdown import MarkdownFormatter
from phase4.copilot.response.json_format import JSONFormatter
from phase4.copilot.response.streaming import StreamingHandler


class ResponseBuilder:
    def __init__(self, config: CopilotConfig) -> None:
        self._config = config
        self._citation_builder = CitationBuilder(
            min_confidence=config.min_citation_confidence,
            max_citations=config.max_citations_per_response,
        )
        self._markdown = MarkdownFormatter()
        self._json = JSONFormatter()
        self._llm = MockProvider()

    def build(
        self,
        query: str,
        tool_results: list[ToolResult],
        state: ConversationState,
        plan: ExecutionPlan,
        format: ResponseFormat = ResponseFormat.MARKDOWN,
        trace: ReasoningTrace | None = None,
    ) -> CopilotResponse:
        citations = self._citation_builder.from_results(tool_results)
        filtered_citations = self._citation_builder.filter_citations(citations)

        tool_calls = self._build_tool_calls(tool_results, plan)

        llm_context = {
            "tool_results": tool_results,
            "intent": plan.intent.value,
            "plan": plan.model_dump(mode="json"),
            "state": state.model_dump(mode="json"),
        }

        if plan.intent == Intent.CLARIFY:
            content = self._build_clarification(plan)
        elif plan.intent == Intent.GREETING:
            content = self._build_greeting()
        else:
            llm_response = self._llm.generate(query, context=llm_context)
            content = llm_response.content

        content = content[: self._config.max_response_length]

        if format == ResponseFormat.JSON:
            content = self._json.format(content, tool_results, filtered_citations)
        elif format == ResponseFormat.MARKDOWN:
            content = self._markdown.format(content, tool_results, filtered_citations)

        followups = []
        if self._config.enable_suggested_followups:
            followups = self._generate_followups(plan.intent, tool_results)

        response = CopilotResponse(
            session_id=state.session_id,
            content=content,
            format=format,
            citations=filtered_citations,
            tool_calls=tool_calls,
            confidence=plan.confidence,
            suggested_followups=followups,
            metadata={
                "intent": plan.intent.value,
                "tool_count": len(tool_calls),
                "citation_count": len(filtered_citations),
            },
        )

        if trace is not None:
            trace.response_id = response.response_id
            response.trace = trace

        return response

    def build_stream(
        self,
        query: str,
        tool_results: list[ToolResult],
        state: ConversationState,
        plan: ExecutionPlan,
    ):
        citations = self._citation_builder.from_results(tool_results)
        filtered_citations = self._citation_builder.filter_citations(citations)

        llm_context = {
            "tool_results": tool_results,
            "intent": plan.intent.value,
        }

        handler = StreamingHandler()

        if plan.intent == Intent.CLARIFY:
            for chunk in handler.stream_text(self._build_clarification(plan)):
                yield chunk
        elif plan.intent == Intent.GREETING:
            for chunk in handler.stream_text(self._build_greeting()):
                yield chunk
        else:
            for chunk in self._llm.generate_stream(query, context=llm_context):
                for c in handler.stream_text(chunk):
                    yield c

        for c in handler.stream_citations(filtered_citations):
            yield c

        final = handler.finish(
            session_id=state.session_id,
            citations=filtered_citations,
            confidence=plan.confidence,
        )
        yield final

    def _build_clarification(self, plan: ExecutionPlan) -> str:
        hints = plan.context_hints
        suggestions = hints.get("suggestions", [])
        text = "I'm not sure what you're asking. Here are some questions I can help with:\n\n"
        for s in suggestions:
            text += f"- {s}\n"
        return text

    def _build_greeting(self) -> str:
        return (
            "Hello! I'm the IdeaFormator AI Business Copilot.\n\n"
            "I can help you explore startup opportunities, trends, "
            "knowledge graph insights, and generate intelligence briefings.\n\n"
            "Try asking:\n"
            "- What startup opportunities exist in AI?\n"
            "- Which trends are growing fastest?\n"
            "- Compare two products\n"
            "- Generate an executive briefing"
        )

    def _generate_followups(self, intent: Intent, results: list[ToolResult]) -> list[str]:
        followups: dict[Intent, list[str]] = {
            Intent.QUERY_OPPORTUNITY: [
                "Why is the top opportunity ranked highly?",
                "What evidence supports this opportunity?",
                "Which companies are involved?",
            ],
            Intent.QUERY_TREND: [
                "Which trends are growing fastest?",
                "What technologies are emerging?",
                "Compare two trends",
            ],
            Intent.QUERY_KG: [
                "Show me the neighbors of this node",
                "Find paths between entities",
                "What companies are in the knowledge graph?",
            ],
            Intent.SEARCH: [
                "Tell me more about the first result",
                "What opportunities relate to this?",
            ],
            Intent.COMPARE: [
                "Show me the details of the first entity",
                "What evidence supports these scores?",
            ],
            Intent.EXPLAIN: [
                "What is the reasoning chain?",
                "Show me the root causes",
            ],
            Intent.EVIDENCE: [
                "What other conclusions are supported?",
                "Show me the full reasoning chain",
            ],
            Intent.BRIEFING: [
                "Generate a more detailed report",
                "What are the top opportunities?",
            ],
            Intent.STATISTICS: [
                "Show me the top opportunities",
                "Which trends are growing?",
            ],
        }
        return followups.get(intent, [])

    def _build_tool_calls(self, results: list[ToolResult], plan: ExecutionPlan) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for i, result in enumerate(results):
            node = plan.nodes[i] if i < len(plan.nodes) else None
            calls.append(ToolCall(
                tool_name=result.tool_name,
                parameters=node.parameters if node else {},
                result_summary=str(list(result.data.keys()))[:100] if isinstance(result.data, dict) else str(result.data)[:100],
                elapsed_ms=result.elapsed_ms,
                success=result.success,
                error=result.error,
            ))
        return calls
