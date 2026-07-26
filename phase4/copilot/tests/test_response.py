from __future__ import annotations

from phase4.copilot.config import CopilotConfig
from phase4.copilot.response.builder import ResponseBuilder
from phase4.copilot.response.json_format import JSONFormatter
from phase4.copilot.response.markdown import MarkdownFormatter
from phase4.copilot.response.streaming import StreamingHandler
from phase4.copilot.schema import (
    Citation,
    CitationSource,
    ConversationState,
    ExecutionPlan,
    Intent,
    ResponseFormat,
    ToolResult,
)


class TestMarkdownFormatter:
    def setup_method(self):
        self.formatter = MarkdownFormatter()

    def test_format_content(self):
        result = self.formatter.format("Hello world", [], [])
        assert "Hello world" in result

    def test_format_with_citations(self):
        c = Citation(
            source_module=CitationSource.KNOWLEDGE_GRAPH,
            source_id="n1",
            source_title="Test Node",
            confidence=0.8,
        )
        result = self.formatter.format("Answer", [], [c])
        assert "Sources" in result
        assert "Test Node" in result

    def test_format_with_tool_trace(self):
        tr = ToolResult(tool_name="test_tool", data={})
        result = self.formatter.format("Answer", [tr], [])
        assert "Tools Used" in result
        assert "test_tool" in result


class TestJSONFormatter:
    def setup_method(self):
        self.formatter = JSONFormatter()

    def test_format_contains_answer(self):
        result = self.formatter.format("Hello", [], [])
        assert "answer" in result
        assert "Hello" in result

    def test_format_contains_citations(self):
        c = Citation(
            source_module=CitationSource.KNOWLEDGE_GRAPH,
            source_id="n1",
            source_title="Node 1",
        )
        result = self.formatter.format("Answer", [], [c])
        assert "citations" in result
        assert "Node 1" in result

    def test_format_contains_data(self):
        tr = ToolResult(tool_name="tool1", data={"key": "val"})
        result = self.formatter.format("Answer", [tr], [])
        assert "data" in result
        assert "tool1" in result

    def test_valid_json(self):
        import json
        result = self.formatter.format("Test", [], [])
        parsed = json.loads(result)
        assert parsed["answer"] == "Test"


class TestStreamingHandler:
    def setup_method(self):
        self.handler = StreamingHandler()

    def test_stream_text_generates_chunks(self):
        chunks = list(self.handler.stream_text("hello world"))
        assert len(chunks) >= 1
        assert all(c.chunk_type == "token" for c in chunks)
        assert not any(c.final for c in chunks)

    def test_stream_citations(self):
        c = Citation(
            source_module=CitationSource.KNOWLEDGE_GRAPH,
            source_id="n1",
            source_title="Node 1",
        )
        chunks = list(self.handler.stream_citations([c]))
        assert len(chunks) >= 1
        assert chunks[0].chunk_type == "citations"

    def test_stream_citations_empty(self):
        chunks = list(self.handler.stream_citations([]))
        assert len(chunks) == 0

    def test_finish_returns_final_chunk(self):
        chunk = self.handler.finish(session_id="s1")
        assert chunk.final
        assert chunk.chunk_type == "done"


class TestResponseBuilder:
    def setup_method(self):
        self.builder = ResponseBuilder(CopilotConfig())

    def test_build_greeting(self):
        state = ConversationState(session_id="s1")
        plan = ExecutionPlan(intent=Intent.GREETING)
        response = self.builder.build("Hello", [], state, plan)
        assert response.content
        assert response.confidence == 1.0

    def test_build_clarify(self):
        state = ConversationState(session_id="s1")
        plan = ExecutionPlan(intent=Intent.CLARIFY, context_hints={"suggestions": ["Ask something"]})
        response = self.builder.build("???", [], state, plan)
        assert "Ask something" in response.content

    def test_build_json_format(self):
        state = ConversationState(session_id="s1")
        plan = ExecutionPlan(intent=Intent.GREETING)
        response = self.builder.build("Hello", [], state, plan, format=ResponseFormat.JSON)
        assert response.format == ResponseFormat.JSON

    def test_build_with_tool_results(self):
        state = ConversationState(session_id="s1")
        plan = ExecutionPlan(intent=Intent.QUERY_KG)
        plan.nodes = []
        tr = ToolResult(tool_name="knowledge_graph", data={"node_count": 10})
        response = self.builder.build("KG query", [tr], state, plan)
        assert response.tool_calls is not None

    def test_build_confidence(self):
        state = ConversationState(session_id="s1")
        plan = ExecutionPlan(intent=Intent.GREETING, confidence=0.95)
        response = self.builder.build("Hi", [], state, plan)
        assert response.confidence == 0.95

    def test_build_followups(self):
        state = ConversationState(session_id="s1")
        plan = ExecutionPlan(intent=Intent.QUERY_OPPORTUNITY)
        response = self.builder.build("Show opportunities", [], state, plan)
        assert len(response.suggested_followups) > 0

    def test_build_clarification_suggestions(self):
        state = ConversationState(session_id="s1")
        plan = ExecutionPlan(intent=Intent.CLARIFY)
        text = self.builder._build_clarification(plan)
        assert len(text) > 0

    def test_build_greeting_text(self):
        text = self.builder._build_greeting()
        assert "IdeaFormator" in text
