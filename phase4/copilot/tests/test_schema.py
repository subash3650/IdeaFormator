from __future__ import annotations

from phase4.copilot.schema import (
    Citation,
    CitationSource,
    ConversationState,
    CopilotResponse,
    ExecutionPlan,
    Intent,
    LLMProviderResponse,
    MemoryEntry,
    MemoryLevel,
    Message,
    PermissionType,
    PlanNode,
    ReasoningStep,
    ReasoningTrace,
    ResponseFormat,
    Role,
    SessionInfo,
    SessionStatus,
    StreamingChunk,
    ToolCall,
    ToolMetadata,
    ToolPriority,
    ToolResult,
    _compute_id,
)


class TestEnums:
    def test_role_values(self):
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"

    def test_intent_values(self):
        assert Intent.QUERY_KG.value == "query_kg"
        assert Intent.COMPARE.value == "compare"

    def test_citation_source_values(self):
        assert CitationSource.KNOWLEDGE_GRAPH.value == "knowledge_graph"

    def test_response_format_values(self):
        assert ResponseFormat.MARKDOWN.value == "markdown"

    def test_session_status_values(self):
        assert SessionStatus.ACTIVE.value == "active"

    def test_memory_level_values(self):
        assert MemoryLevel.SHORT_TERM.value == "short_term"

    def test_permission_type_values(self):
        assert PermissionType.READ_ONLY.value == "read_only"

    def test_tool_priority_values(self):
        assert ToolPriority.HIGH.value == "high"


class TestModels:
    def test_tool_metadata(self):
        meta = ToolMetadata(
            name="test",
            description="test tool",
            supported_intents=[Intent.SEARCH],
        )
        assert meta.name == "test"
        assert meta.priority == ToolPriority.MEDIUM
        assert meta.permissions == PermissionType.READ_ONLY
        assert not meta.supports_streaming

    def test_tool_call_auto_id(self):
        tc = ToolCall(tool_name="test")
        assert tc.tool_call_id
        assert len(tc.tool_call_id) == 16

    def test_citation_auto_id(self):
        c = Citation(
            source_module=CitationSource.KNOWLEDGE_GRAPH,
            source_id="node_123",
            source_title="Test Node",
        )
        assert c.citation_id
        assert c.source_id == "node_123"

    def test_tool_result(self):
        tr = ToolResult(tool_name="test", data={"key": "value"})
        assert tr.tool_name == "test"
        assert tr.success
        assert tr.data["key"] == "value"

    def test_plan_node(self):
        node = PlanNode(step_index=0, tool_name="test", depends_on=[])
        assert node.node_id
        assert node.step_index == 0

    def test_execution_plan(self):
        plan = ExecutionPlan(intent=Intent.SEARCH)
        assert plan.plan_id
        assert plan.nodes == []

    def test_reasoning_step(self):
        step = ReasoningStep(step_type="tool_call", input="in", output="out")
        assert step.step_type == "tool_call"

    def test_reasoning_trace(self):
        trace = ReasoningTrace(query="test query", intent=Intent.SEARCH)
        assert trace.trace_id
        assert trace.query == "test query"

    def test_copilot_response(self):
        resp = CopilotResponse(
            session_id="sess_123",
            content="Hello",
            citations=[Citation(source_module=CitationSource.KNOWLEDGE_GRAPH, source_id="n1", source_title="N1")],
        )
        assert resp.response_id
        assert resp.content == "Hello"
        assert len(resp.citations) == 1

    def test_conversation_state(self):
        state = ConversationState(session_id="sess_123")
        assert state.session_id == "sess_123"
        assert state.turn_count == 0

    def test_session_info(self):
        info = SessionInfo(session_id="sess_123")
        assert info.created_at
        assert info.status == SessionStatus.ACTIVE

    def test_llm_provider_response(self):
        resp = LLMProviderResponse(content="Hello", provider="mock")
        assert resp.content == "Hello"

    def test_streaming_chunk(self):
        chunk = StreamingChunk(chunk_type="token", data="hello")
        assert chunk.chunk_type == "token"
        assert not chunk.final

    def test_memory_entry(self):
        entry = MemoryEntry(role=Role.USER, content="hello")
        assert entry.entry_id
        assert entry.level == MemoryLevel.SHORT_TERM

    def test_message(self):
        msg = Message(role=Role.USER, content="hello")
        assert msg.message_id
        assert msg.timestamp

    def test_compute_id(self):
        id1 = _compute_id("test", "a", "b")
        id2 = _compute_id("test", "a", "b")
        assert id1 == id2
        assert len(id1) == 16

    def test_deterministic_ids(self):
        id1 = _compute_id("msg", "sess_1", "user")
        id2 = _compute_id("msg", "sess_1", "user")
        assert id1 == id2
