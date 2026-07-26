from __future__ import annotations

from phase4.copilot.config import CopilotConfig
from phase4.copilot.retriever.retriever import UnifiedRetriever
from phase4.copilot.retriever.context_builder import ContextBuilder
from phase4.copilot.schema import ConversationState, Role, Message


class TestUnifiedRetriever:
    def setup_method(self):
        self.retriever = UnifiedRetriever(CopilotConfig())

    def test_retrieve_from_kg(self):
        result = self.retriever.retrieve_kg(query="test")
        assert isinstance(result, dict)
        assert "available" in result

    def test_retrieve_from_reasoning(self):
        result = self.retriever.retrieve_reasoning()
        assert isinstance(result, dict)
        assert "available" in result

    def test_retrieve_from_opportunity(self):
        result = self.retriever.retrieve_opportunities(query="test")
        assert isinstance(result, dict)
        assert "available" in result

    def test_retrieve_from_trend(self):
        result = self.retriever.retrieve_trends(query="test")
        assert isinstance(result, dict)
        assert "available" in result

    def test_retrieve_from_evidence(self):
        result = self.retriever.retrieve_evidence(conclusion_id="nonexistent")
        assert isinstance(result, dict)
        assert "available" in result

    def test_retrieve_unknown_source(self):
        assert not hasattr(self.retriever, "retrieve")

    def test_retrieve_top_k(self):
        result = self.retriever.retrieve_kg(query="AI")
        assert isinstance(result, dict)

    def test_multi_source_retrieve(self):
        kg = self.retriever.retrieve_kg(query="AI")
        trend = self.retriever.retrieve_trends(query="AI")
        assert isinstance(kg, dict)
        assert isinstance(trend, dict)


class TestContextBuilder:
    def setup_method(self):
        self.builder = ContextBuilder(CopilotConfig())

    def test_empty_context(self):
        from phase4.copilot.memory.conversation import ConversationMemory
        from phase4.copilot.planner.planner import QueryPlanner
        memory = ConversationMemory(CopilotConfig())
        state = ConversationState(session_id="s1")
        plan = QueryPlanner(CopilotConfig()).plan("test")
        ctx = self.builder.build("test", state, plan, memory)
        assert isinstance(ctx, dict)
        assert "query" in ctx

    def test_with_history(self):
        from phase4.copilot.memory.conversation import ConversationMemory
        from phase4.copilot.planner.planner import QueryPlanner
        memory = ConversationMemory(CopilotConfig())
        state = ConversationState(session_id="s1")
        plan = QueryPlanner(CopilotConfig()).plan("Hello")
        ctx = self.builder.build("Hello", state, plan, memory)
        assert "query" in ctx
        assert ctx["query"] == "Hello"

    def test_with_module_summary(self):
        from phase4.copilot.memory.conversation import ConversationMemory
        from phase4.copilot.planner.planner import QueryPlanner
        memory = ConversationMemory(CopilotConfig())
        state = ConversationState(session_id="s1")
        plan = QueryPlanner(CopilotConfig()).plan("search AI")
        ctx = self.builder.build("search AI", state, plan, memory)
        assert "module_summary" not in ctx

    def test_context_message_structure(self):
        from phase4.copilot.memory.conversation import ConversationMemory
        from phase4.copilot.planner.planner import QueryPlanner
        memory = ConversationMemory(CopilotConfig())
        state = ConversationState(session_id="s1")
        plan = QueryPlanner(CopilotConfig()).plan("test")
        ctx = self.builder.build("test", state, plan, memory)
        assert "query" in ctx
        assert ctx["query"] == "test"
