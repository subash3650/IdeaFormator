from __future__ import annotations

from phase4.copilot.config import CopilotConfig
from phase4.copilot.tools.kg_tool import KnowledgeGraphTool
from phase4.copilot.tools.reasoning_tool import ReasoningTool
from phase4.copilot.tools.opportunity_tool import OpportunityTool
from phase4.copilot.tools.trend_tool import TrendTool
from phase4.copilot.tools.presentation_tool import PresentationTool
from phase4.copilot.tools.search_tool import SearchTool
from phase4.copilot.tools.comparison_tool import ComparisonTool
from phase4.copilot.tools.evidence_tool import EvidenceTool
from phase4.copilot.tools.registry import available_tools, create_tool


class TestKnowledgeGraphTool:
    def setup_method(self):
        self.tool = KnowledgeGraphTool(CopilotConfig())

    def test_name(self):
        assert self.tool.name == "knowledge_graph"

    def test_metadata(self):
        meta = self.tool.metadata
        assert meta.name == "knowledge_graph"

    def test_execute_empty(self):
        result = self.tool.execute({"action": "search", "query": ""})
        assert result.tool_name == "knowledge_graph"
        assert result.data is not None

    def test_execute_stats(self):
        result = self.tool.execute({"action": "stats"})
        assert "node_count" in result.data or "error" in result.data


class TestReasoningTool:
    def setup_method(self):
        self.tool = ReasoningTool(CopilotConfig())

    def test_name(self):
        assert self.tool.name == "reasoning"

    def test_execute_stats(self):
        result = self.tool.execute({"action": "stats"})
        assert result.tool_name == "reasoning"


class TestOpportunityTool:
    def setup_method(self):
        self.tool = OpportunityTool(CopilotConfig())

    def test_name(self):
        assert self.tool.name == "opportunity"

    def test_execute_stats(self):
        result = self.tool.execute({"action": "stats"})
        assert result.tool_name == "opportunity"

    def test_execute_top(self):
        result = self.tool.execute({"action": "top"})
        assert "results" in result.data or "error" in result.data


class TestTrendTool:
    def setup_method(self):
        self.tool = TrendTool(CopilotConfig())

    def test_name(self):
        assert self.tool.name == "trend"

    def test_execute_stats(self):
        result = self.tool.execute({"action": "stats"})
        assert result.tool_name == "trend"


class TestPresentationTool:
    def setup_method(self):
        self.tool = PresentationTool(CopilotConfig())

    def test_name(self):
        assert self.tool.name == "presentation"

    def test_execute_list(self):
        result = self.tool.execute({"action": "list"})
        assert result.tool_name == "presentation"


class TestSearchTool:
    def setup_method(self):
        self.tool = SearchTool(CopilotConfig())

    def test_name(self):
        assert self.tool.name == "search"

    def test_execute(self):
        result = self.tool.execute({"query": "AI", "modules": ["kg"]})
        assert result.tool_name == "search"
        assert "knowledge_graph" in result.data or "total_matches" in result.data

    def test_execute_empty_query(self):
        result = self.tool.execute({"query": ""})
        assert result.tool_name == "search"


class TestComparisonTool:
    def setup_method(self):
        self.tool = ComparisonTool(CopilotConfig())

    def test_name(self):
        assert self.tool.name == "comparison"

    def test_execute_missing_entities(self):
        result = self.tool.execute({
            "entity_type": "opportunity",
            "entity_a": "nonexistent_a",
            "entity_b": "nonexistent_b",
        })
        assert "error" in result.data or result.tool_name == "comparison"

    def test_execute_company_comparison(self):
        result = self.tool.execute({
            "entity_type": "company",
            "a": "TestCorp",
            "b": "OtherCorp",
        })
        assert result.tool_name == "comparison"


class TestEvidenceTool:
    def setup_method(self):
        self.tool = EvidenceTool(CopilotConfig())

    def test_name(self):
        assert self.tool.name == "evidence"

    def test_execute_conclusion(self):
        result = self.tool.execute({"action": "for_conclusion", "target_id": "nonexistent"})
        assert result.tool_name == "evidence"

    def test_execute_opportunity(self):
        result = self.tool.execute({"action": "for_opportunity", "target_id": "nonexistent"})
        assert result.tool_name == "evidence"

    def test_execute_trend(self):
        result = self.tool.execute({"action": "for_trend", "target_id": "nonexistent"})
        assert result.tool_name == "evidence"


class TestToolRegistry:
    def test_available_tools(self):
        tools = available_tools()
        assert "knowledge_graph" in tools
        assert "reasoning" in tools
        assert "opportunity" in tools
        assert "trend" in tools
        assert "presentation" in tools
        assert "search" in tools
        assert "comparison" in tools
        assert "evidence" in tools

    def test_create_tool(self):
        tool = create_tool("knowledge_graph", config=CopilotConfig())
        assert tool.name == "knowledge_graph"

    def test_create_unknown_raises(self):
        try:
            create_tool("nonexistent")
            assert False
        except KeyError:
            pass
