from __future__ import annotations

from phase4.copilot.config import CopilotConfig
from phase4.copilot.orchestrator.executor import ToolExecutor
from phase4.copilot.schema import ConversationState, ExecutionPlan, Intent, PlanNode


class TestToolExecutor:
    def setup_method(self):
        self.executor = ToolExecutor(CopilotConfig())
        self.state = ConversationState(session_id="test_sess")

    def test_execute_empty_plan(self):
        plan = ExecutionPlan(intent=Intent.GREETING)
        results = self.executor.execute_plan(plan, self.state)
        assert results == []

    def test_execute_single_node(self):
        plan = ExecutionPlan(intent=Intent.SEARCH)
        plan.nodes = [
            PlanNode(step_index=0, tool_name="search", parameters={"query": "test", "modules": ["kg"]}, depends_on=[]),
        ]
        results = self.executor.execute_plan(plan, self.state)
        assert len(results) == 1
        assert results[0].tool_name == "search"

    def test_execute_parallel_nodes(self):
        plan = ExecutionPlan(intent=Intent.BRIEFING)
        plan.nodes = [
            PlanNode(step_index=0, tool_name="search", parameters={"query": "test", "modules": ["kg"]}, depends_on=[]),
            PlanNode(step_index=1, tool_name="search", parameters={"query": "test", "modules": ["opportunity"]}, depends_on=[]),
        ]
        results = self.executor.execute_plan(plan, self.state)
        assert len(results) == 2

    def test_execute_dependent_nodes(self):
        plan = ExecutionPlan(intent=Intent.BRIEFING)
        plan.nodes = [
            PlanNode(step_index=0, tool_name="search", parameters={"query": "test", "modules": ["kg"]}, depends_on=[]),
            PlanNode(step_index=1, tool_name="search", parameters={"query": "test", "modules": ["opportunity"]}, depends_on=[0]),
        ]
        results = self.executor.execute_plan(plan, self.state)
        assert len(results) == 2

    def test_execute_unknown_tool(self):
        plan = ExecutionPlan(intent=Intent.UNKNOWN)
        plan.nodes = [
            PlanNode(step_index=0, tool_name="nonexistent_tool", parameters={}, depends_on=[]),
        ]
        results = self.executor.execute_plan(plan, self.state)
        assert len(results) == 1
        assert not results[0].success

    def test_execute_node_sync(self):
        node = PlanNode(step_index=0, tool_name="search", parameters={"query": "test", "modules": ["kg"]}, depends_on=[])
        result, step = self.executor.execute_node_sync(node, self.state)
        assert result.tool_name == "search"
        assert step.tool_name == "search"

    def test_node_has_dependency_info(self):
        plan = ExecutionPlan(intent=Intent.EVIDENCE)
        plan.nodes = [
            PlanNode(step_index=0, tool_name="search", parameters={"query": "test", "modules": ["kg"]}, depends_on=[]),
            PlanNode(step_index=1, tool_name="search", parameters={"query": "test", "modules": ["kg"]}, depends_on=[0, 999]),
        ]
        results = self.executor.execute_plan(plan, self.state)
        assert len(results) >= 1
