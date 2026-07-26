from __future__ import annotations

from typing import Any

from phase4.copilot.config import CopilotConfig
from phase4.copilot.planner.intent import IntentClassifier
from phase4.copilot.planner.extractor import ParameterExtractor
from phase4.copilot.schema import ConversationState, ExecutionPlan, Intent, PlanNode
from phase4.copilot.tools.registry import available_tools


_INTENT_TO_TOOLS: dict[Intent, list[tuple[str, str, list[int]]]] = {
    Intent.QUERY_KG: [("knowledge_graph", "search", [])],
    Intent.QUERY_REASONING: [
        ("reasoning", "stats", []),
    ],
    Intent.QUERY_OPPORTUNITY: [
        ("opportunity", "search", []),
    ],
    Intent.QUERY_TREND: [
        ("trend", "search", []),
    ],
    Intent.QUERY_PRESENTATION: [
        ("presentation", "list", []),
    ],
    Intent.COMPARE: [
        ("comparison", "compare", []),
    ],
    Intent.EXPLAIN: [
        ("reasoning", "explain", []),
    ],
    Intent.EVIDENCE: [
        ("evidence", "for_conclusion", []),
        ("knowledge_graph", "get_node", [0]),
    ],
    Intent.SEARCH: [
        ("search", "search", []),
    ],
    Intent.BRIEFING: [
        ("opportunity", "top", []),
        ("trend", "stats", []),
        ("presentation", "generate", [0, 1]),
    ],
    Intent.STATISTICS: [
        ("opportunity", "stats", []),
        ("trend", "stats", []),
        ("reasoning", "stats", [0, 1]),
        ("knowledge_graph", "search", []),
    ],
    Intent.GREETING: [],
    Intent.CLARIFY: [],
    Intent.UNKNOWN: [
        ("search", "search", []),
    ],
}


class QueryPlanner:
    def __init__(self, config: CopilotConfig) -> None:
        self._config = config
        self._classifier = IntentClassifier()
        self._extractor = ParameterExtractor()

    @property
    def classifier(self) -> IntentClassifier:
        return self._classifier

    def plan(self, query: str, state: ConversationState | None = None) -> ExecutionPlan:
        intent, confidence = self._classifier.classify(query, state)

        if confidence < self._config.planner_confidence_threshold:
            plan = ExecutionPlan(
                intent=Intent.CLARIFY,
                confidence=confidence,
                context_hints={"query": query, "suggestions": self._generate_suggestions(query)},
            )
            return plan

        params = self._extractor.extract(query, intent, state)

        tool_specs = _INTENT_TO_TOOLS.get(intent, [])
        if not tool_specs:
            if intent in (Intent.GREETING, Intent.CLARIFY):
                return ExecutionPlan(intent=intent, nodes=[], confidence=confidence)
            plan = ExecutionPlan(
                intent=Intent.CLARIFY,
                confidence=confidence,
                context_hints={"query": query},
            )
            return plan

        nodes: list[PlanNode] = []
        for i, (tool_name, action, deps) in enumerate(tool_specs):
            tool_params = dict(params)
            tool_params["action"] = action
            node = PlanNode(
                step_index=i,
                tool_name=tool_name,
                parameters=tool_params,
                depends_on=deps,
                description=f"{tool_name}:{action}",
            )
            nodes.append(node)

        plan = ExecutionPlan(
            intent=intent,
            nodes=nodes,
            context_hints=params,
            confidence=confidence,
        )
        return plan

    def plan_briefing(self, template: str | None = None) -> ExecutionPlan:
        nodes = [
            PlanNode(step_index=0, tool_name="opportunity", parameters={"action": "top", "top_k": 10}, depends_on=[], description="Get top opportunities"),
            PlanNode(step_index=1, tool_name="trend", parameters={"action": "stats"}, depends_on=[], description="Get trend statistics"),
            PlanNode(step_index=2, tool_name="presentation", parameters={"action": "generate", "report_type": "executive_summary", "template": template or "executive"}, depends_on=[0, 1], description="Generate briefing report"),
        ]
        return ExecutionPlan(
            intent=Intent.BRIEFING,
            nodes=nodes,
            confidence=1.0,
        )

    def _generate_suggestions(self, query: str) -> list[str]:
        return [
            "What startup opportunities exist in AI?",
            "Which trends are growing fastest?",
            "Compare two products",
            "Show me the top opportunities",
            "Generate an executive briefing",
            "Explain this opportunity ranking",
        ]
