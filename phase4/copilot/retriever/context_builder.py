from __future__ import annotations

from typing import Any

from phase4.copilot.config import CopilotConfig
from phase4.copilot.memory.conversation import ConversationMemory
from phase4.copilot.schema import ConversationState, ExecutionPlan
from phase4.copilot.retriever.retriever import UnifiedRetriever


class ContextBuilder:
    def __init__(self, config: CopilotConfig) -> None:
        self._config = config
        self._retriever = UnifiedRetriever(config)

    def build(
        self,
        query: str,
        state: ConversationState,
        plan: ExecutionPlan,
        memory: ConversationMemory,
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            "query": query,
            "intent": plan.intent.value,
            "confidence": plan.confidence,
            "plan": plan.model_dump(mode="json"),
            "turn_count": state.turn_count,
            "last_intent": state.last_intent.value if state.last_intent else None,
        }

        recent = memory.get_recent(5)
        if recent:
            context["recent_history"] = [
                {"role": e.role.value, "content": e.content[:200]} for e in recent
            ]

        full_ctx = memory.get_full_context()
        context["conversation_context"] = [
            {"role": e.role.value, "content": e.content[:200]} for e in full_ctx[-10:]
        ]

        if plan.intent.value in ("query_kg", "search", "evidence", "explain"):
            context["kg_summary"] = self._retriever.retrieve_kg()

        if plan.intent.value in ("query_opportunity", "search", "briefing"):
            context["opportunity_summary"] = self._retriever.retrieve_opportunities()

        if plan.intent.value in ("query_trend", "search", "briefing"):
            context["trend_summary"] = self._retriever.retrieve_trends()

        if plan.intent.value in ("query_reasoning", "explain", "evidence"):
            context["reasoning_summary"] = self._retriever.retrieve_reasoning()

        if state.mentioned_entities:
            context["mentioned_entities"] = list(state.mentioned_entities)[:10]

        if state.topic_stack:
            context["topic_stack"] = list(state.topic_stack)

        return context
