from __future__ import annotations

from typing import Any

from phase4.copilot.schema import ConversationState, Intent


class IntentClassifier:
    def __init__(self) -> None:
        self._patterns: list[tuple[list[str], Intent, float]] = [
            (["greet", "hello", "hi ", "hey", "good morning", "good evening"], Intent.GREETING, 0.95),
            (["compare", "vs ", "versus", "difference", "versus ", " vs"], Intent.COMPARE, 0.9),
            (["compare opportunity", "compare trend", "compare company", "compare product"], Intent.COMPARE, 0.95),
            (["explain", "why is", "why does", "reasoning", "how was"], Intent.EXPLAIN, 0.85),
            (["briefing", "executive summary", "generate report", "create report"], Intent.BRIEFING, 0.9),
            (["evidence", "support", "what evidence", "proof"], Intent.EVIDENCE, 0.9),
            (["trend", "growing", "declining", "emerging", "what.*trend"], Intent.QUERY_TREND, 0.85),
            (["fastest growing", "which trend"], Intent.QUERY_TREND, 0.9),
            (["opportunity", "startup", "business idea", "what.*opportunit"], Intent.QUERY_OPPORTUNITY, 0.85),
            (["top opportunity", "best opportunity", "highest ranked"], Intent.QUERY_OPPORTUNITY, 0.9),
            (["knowledge graph", "kg ", "graph node", "entity"], Intent.QUERY_KG, 0.85),
            (["report", "presentation", "show report", "list report"], Intent.QUERY_PRESENTATION, 0.85),
            (["stats", "statistics", "count", "how many", "summary"], Intent.STATISTICS, 0.7),
            (["search", "find", "lookup", "tell me about"], Intent.SEARCH, 0.7),
            (["company", "what compan"], Intent.SEARCH, 0.6),
            (["technology", "what tech"], Intent.SEARCH, 0.6),
        ]

    def classify(self, query: str, state: ConversationState | None = None) -> tuple[Intent, float]:
        q = query.lower().strip()

        for keywords, intent, confidence in self._patterns:
            for kw in keywords:
                if kw in q:
                    return intent, confidence

        if state and state.last_intent in (
            Intent.QUERY_OPPORTUNITY, Intent.QUERY_TREND,
            Intent.QUERY_KG, Intent.QUERY_REASONING,
        ):
            followup_keywords = ["more", "elaborate", "tell me more", "details", "specifically"]
            for kw in followup_keywords:
                if kw in q:
                    return Intent.EXPLAIN, 0.7

        return Intent.UNKNOWN, 0.2

    def add_pattern(self, keywords: list[str], intent: Intent, confidence: float) -> None:
        self._patterns.insert(0, (keywords, intent, confidence))
