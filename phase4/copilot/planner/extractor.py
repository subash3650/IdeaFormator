from __future__ import annotations

import re
from typing import Any

from phase4.copilot.schema import ConversationState, Intent


class ParameterExtractor:
    def __init__(self) -> None:
        self._id_pattern = re.compile(r"[a-f0-9]{16,64}")
        self._number_pattern = re.compile(r"\b(\d+)\b")

    def extract(self, query: str, intent: Intent, state: ConversationState | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        q = query.lower()

        top_k = self._extract_number(q, default=10)
        params["top_k"] = top_k

        if intent == Intent.COMPARE:
            parts = re.split(r"\b(vs|versus|and)\b", query, flags=re.IGNORECASE)
            if len(parts) >= 3:
                params["entity_a"] = parts[0].strip()
                params["entity_b"] = parts[2].strip()
            else:
                words = q.split()
                if "compare" in words:
                    idx = words.index("compare")
                    remaining = words[idx + 1:]
                    if "and" in remaining:
                        and_idx = remaining.index("and")
                        params["entity_a"] = remaining[0]
                        params["entity_b"] = remaining[and_idx + 1] if len(remaining) > and_idx + 1 else remaining[-1]

            for et in ["opportunity", "trend", "company", "product"]:
                if et in q:
                    params["entity_type"] = et
                    break

        if intent == Intent.QUERY_OPPORTUNITY:
            companies = self._extract_after_keyword(q, ["company", "companies", "by"])
            if companies:
                params["company"] = companies
            for model in ["saas", "marketplace", "api", "b2b", "consumer", "mobile", "agent", "extension"]:
                if model in q:
                    params["opportunity_type"] = model
                    break

        if intent == Intent.QUERY_TREND:
            for ttype in ["growing", "declining", "emerging"]:
                if ttype in q:
                    params["action"] = ttype
                    break
            company = self._extract_after_keyword(q, ["company", "companies", "for", "by"])
            if company:
                params["value"] = company

        if intent == Intent.EXPLAIN:
            ids = self._id_pattern.findall(query)
            if ids:
                params["inference_id"] = ids[0]

        if intent == Intent.EVIDENCE:
            ids = self._id_pattern.findall(query)
            if ids:
                params["target_id"] = ids[0]
            if "opportunity" in q:
                params["action"] = "for_opportunity"
            elif "trend" in q:
                params["action"] = "for_trend"
            else:
                params["action"] = "for_conclusion"

        if intent == Intent.SEARCH:
            search_terms = self._extract_search_query(q)
            if search_terms:
                params["query"] = search_terms

        if intent == Intent.BRIEFING:
            if "investor" in q:
                params["template"] = "investor"
            elif "technology" in q or "tech" in q:
                params["template"] = "technology"
            elif "market" in q:
                params["template"] = "market"
            elif "founder" in q:
                params["template"] = "founder"

        if intent == Intent.QUERY_KG:
            search_terms = self._extract_search_query(q)
            if search_terms:
                params["query"] = search_terms
            for ntype in ["entity", "observation", "evidence", "company", "product", "technology", "problem"]:
                if ntype in q:
                    params["node_type"] = ntype
                    break

        if intent == Intent.QUERY_REASONING:
            if "root cause" in q or "root_cause" in q:
                params["action"] = "root_causes"
            elif "chain" in q:
                params["action"] = "chains"
            elif "inference" in q:
                params["action"] = "inferences"
            ids = self._id_pattern.findall(query)
            if ids:
                params["inference_id"] = ids[0]

        if state:
            if not params.get("query") and intent == Intent.QUERY_KG:
                params["query"] = q
            if not params.get("company") and state.mentioned_entities:
                pass

        return params

    def _extract_number(self, text: str, default: int = 10) -> int:
        matches = self._number_pattern.findall(text)
        for m in matches:
            val = int(m)
            if 1 <= val <= 100:
                return val
        return default

    def _extract_after_keyword(self, text: str, keywords: list[str]) -> str | None:
        for kw in keywords:
            parts = text.split(kw)
            if len(parts) > 1:
                after = parts[-1].strip().strip("?").strip()
                if after and len(after) < 50:
                    return after
        return None

    def _extract_search_query(self, text: str) -> str | None:
        for prefix in ["find", "search", "tell me about", "what is", "what are", "show me", "about"]:
            if text.startswith(prefix):
                after = text[len(prefix):].strip().strip("?").strip()
                if after and len(after) < 100:
                    return after
        return text.strip().strip("?")
