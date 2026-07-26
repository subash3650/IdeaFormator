from __future__ import annotations

from phase4.copilot.config import CopilotConfig
from phase4.copilot.planner.intent import IntentClassifier
from phase4.copilot.planner.extractor import ParameterExtractor
from phase4.copilot.planner.planner import QueryPlanner
from phase4.copilot.schema import ConversationState, Intent


class TestIntentClassifier:
    def setup_method(self):
        self.classifier = IntentClassifier()

    def test_greeting(self):
        intent, conf = self.classifier.classify("Hello")
        assert intent == Intent.GREETING
        assert conf > 0.5

    def test_compare(self):
        intent, conf = self.classifier.classify("Compare these two opportunities")
        assert intent == Intent.COMPARE

    def test_explain(self):
        intent, conf = self.classifier.classify("Why is this opportunity ranked highly?")
        assert intent == Intent.EXPLAIN

    def test_evidence(self):
        intent, conf = self.classifier.classify("What evidence supports this?")
        assert intent == Intent.EVIDENCE

    def test_trend_query(self):
        intent, conf = self.classifier.classify("Which trends are growing fastest?")
        assert intent == Intent.QUERY_TREND

    def test_opportunity_query(self):
        intent, conf = self.classifier.classify("What startup opportunities exist?")
        assert intent == Intent.QUERY_OPPORTUNITY

    def test_kg_query(self):
        intent, conf = self.classifier.classify("What entities are in the knowledge graph?")
        assert intent == Intent.QUERY_KG

    def test_briefing(self):
        intent, conf = self.classifier.classify("Generate an executive briefing")
        assert intent == Intent.BRIEFING

    def test_statistics(self):
        intent, conf = self.classifier.classify("Show me statistics for all modules")
        assert intent == Intent.STATISTICS

    def test_search(self):
        intent, conf = self.classifier.classify("Search for machine learning companies")
        assert intent == Intent.SEARCH

    def test_unknown(self):
        intent, conf = self.classifier.classify("asdfghjkl")
        assert intent == Intent.UNKNOWN

    def test_presentation_query(self):
        intent, conf = self.classifier.classify("What reports are available?")
        assert intent == Intent.QUERY_PRESENTATION

    def test_followup_with_state(self):
        state = ConversationState(session_id="s1")
        state.last_intent = Intent.QUERY_OPPORTUNITY
        intent, conf = self.classifier.classify("Tell me more", state)
        assert intent == Intent.EXPLAIN

    def test_add_pattern(self):
        self.classifier.add_pattern(["custom query"], Intent.STATISTICS, 0.9)
        intent, conf = self.classifier.classify("custom query")
        assert intent == Intent.STATISTICS
        assert conf == 0.9


class TestParameterExtractor:
    def setup_method(self):
        self.extractor = ParameterExtractor()

    def test_extract_number(self):
        params = self.extractor.extract("Show 5 items", Intent.QUERY_OPPORTUNITY)
        assert params.get("top_k") == 5

    def test_extract_default_number(self):
        params = self.extractor.extract("Show opportunities", Intent.QUERY_OPPORTUNITY)
        assert params.get("top_k") == 10

    def test_extract_compare(self):
        params = self.extractor.extract("Compare opportunity A vs opportunity B", Intent.COMPARE)
        assert params.get("entity_a") is not None
        assert params.get("entity_b") is not None

    def test_extract_trend_type(self):
        params = self.extractor.extract("What trends are growing?", Intent.QUERY_TREND)
        assert params.get("action") == "growing"

    def test_extract_search_query(self):
        params = self.extractor.extract("Find companies working on AI", Intent.SEARCH)
        assert params.get("query") is not None

    def test_extract_node_type(self):
        params = self.extractor.extract("Find company nodes in KG", Intent.QUERY_KG)
        assert params.get("node_type") == "company"

    def test_extract_briefing_template(self):
        params = self.extractor.extract("Generate an investor briefing", Intent.BRIEFING)
        assert params.get("template") == "investor"

    def test_extract_opportunity_type(self):
        params = self.extractor.extract("Find SaaS opportunities", Intent.QUERY_OPPORTUNITY)
        assert params.get("opportunity_type") == "saas"


class TestQueryPlanner:
    def setup_method(self):
        self.planner = QueryPlanner(CopilotConfig())

    def test_plan_greeting(self):
        plan = self.planner.plan("hello")
        assert plan.intent == Intent.GREETING
        assert len(plan.nodes) == 0

    def test_plan_clarify(self):
        plan = self.planner.plan("asdfghjkl")
        assert plan.intent == Intent.CLARIFY

    def test_plan_search(self):
        plan = self.planner.plan("Find AI companies")
        assert plan.intent == Intent.SEARCH
        assert len(plan.nodes) >= 1

    def test_plan_opportunity(self):
        plan = self.planner.plan("What is a top opportunity?")
        assert plan.intent == Intent.QUERY_OPPORTUNITY
        assert plan.nodes[0].tool_name == "opportunity"

    def test_plan_trend(self):
        plan = self.planner.plan("Which trends are growing?")
        assert plan.intent == Intent.QUERY_TREND
        assert plan.nodes[0].tool_name == "trend"

    def test_plan_briefing(self):
        plan = self.planner.plan("Generate report")
        assert plan.intent == Intent.BRIEFING
        assert len(plan.nodes) >= 2

    def test_plan_evidence(self):
        plan = self.planner.plan("What evidence supports this conclusion?")
        assert plan.intent == Intent.EVIDENCE
        assert len(plan.nodes) >= 1

    def test_plan_compare(self):
        plan = self.planner.plan("Compare two products")
        assert plan.intent == Intent.COMPARE

    def test_plan_briefing_direct(self):
        plan = self.planner.plan_briefing()
        assert plan.intent == Intent.BRIEFING
        assert len(plan.nodes) == 3

    def test_plan_maintains_confidence(self):
        plan = self.planner.plan("What are the top opportunities?")
        assert plan.confidence > 0
