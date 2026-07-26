from __future__ import annotations

from phase4.copilot.config import CopilotConfig
from phase4.copilot.engine import CopilotEngine
from phase4.copilot.schema import ResponseFormat


class TestCopilotEndToEnd:
    def setup_method(self):
        self.engine = CopilotEngine(CopilotConfig())

    def test_hello_to_goodbye(self):
        r1 = self.engine.chat("Hello", session_id="e2e_1")
        assert r1.content
        r2 = self.engine.chat("What is a top opportunity?", session_id="e2e_1")
        assert r2.content
        history = self.engine.get_session_history("e2e_1")
        assert len(history) >= 4

    def test_session_persistence(self):
        self.engine.chat("First message", session_id="e2e_persist")
        self.engine.chat("Second message", session_id="e2e_persist")
        history = self.engine.get_session_history("e2e_persist")
        assert len(history) == 4

    def test_stream_then_chat(self):
        chunks = list(self.engine.chat_stream("Hello", session_id="e2e_stream"))
        assert len(chunks) > 0
        r2 = self.engine.chat("What is trending?", session_id="e2e_stream")
        assert r2.content

    def test_stats_no_sessions(self):
        engine = CopilotEngine(CopilotConfig())
        stats = engine.stats()
        assert "total_sessions" in stats

    def test_stats_after_queries(self):
        self.engine.chat("Hello", session_id="e2e_stats1")
        self.engine.chat("Show KG stats", session_id="e2e_stats2")
        stats = self.engine.stats()
        assert stats["total_sessions"] >= 2

    def test_invalid_session(self):
        history = self.engine.get_session_history("nonexistent_session")
        assert history == []

    def test_chat_with_context(self):
        self.engine.chat("I am interested in AI", session_id="e2e_ctx")
        r2 = self.engine.chat("Tell me more", session_id="e2e_ctx")
        assert r2.content

    def test_json_output(self):
        response = self.engine.chat("Hello", session_id="e2e_json", response_format=ResponseFormat.JSON)
        assert response.format.value == "json"

    def test_briefing_flow(self):
        response = self.engine.chat("Generate report", session_id="e2e_brief")
        assert response.content
        assert response.tool_calls is not None

    def test_ask_without_explicit_session(self):
        sid = self.engine.new_session()
        response = self.engine.chat("Find trends", session_id=sid)
        assert response.content

    def test_consecutive_chats(self):
        sid = "e2e_consec"
        r1 = self.engine.chat("What are opportunities?", session_id=sid)
        assert r1.content
        r2 = self.engine.chat("Show me evidence", session_id=sid, response_format=ResponseFormat.JSON)
        assert r2.content

    def test_multi_session_isolation(self):
        s_a = self.engine.new_session()
        s_b = self.engine.new_session()
        self.engine.chat("For session A only", session_id=s_a)
        hist_a = self.engine.get_session_history(s_a)
        hist_b = self.engine.get_session_history(s_b)
        assert len(hist_a) == 2
        assert len(hist_b) == 0
