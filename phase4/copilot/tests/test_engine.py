from __future__ import annotations

from phase4.copilot.config import CopilotConfig
from phase4.copilot.engine import CopilotEngine
from phase4.copilot.schema import CopilotResponse, ResponseFormat


class TestCopilotEngine:
    def setup_method(self):
        self.engine = CopilotEngine(CopilotConfig())

    def test_chat_greeting(self):
        response = self.engine.chat("Hello", session_id="test_greeting")
        assert isinstance(response, CopilotResponse)
        assert response.content
        assert response.response_id

    def test_chat_query(self):
        response = self.engine.chat("What is a top opportunity?", session_id="test_query")
        assert response.content
        assert response.response_id

    def test_chat_empty(self):
        response = self.engine.chat("", session_id="test_empty")
        assert response.content or not response.response_id

    def test_chat_stream(self):
        chunks = list(self.engine.chat_stream("Hello", session_id="test_stream"))
        assert len(chunks) >= 1
        assert any(c.final for c in chunks)

    def test_ask(self):
        sid = self.engine.new_session()
        response = self.engine.chat("Find AI companies", session_id=sid)
        assert response.content
        assert response.tool_calls is not None

    def test_ask_json(self):
        response = self.engine.chat("What is trending?", session_id="ask_json", response_format=ResponseFormat.JSON)
        assert response.format.value == "json"

    def test_new_session(self):
        sid = self.engine.new_session()
        assert sid is not None
        assert len(sid) > 0

    def test_get_session_history(self):
        self.engine.chat("Test message", session_id="test_hist")
        history = self.engine.get_session_history("test_hist")
        assert len(history) >= 1

    def test_get_stats(self):
        stats = self.engine.stats()
        assert "total_sessions" in stats
        assert "enabled_tools" in stats

    def test_multiple_sessions(self):
        s1 = self.engine.new_session()
        s2 = self.engine.new_session()
        assert s1 != s2
        stats = self.engine.stats()
        assert stats["total_sessions"] >= 2

    def test_engine_config(self):
        assert self.engine.config is not None

    def test_chat_stream_final_chunk(self):
        chunks = list(self.engine.chat_stream("Hello", session_id="test_final"))
        final = [c for c in chunks if c.final]
        assert len(final) == 1
