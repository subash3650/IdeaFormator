from __future__ import annotations

import time

from phase4.copilot.config import CopilotConfig
from phase4.copilot.memory.conversation import ConversationMemory
from phase4.copilot.memory.entity_tracker import EntityTracker
from phase4.copilot.memory.session import SessionManager
from phase4.copilot.schema import Message, Role, SessionStatus


class TestConversationMemory:
    def setup_method(self):
        self.memory = ConversationMemory(CopilotConfig())

    def test_add_message(self):
        msg = Message(role=Role.USER, content="hello")
        self.memory.add_message(msg)
        assert self.memory.message_count() == 1

    def test_get_last_user_message(self):
        self.memory.add_message(Message(role=Role.ASSISTANT, content="hi"))
        self.memory.add_message(Message(role=Role.USER, content="hello"))
        last = self.memory.get_last_user_message()
        assert last is not None
        assert last.content == "hello"

    def test_get_last_assistant_message(self):
        self.memory.add_message(Message(role=Role.USER, content="hello"))
        self.memory.add_message(Message(role=Role.ASSISTANT, content="hi"))
        last = self.memory.get_last_assistant_message()
        assert last is not None
        assert last.content == "hi"

    def test_message_count_starts_at_zero(self):
        assert self.memory.message_count() == 0

    def test_add_system(self):
        self.memory.add_system("System message")
        assert self.memory.message_count() == 1

    def test_pin(self):
        self.memory.pin("Important info")
        recent = self.memory.get_recent(10)
        pinned = [e for e in recent if e.level.value == "pinned"]
        assert len(pinned) >= 1

    def test_promote_to_long_term(self):
        self.memory.promote_to_long_term("Long term info")
        ctx = self.memory.get_full_context()
        long_term = [e for e in ctx if e.level.value == "long_term"]
        assert len(long_term) >= 1

    def test_get_recent_limit(self):
        for i in range(10):
            self.memory.add_message(Message(role=Role.USER, content=f"msg_{i}"))
        recent = self.memory.get_recent(3)
        assert len(recent) <= 3

    def test_get_history(self):
        for i in range(5):
            self.memory.add_message(Message(role=Role.USER, content=f"msg_{i}"))
        history = self.memory.get_history()
        assert len(history) == 5

    def test_clear(self):
        self.memory.add_message(Message(role=Role.USER, content="hello"))
        self.memory.clear()
        assert self.memory.message_count() == 0

    def test_compress(self):
        for i in range(30):
            self.memory.add_message(Message(role=Role.USER, content=f"msg_{i}"))
        self.memory.compress()
        assert len(self.memory.state.conversation) <= 20
        assert self.memory.state.compressed_summary


class TestEntityTracker:
    def setup_method(self):
        self.tracker = EntityTracker(max_entities=100)

    def test_track_new_entity(self):
        self.tracker.track("e1", "company", "TestCorp")
        assert self.tracker.count() == 1

    def test_track_existing_entity_increments(self):
        self.tracker.track("e1", "company", "TestCorp")
        self.tracker.track("e1", "company", "TestCorp")
        entity = self.tracker.get_by_id("e1")
        assert entity is not None
        assert entity["mention_count"] == 2

    def test_get_by_type(self):
        self.tracker.track("e1", "company", "CorpA")
        self.tracker.track("e2", "product", "ProductX")
        companies = self.tracker.get_by_type("company")
        assert len(companies) == 1

    def test_get_by_id_missing(self):
        assert self.tracker.get_by_id("nonexistent") is None

    def test_is_mentioned(self):
        self.tracker.track("e1", "company", "TestCorp")
        assert self.tracker.is_mentioned("e1")
        assert not self.tracker.is_mentioned("e2")

    def test_clear(self):
        self.tracker.track("e1", "company", "TestCorp")
        self.tracker.clear()
        assert self.tracker.count() == 0

    def test_search(self):
        self.tracker.track("e1", "company", "OpenAI")
        results = self.tracker.search("open")
        assert len(results) == 1

    def test_get_recent(self):
        self.tracker.track("e1", "company", "CorpA")
        self.tracker.track("e2", "product", "ProdX")
        recent = self.tracker.get_recent(1)
        assert len(recent) == 1


class TestSessionManager:
    def setup_method(self):
        self.manager = SessionManager(CopilotConfig())

    def test_create_session(self):
        info, memory, tracker, state = self.manager.create_session()
        assert info.session_id
        assert info.status == SessionStatus.ACTIVE

    def test_get_or_create_existing(self):
        info, mem, track, state = self.manager.get_or_create(None)
        assert info.session_id is not None

    def test_get_session_nonexistent(self):
        assert self.manager.get_session("nonexistent") is None

    def test_close_session(self):
        info, mem, track, state = self.manager.create_session()
        result = self.manager.close_session(info.session_id)
        assert result is True

    def test_close_session_nonexistent(self):
        assert not self.manager.close_session("nonexistent")

    def test_list_active(self):
        self.manager.create_session()
        assert len(self.manager.list_active()) >= 1

    def test_cleanup_expired(self):
        assert self.manager.cleanup_expired() >= 0

    def test_session_count(self):
        before = self.manager.session_count()
        self.manager.create_session()
        assert self.manager.session_count() == before + 1

    def test_update_state(self):
        info, mem, track, state = self.manager.get_or_create(None)
        state.turn_count = 5
        self.manager.update_state(info.session_id, state)
        loaded = self.manager.get_state(info.session_id)
        assert loaded is not None
        assert loaded.turn_count == 5
