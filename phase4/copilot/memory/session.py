from __future__ import annotations

import time
from typing import Any

from phase4.copilot.config import CopilotConfig
from phase4.copilot.schema import ConversationState, SessionInfo, SessionStatus
from phase4.copilot.memory.conversation import ConversationMemory
from phase4.copilot.memory.entity_tracker import EntityTracker


class SessionManager:
    def __init__(self, config: CopilotConfig) -> None:
        self._config = config
        self._sessions: dict[str, _SessionData] = {}

    def create_session(self, session_id: str | None = None) -> tuple[SessionInfo, ConversationMemory, EntityTracker, ConversationState]:
        from phase4.copilot.schema import _now_iso, _compute_id

        if session_id is None:
            session_id = _compute_id("sess", str(time.time_ns()))
        now = _now_iso()

        info = SessionInfo(
            session_id=session_id,
            created_at=now,
            last_active_at=now,
            message_count=0,
            status=SessionStatus.ACTIVE,
        )

        state = ConversationState(session_id=session_id)
        memory = ConversationMemory(self._config)
        tracker = EntityTracker(self._config.max_entity_history)

        self._sessions[session_id] = _SessionData(
            info=info,
            state=state,
            memory=memory,
            tracker=tracker,
        )
        return info, memory, tracker, state

    def get_session(self, session_id: str) -> _SessionData | None:
        data = self._sessions.get(session_id)
        if data is None:
            return None
        if data.info.status != SessionStatus.ACTIVE:
            return None
        if self._is_expired(data):
            data.info = data.info.model_copy(update={"status": SessionStatus.EXPIRED})
            return None
        return data

    def get_or_create(self, session_id: str | None) -> tuple[SessionInfo, ConversationMemory, EntityTracker, ConversationState]:
        if session_id is not None:
            data = self.get_session(session_id)
            if data is not None:
                self._touch(data)
                return data.info, data.memory, data.tracker, data.state
        return self.create_session(session_id)

    def close_session(self, session_id: str) -> bool:
        data = self._sessions.get(session_id)
        if data is None:
            return False
        data.info = data.info.model_copy(update={"status": SessionStatus.CLOSED})
        return True

    def list_active(self) -> list[SessionInfo]:
        now = time.time()
        result: list[SessionInfo] = []
        for sid, data in self._sessions.items():
            if data.info.status == SessionStatus.ACTIVE:
                if not self._is_expired(data):
                    result.append(data.info)
                else:
                    data.info = data.info.model_copy(update={"status": SessionStatus.EXPIRED})
        return result

    def cleanup_expired(self) -> int:
        now = time.time()
        expired_ids = [
            sid for sid, data in self._sessions.items()
            if self._is_expired(data) or data.info.status == SessionStatus.EXPIRED
        ]
        for sid in expired_ids:
            del self._sessions[sid]
        return len(expired_ids)

    def session_count(self) -> int:
        return len(self._sessions)

    def get_state(self, session_id: str) -> ConversationState | None:
        data = self._sessions.get(session_id)
        if data is None:
            return None
        return data.state

    def update_state(self, session_id: str, state: ConversationState) -> None:
        data = self._sessions.get(session_id)
        if data is not None:
            data.state = state

    def _touch(self, data: _SessionData) -> None:
        from phase4.copilot.schema import _now_iso
        data.info = data.info.model_copy(update={"last_active_at": _now_iso(), "message_count": data.memory.message_count()})

    def _is_expired(self, data: _SessionData) -> bool:
        from datetime import datetime, timezone
        try:
            last = datetime.fromisoformat(data.info.last_active_at)
            now = datetime.now(timezone.utc)
            elapsed = (now - last).total_seconds() / 60
            return elapsed > self._config.session_ttl_minutes
        except Exception:
            return False


class _SessionData:
    def __init__(
        self,
        info: SessionInfo,
        state: ConversationState,
        memory: ConversationMemory,
        tracker: EntityTracker,
    ) -> None:
        self.info = info
        self.state = state
        self.memory = memory
        self.tracker = tracker
