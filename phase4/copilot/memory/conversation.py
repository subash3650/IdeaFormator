from __future__ import annotations

from typing import Any

from phase4.copilot.config import CopilotConfig
from phase4.copilot.schema import (
    ConversationMemoryState,
    MemoryEntry,
    MemoryLevel,
    Message,
    Role,
)


class ConversationMemory:
    def __init__(self, config: CopilotConfig) -> None:
        self._config = config
        self._state = ConversationMemoryState()

    @property
    def state(self) -> ConversationMemoryState:
        return self._state

    def add_message(self, message: Message) -> None:
        entry = MemoryEntry(
            role=message.role,
            content=message.content,
            level=MemoryLevel.SHORT_TERM,
            metadata={
                "message_id": message.message_id,
                "tool_calls": len(message.tool_calls),
                "citations": len(message.citations),
            },
        )
        self._state.short_term.append(entry)

        conv_entry = MemoryEntry(
            role=message.role,
            content=message.content,
            level=MemoryLevel.CONVERSATION,
            metadata=entry.metadata,
        )
        self._state.conversation.append(conv_entry)

        self._trim()
        self._check_compression()

    def add_system(self, content: str) -> None:
        entry = MemoryEntry(
            role=Role.SYSTEM,
            content=content,
            level=MemoryLevel.CONVERSATION,
        )
        self._state.conversation.append(entry)

    def pin(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        entry = MemoryEntry(
            role=Role.SYSTEM,
            content=content,
            level=MemoryLevel.PINNED,
            metadata=metadata or {},
        )
        self._state.pinned.append(entry)
        if len(self._state.pinned) > self._config.max_pinned:
            self._state.pinned = self._state.pinned[-self._config.max_pinned :]

    def promote_to_long_term(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        entry = MemoryEntry(
            role=Role.ASSISTANT,
            content=content,
            level=MemoryLevel.LONG_TERM,
            metadata=metadata or {},
        )
        self._state.long_term.append(entry)
        if len(self._state.long_term) > self._config.max_long_term:
            self._state.long_term = self._state.long_term[-self._config.max_long_term :]

    def get_recent(self, limit: int = 10) -> list[MemoryEntry]:
        combined = list(self._state.pinned)
        combined.extend(self._state.short_term)
        return combined[-limit:]

    def get_full_context(self) -> list[MemoryEntry]:
        result: list[MemoryEntry] = []
        result.extend(self._state.pinned)
        if self._state.compressed_summary:
            result.append(MemoryEntry(
                role=Role.SYSTEM,
                content=f"[Compressed summary: {self._state.compressed_summary}]",
                level=MemoryLevel.LONG_TERM,
            ))
        result.extend(self._state.long_term[-10:])
        result.extend(self._state.conversation)
        return result

    def get_history(self, limit: int | None = None) -> list[MemoryEntry]:
        entries = list(self._state.conversation)
        if limit is not None:
            entries = entries[-limit:]
        return entries

    def get_last_user_message(self) -> MemoryEntry | None:
        for entry in reversed(self._state.conversation):
            if entry.role == Role.USER:
                return entry
        return None

    def get_last_assistant_message(self) -> MemoryEntry | None:
        for entry in reversed(self._state.conversation):
            if entry.role == Role.ASSISTANT:
                return entry
        return None

    def message_count(self) -> int:
        return len(self._state.conversation)

    def clear(self) -> None:
        self._state = ConversationMemoryState()

    def compress(self) -> None:
        if not self._state.conversation:
            return
        total_chars = sum(len(e.content) for e in self._state.conversation)
        entry_count = len(self._state.conversation)
        self._state.compressed_summary = (
            f"{entry_count} messages, ~{total_chars} characters total. "
            f"Last topic: {self._state.conversation[-1].content[:100] if self._state.conversation else 'none'}"
        )
        keep_count = min(20, len(self._state.conversation) // 2)
        self._state.conversation = self._state.conversation[-keep_count:]
        self._state.needs_compression = False

    def _trim(self) -> None:
        if len(self._state.short_term) > self._config.max_short_term:
            self._state.short_term = self._state.short_term[-self._config.max_short_term :]
        if len(self._state.conversation) > self._config.max_conversation:
            overflow = len(self._state.conversation) - self._config.max_conversation
            for entry in self._state.conversation[:overflow]:
                promoted = MemoryEntry(
                    role=entry.role,
                    content=entry.content,
                    level=MemoryLevel.LONG_TERM,
                    metadata=entry.metadata,
                )
                self._state.long_term.append(promoted)
            self._state.conversation = self._state.conversation[overflow:]
            if len(self._state.long_term) > self._config.max_long_term:
                self._state.long_term = self._state.long_term[-self._config.max_long_term :]

    def _check_compression(self) -> None:
        if len(self._state.conversation) >= self._config.compression_threshold:
            self._state.needs_compression = True
