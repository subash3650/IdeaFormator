from __future__ import annotations

import json
from typing import Any, AsyncGenerator

from pydantic import BaseModel, ConfigDict


class SSEEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event: str = "message"
    data: dict[str, Any] = {}


class SSEStreamer:
    @staticmethod
    def format(event: str, data: dict[str, Any]) -> str:
        payload = json.dumps(data)
        return f"event: {event}\ndata: {payload}\n\n"

    @staticmethod
    def format_done(session_id: str = "") -> str:
        return SSEStreamer.format("done", {"session_id": session_id})

    @staticmethod
    def format_error(message: str, code: str = "ERROR") -> str:
        return SSEStreamer.format("error", {"code": code, "message": message})

    @staticmethod
    def format_token(token: str, index: int = 0) -> str:
        return SSEStreamer.format("token", {"token": token, "index": index})

    @staticmethod
    def format_citations(citations: list[dict[str, Any]]) -> str:
        return SSEStreamer.format("citations", {"citations": citations})

    @classmethod
    async def stream_tokens(cls, tokens: list[str]) -> AsyncGenerator[str, None]:
        for i, token in enumerate(tokens):
            yield cls.format_token(token, i)

    @classmethod
    async def stream_final(cls, session_id: str = "") -> AsyncGenerator[str, None]:
        yield cls.format_done(session_id)
