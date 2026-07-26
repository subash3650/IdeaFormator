from __future__ import annotations

from typing import Any, Generator

from phase4.copilot.schema import Citation, CopilotResponse, StreamingChunk


class StreamingHandler:
    def __init__(self, buffer_size: int = 50) -> None:
        self._buffer_size = buffer_size
        self._index = 0

    def stream_text(self, text: str) -> Generator[StreamingChunk, None, None]:
        words = text.split()
        for word in words:
            yield StreamingChunk(
                chunk_type="token",
                data=word + " ",
                index=self._index,
                final=False,
            )
            self._index += 1

    def stream_citations(self, citations: list[Citation]) -> Generator[StreamingChunk, None, None]:
        if not citations:
            return
        lines = ["\n\n**Sources:**"]
        for i, c in enumerate(citations, 1):
            conf = f" (confidence: {c.confidence:.2f})" if c.confidence else ""
            lines.append(f"  [{i}] {c.source_title}{conf}")
        yield StreamingChunk(
            chunk_type="citations",
            data="\n".join(lines),
            index=self._index,
            final=False,
        )
        self._index += 1

    def finish(self, session_id: str = "", citations: list[Citation] | None = None, confidence: float = 1.0) -> StreamingChunk:
        resp = CopilotResponse(
            session_id=session_id,
            content="",
            citations=citations or [],
            confidence=confidence,
        )
        return StreamingChunk(
            chunk_type="done",
            data=resp.model_dump_json(),
            index=self._index,
            final=True,
        )
