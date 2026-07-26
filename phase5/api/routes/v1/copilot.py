from __future__ import annotations

from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from phase5.api.dependencies.auth import require_read
from phase5.api.dependencies.engine import get_knowledge_dir
from phase5.api.responses.envelope import build_success, build_error
from phase5.api.services.copilot import CopilotService

router = APIRouter(prefix="/copilot")


@router.post("/chat", summary="Send a chat message")
async def copilot_chat(
    request: Request,
    query: str = Query(default="", description="User query"),
    session_id: str | None = Query(default=None, description="Session ID for conversation context"),
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    if not query.strip():
        return JSONResponse(status_code=422, content=build_error("VALIDATION", "Query is required").model_dump(mode="json"))
    service = CopilotService(knowledge_dir)
    response = await service.chat(query, session_id)
    return JSONResponse(content=build_success(response).model_dump(mode="json"))


@router.post("/stream", summary="Stream a chat response via SSE")
async def copilot_stream(
    request: Request,
    query: str = Query(default="", description="User query"),
    session_id: str | None = Query(default=None, description="Session ID"),
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> StreamingResponse:
    if not query.strip():
        return StreamingResponse(content="data: {\"error\": \"Query is required\"}\n\n", media_type="text/event-stream")

    service = CopilotService(knowledge_dir)

    async def event_stream() -> AsyncGenerator[str, None]:
        async for event in service.stream(query, session_id):
            yield event

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/sessions", summary="Create a new chat session")
async def create_session(
    request: Request,
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = CopilotService(knowledge_dir)
    session_id = await service.new_session()
    return JSONResponse(content=build_success({"session_id": session_id}).model_dump(mode="json"))


@router.get("/sessions/{session_id}/history", summary="Get session history")
async def session_history(
    request: Request,
    session_id: str,
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = CopilotService(knowledge_dir)
    history = await service.get_session_history(session_id)
    return JSONResponse(content=build_success(history).model_dump(mode="json"))


@router.get("/stats", summary="Copilot statistics")
async def copilot_stats(
    request: Request,
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = CopilotService(knowledge_dir)
    stats = await service.stats()
    return JSONResponse(content=build_success(stats).model_dump(mode="json"))
