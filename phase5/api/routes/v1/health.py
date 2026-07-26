from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from phase5.api.responses.envelope import build_success
from phase5.api.schemas.health import HealthResponse, ReadinessResponse

router = APIRouter()


@router.get("/health", summary="Health check")
async def health() -> JSONResponse:
    data = HealthResponse()
    return JSONResponse(content=build_success(data).model_dump(mode="json"))


@router.get("/health/live", summary="Liveness probe")
async def liveness() -> JSONResponse:
    return JSONResponse(content={"status": "alive"})


@router.get("/health/ready", summary="Readiness probe")
async def readiness(request: Request) -> JSONResponse:
    modules: dict[str, str] = {}
    for name in ("knowledge_graph", "reasoning", "opportunity", "trend", "presentation", "copilot"):
        modules[name] = "available"
    data = ReadinessResponse(ready=True, modules=modules)
    return JSONResponse(content=build_success(data).model_dump(mode="json"))
