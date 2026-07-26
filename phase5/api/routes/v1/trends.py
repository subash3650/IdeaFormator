from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from phase5.api.dependencies.auth import require_read
from phase5.api.dependencies.engine import get_knowledge_dir
from phase5.api.responses.envelope import build_success, build_error
from phase5.api.services.trend import TrendService

router = APIRouter(prefix="/trends")


@router.get("", summary="List all trends")
async def list_trends(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = TrendService(knowledge_dir)
    items = await service.list_all(limit=limit, offset=offset)
    return JSONResponse(content=build_success(items).model_dump(mode="json"))


@router.get("/growing", summary="Get growing trends")
async def growing_trends(
    request: Request,
    top_k: int = Query(default=10, ge=1, le=50),
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = TrendService(knowledge_dir)
    items = await service.growing(top_k=top_k)
    return JSONResponse(content=build_success(items).model_dump(mode="json"))


@router.get("/emerging", summary="Get emerging trends")
async def emerging_trends(
    request: Request,
    top_k: int = Query(default=10, ge=1, le=50),
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = TrendService(knowledge_dir)
    items = await service.emerging(top_k=top_k)
    return JSONResponse(content=build_success(items).model_dump(mode="json"))


@router.get("/search", summary="Search trends")
async def search_trends(
    request: Request,
    query: str = Query(default="", description="Search query"),
    top_k: int = Query(default=10, ge=1, le=100),
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = TrendService(knowledge_dir)
    results = await service.search(query, top_k)
    return JSONResponse(content=build_success(results).model_dump(mode="json"))


@router.get("/stats", summary="Trend statistics")
async def trend_stats(
    request: Request,
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = TrendService(knowledge_dir)
    stats = await service.stats()
    return JSONResponse(content=build_success(stats).model_dump(mode="json"))


@router.get("/{trend_id}", summary="Get trend by ID")
async def get_trend(
    request: Request,
    trend_id: str,
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = TrendService(knowledge_dir)
    item = await service.get_by_id(trend_id)
    if item is None:
        return JSONResponse(status_code=404, content=build_error("NOT_FOUND", f"Trend {trend_id} not found").model_dump(mode="json"))
    return JSONResponse(content=build_success(item).model_dump(mode="json"))
