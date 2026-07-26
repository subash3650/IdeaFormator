from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from phase5.api.dependencies.auth import require_read
from phase5.api.dependencies.engine import get_knowledge_dir
from phase5.api.responses.envelope import build_success, build_error
from phase5.api.services.opportunity import OpportunityService

router = APIRouter(prefix="/opportunities")


@router.get("", summary="List all opportunities")
async def list_opportunities(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = OpportunityService(knowledge_dir)
    items = await service.list_all(limit=limit, offset=offset)
    return JSONResponse(content=build_success(items).model_dump(mode="json"))


@router.get("/top", summary="Get top-ranked opportunities")
async def top_opportunities(
    request: Request,
    top_k: int = Query(default=10, ge=1, le=50),
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = OpportunityService(knowledge_dir)
    items = await service.get_top(top_k=top_k)
    return JSONResponse(content=build_success(items).model_dump(mode="json"))


@router.get("/search", summary="Search opportunities")
async def search_opportunities(
    request: Request,
    query: str = Query(default="", description="Search query"),
    top_k: int = Query(default=10, ge=1, le=100),
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = OpportunityService(knowledge_dir)
    results = await service.search(query, top_k)
    return JSONResponse(content=build_success(results).model_dump(mode="json"))


@router.get("/stats", summary="Opportunity statistics")
async def opportunity_stats(
    request: Request,
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = OpportunityService(knowledge_dir)
    stats = await service.stats()
    return JSONResponse(content=build_success(stats).model_dump(mode="json"))


@router.get("/{opportunity_id}", summary="Get opportunity by ID")
async def get_opportunity(
    request: Request,
    opportunity_id: str,
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = OpportunityService(knowledge_dir)
    item = await service.get_by_id(opportunity_id)
    if item is None:
        return JSONResponse(status_code=404, content=build_error("NOT_FOUND", f"Opportunity {opportunity_id} not found").model_dump(mode="json"))
    return JSONResponse(content=build_success(item).model_dump(mode="json"))
