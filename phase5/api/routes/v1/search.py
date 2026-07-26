from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from phase5.api.dependencies.auth import require_read
from phase5.api.dependencies.engine import get_knowledge_dir
from phase5.api.responses.envelope import build_success
from phase5.api.services.search import SearchService

router = APIRouter(prefix="/search")


@router.post("", summary="Cross-module search")
async def search_all(
    request: Request,
    query: str = Query(default="", description="Search query"),
    modules: str = Query(default="kg,opportunity,trend", description="Comma-separated modules"),
    top_k: int = Query(default=5, ge=1, le=50),
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    module_list = [m.strip() for m in modules.split(",") if m.strip()]
    service = SearchService(knowledge_dir)
    results = await service.search_all(query=query, modules=module_list, top_k=top_k)
    return JSONResponse(content=build_success(results).model_dump(mode="json"))
