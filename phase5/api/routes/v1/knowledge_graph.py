from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from phase5.api.dependencies.auth import get_current_user, require_read
from phase5.api.dependencies.engine import get_knowledge_dir
from phase5.api.auth.models import UserContext
from phase5.api.responses.envelope import build_success
from phase5.api.schemas.knowledge_graph import GraphStatsResponse
from phase5.api.services.knowledge_graph import KGService

router = APIRouter(prefix="/knowledge-graph")


@router.get("/nodes", summary="Search knowledge graph nodes")
async def search_nodes(
    request: Request,
    query: str = Query(default="", description="Search query"),
    top_k: int = Query(default=10, ge=1, le=100),
    type_filter: str = Query(default="", description="Filter by node type"),
    user: UserContext = Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = KGService(knowledge_dir)
    results = await service.search(query=query, top_k=top_k, type_filter=type_filter)
    return JSONResponse(content=build_success(results).model_dump(mode="json"))


@router.get("/nodes/{node_id}", summary="Get a specific node")
async def get_node(
    request: Request,
    node_id: str,
    user: UserContext = Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = KGService(knowledge_dir)
    node = await service.get_node(node_id)
    if node is None:
        from fastapi.responses import JSONResponse
        from phase5.api.responses.envelope import build_error
        return JSONResponse(status_code=404, content=build_error("NOT_FOUND", f"Node {node_id} not found").model_dump(mode="json"))
    return JSONResponse(content=build_success(node).model_dump(mode="json"))


@router.get("/stats", summary="Knowledge graph statistics")
async def graph_stats(
    request: Request,
    user: UserContext = Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = KGService(knowledge_dir)
    stats = await service.stats()
    return JSONResponse(content=build_success(stats).model_dump(mode="json"))
