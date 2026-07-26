from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from phase5.api.dependencies.auth import require_read
from phase5.api.dependencies.engine import get_knowledge_dir
from phase5.api.responses.envelope import build_success
from phase5.api.services.reasoning import ReasoningService

router = APIRouter(prefix="/reasoning")


@router.get("/inferences", summary="List all inferences")
async def list_inferences(
    request: Request,
    knowledge_dir: Path = Depends(get_knowledge_dir),
    user=Depends(require_read),
) -> JSONResponse:
    service = ReasoningService(knowledge_dir)
    inferences = await service.get_inferences()
    return JSONResponse(content=build_success(inferences).model_dump(mode="json"))


@router.get("/chains", summary="List all reasoning chains")
async def list_chains(
    request: Request,
    knowledge_dir: Path = Depends(get_knowledge_dir),
    user=Depends(require_read),
) -> JSONResponse:
    service = ReasoningService(knowledge_dir)
    chains = await service.get_chains()
    return JSONResponse(content=build_success(chains).model_dump(mode="json"))


@router.get("/root-causes", summary="List root causes")
async def list_root_causes(
    request: Request,
    knowledge_dir: Path = Depends(get_knowledge_dir),
    user=Depends(require_read),
) -> JSONResponse:
    service = ReasoningService(knowledge_dir)
    causes = await service.get_root_causes()
    return JSONResponse(content=build_success(causes).model_dump(mode="json"))


@router.get("/evidence", summary="List evidence aggregations")
async def list_evidence(
    request: Request,
    knowledge_dir: Path = Depends(get_knowledge_dir),
    user=Depends(require_read),
) -> JSONResponse:
    service = ReasoningService(knowledge_dir)
    evidence = await service.get_evidence()
    return JSONResponse(content=build_success(evidence).model_dump(mode="json"))


@router.get("/stats", summary="Reasoning statistics")
async def reasoning_stats(
    request: Request,
    knowledge_dir: Path = Depends(get_knowledge_dir),
    user=Depends(require_read),
) -> JSONResponse:
    service = ReasoningService(knowledge_dir)
    stats = await service.stats()
    return JSONResponse(content=build_success(stats).model_dump(mode="json"))
