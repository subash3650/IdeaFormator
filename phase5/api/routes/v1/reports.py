from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from phase5.api.dependencies.auth import require_read
from phase5.api.dependencies.engine import get_knowledge_dir
from phase5.api.responses.envelope import build_success, build_error
from phase5.api.services.presentation import PresentationService

router = APIRouter(prefix="/reports")


@router.get("", summary="List reports")
async def list_reports(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = PresentationService(knowledge_dir)
    items = await service.list_reports(limit=limit)
    return JSONResponse(content=build_success(items).model_dump(mode="json"))


@router.post("/generate", summary="Generate a report")
async def generate_report(
    request: Request,
    report_type: str = Query(default="executive_summary", description="Report type"),
    template: str | None = Query(default=None, description="Template name"),
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = PresentationService(knowledge_dir)
    result = await service.generate(report_type, template)
    return JSONResponse(content=build_success(result).model_dump(mode="json"))


@router.get("/search", summary="Search reports")
async def search_reports(
    request: Request,
    query: str = Query(default=""),
    report_type: str = Query(default=""),
    tag: str = Query(default=""),
    company: str = Query(default=""),
    technology: str = Query(default=""),
    limit: int = Query(default=10, ge=1, le=50),
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = PresentationService(knowledge_dir)
    items = await service.search(query, report_type, tag, company, technology, limit)
    return JSONResponse(content=build_success(items).model_dump(mode="json"))


@router.get("/stats", summary="Report statistics")
async def report_stats(
    request: Request,
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = PresentationService(knowledge_dir)
    stats = await service.stats()
    return JSONResponse(content=build_success(stats).model_dump(mode="json"))


@router.get("/{report_id}", summary="Get report by ID")
async def get_report(
    request: Request,
    report_id: str,
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = PresentationService(knowledge_dir)
    report = await service.get_report(report_id)
    if report is None:
        return JSONResponse(status_code=404, content=build_error("NOT_FOUND", f"Report {report_id} not found").model_dump(mode="json"))
    return JSONResponse(content=build_success(report).model_dump(mode="json"))
