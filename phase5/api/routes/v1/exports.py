from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from phase5.api.dependencies.auth import require_read
from phase5.api.dependencies.engine import get_knowledge_dir
from phase5.api.responses.envelope import build_success, build_error
from phase5.api.services.export_service import ExportService
from phase5.api.services.background import get_job_manager

router = APIRouter(prefix="/exports")


@router.post("/opportunities", summary="Export opportunities")
async def export_opportunities(
    request: Request,
    format: str = Query(default="report", description="Export format: report, csv, summary, dashboard"),
    background: bool = Query(default=False, description="Run as background job"),
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = ExportService(knowledge_dir)
    if background:
        jm = get_job_manager()
        job_id = jm.submit(
            lambda: service._run_in_thread(service.export_opportunities, format)
        )
        return JSONResponse(content=build_success({"job_id": job_id, "status": "pending"}).model_dump(mode="json"))
    result = await service.export_opportunities(format=format)
    return JSONResponse(content=build_success(result).model_dump(mode="json"))


@router.post("/trends", summary="Export trends")
async def export_trends(
    request: Request,
    format: str = Query(default="report", description="Export format: report, csv, summary, dashboard"),
    background: bool = Query(default=False),
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = ExportService(knowledge_dir)
    if background:
        jm = get_job_manager()
        job_id = jm.submit(
            lambda: service._run_in_thread(service.export_trends, format)
        )
        return JSONResponse(content=build_success({"job_id": job_id, "status": "pending"}).model_dump(mode="json"))
    result = await service.export_trends(format=format)
    return JSONResponse(content=build_success(result).model_dump(mode="json"))


@router.post("/reports", summary="Export reports")
async def export_reports(
    request: Request,
    report_id: str | None = Query(default=None),
    format: str = Query(default="json"),
    background: bool = Query(default=False),
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = ExportService(knowledge_dir)
    if background:
        jm = get_job_manager()
        job_id = jm.submit(
            lambda: service._run_in_thread(service.export_reports, report_id, format)
        )
        return JSONResponse(content=build_success({"job_id": job_id, "status": "pending"}).model_dump(mode="json"))
    result = await service.export_reports(report_id=report_id, format=format)
    return JSONResponse(content=build_success(result).model_dump(mode="json"))


@router.get("/stats", summary="Export service statistics")
async def export_stats(
    request: Request,
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = ExportService(knowledge_dir)
    stats = await service.stats()
    return JSONResponse(content=build_success(stats).model_dump(mode="json"))


@router.get("/jobs/{job_id}", summary="Check job status")
async def job_status(
    request: Request,
    job_id: str,
    user=Depends(require_read),
) -> JSONResponse:
    jm = get_job_manager()
    result = jm.get_result(job_id)
    if result is None:
        return JSONResponse(status_code=404, content=build_error("NOT_FOUND", f"Job {job_id} not found").model_dump(mode="json"))
    return JSONResponse(content=build_success(result).model_dump(mode="json"))
