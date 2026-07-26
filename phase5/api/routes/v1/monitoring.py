from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from phase5.api.dependencies.auth import require_admin
from phase5.api.responses.envelope import build_success
from phase5.api.services.cache import get_cache
from phase5.api.services.background import get_job_manager

router = APIRouter(prefix="/monitoring")


@router.get("/cache", summary="Cache statistics")
async def cache_stats(
    request: Request,
    user=Depends(require_admin),
) -> JSONResponse:
    cache = get_cache()
    return JSONResponse(content=build_success(cache.stats()).model_dump(mode="json"))


@router.post("/cache/clear", summary="Clear cache")
async def clear_cache(
    request: Request,
    user=Depends(require_admin),
) -> JSONResponse:
    cache = get_cache()
    cache.clear()
    return JSONResponse(content=build_success({"cleared": True}).model_dump(mode="json"))


@router.get("/jobs", summary="List background jobs")
async def list_jobs(
    request: Request,
    user=Depends(require_admin),
) -> JSONResponse:
    jm = get_job_manager()
    jobs = jm.list_jobs()
    return JSONResponse(content=build_success(jobs).model_dump(mode="json"))


@router.get("/ping", summary="Ping monitoring endpoint")
async def ping(
    request: Request,
) -> JSONResponse:
    return JSONResponse(content=build_success({"timestamp": time.time()}).model_dump(mode="json"))
