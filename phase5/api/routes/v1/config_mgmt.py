from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from phase5.api.dependencies.auth import require_admin
from phase5.api.responses.envelope import build_success, build_error
from phase5.api.config.api_config import get_unified_config, update_unified_config

router = APIRouter(prefix="/config")


@router.get("", summary="Get runtime configuration")
async def get_config(
    request: Request,
    user=Depends(require_admin),
) -> JSONResponse:
    cfg = get_unified_config()
    return JSONResponse(content=build_success(cfg).model_dump(mode="json"))


@router.put("", summary="Update runtime configuration")
async def update_config(
    request: Request,
    updates: dict,
    user=Depends(require_admin),
) -> JSONResponse:
    if not updates:
        return JSONResponse(status_code=422, content=build_error("VALIDATION", "No updates provided").model_dump(mode="json"))
    result = update_unified_config(updates)
    return JSONResponse(content=build_success(result).model_dump(mode="json"))
