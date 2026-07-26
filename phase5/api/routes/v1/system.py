from __future__ import annotations

import sys

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from phase5.api.responses.envelope import build_success
from phase5.api.schemas.system import CapabilitiesResponse, SystemInfoResponse, VersionResponse

router = APIRouter()


@router.get("/system/info", summary="System information")
async def system_info(request: Request) -> JSONResponse:
    settings = request.app.state.settings
    data = SystemInfoResponse(
        version=settings.app_version,
        schema_version=settings.schema_version,
        pipeline_version=settings.pipeline_version,
        environment=settings.environment,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        modules=[
            "knowledge_graph", "reasoning", "opportunity", "trend",
            "presentation", "copilot", "embeddings", "similarity",
            "clustering", "evaluation",
        ],
    )
    return JSONResponse(content=build_success(data).model_dump(mode="json"))


@router.get("/system/version", summary="API version information")
async def version_endpoint(request: Request) -> JSONResponse:
    settings = request.app.state.settings
    data = VersionResponse(
        version=settings.app_version,
        schema_version=settings.schema_version,
        pipeline_version=settings.pipeline_version,
        build="",
    )
    return JSONResponse(content=build_success(data).model_dump(mode="json"))


@router.get("/system/capabilities", summary="API capabilities")
async def capabilities(request: Request) -> JSONResponse:
    data = CapabilitiesResponse(
        capabilities={
            "knowledge_graph": True,
            "reasoning": True,
            "opportunity": True,
            "trend": True,
            "presentation": True,
            "copilot": True,
            "streaming": True,
        },
        exports=["json", "markdown", "pdf", "docx", "pptx"],
    )
    return JSONResponse(content=build_success(data).model_dump(mode="json"))
