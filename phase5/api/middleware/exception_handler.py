from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from phase5.api.exceptions.base import APIError, ErrorCodeEnum
from phase5.api.responses.envelope import build_error

logger = logging.getLogger("api.error")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        rid = getattr(request.state, "request_id", "")
        duration = _get_duration(request)
        envelope = build_error(
            code=exc.code.value,
            message=exc.message,
            status_code=exc.status_code,
            details=exc.details,
            request_id=rid,
            duration_ms=duration,
        )
        _log_error(rid, exc)
        return JSONResponse(status_code=exc.status_code, content=envelope.model_dump(mode="json"))

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        rid = getattr(request.state, "request_id", "")
        duration = _get_duration(request)
        code = _status_to_code(exc.status_code)
        envelope = build_error(
            code=code,
            message=str(exc.detail) if exc.detail else "",
            status_code=exc.status_code,
            request_id=rid,
            duration_ms=duration,
        )
        _log_error(rid, exc)
        return JSONResponse(status_code=exc.status_code, content=envelope.model_dump(mode="json"))

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        rid = getattr(request.state, "request_id", "")
        duration = _get_duration(request)
        details: dict[str, Any] = {"errors": exc.errors()}
        envelope = build_error(
            code=ErrorCodeEnum.VALIDATION.value,
            message="Request validation failed",
            status_code=422,
            details=details,
            request_id=rid,
            duration_ms=duration,
        )
        _log_error(rid, exc)
        return JSONResponse(status_code=422, content=envelope.model_dump(mode="json"))

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        rid = getattr(request.state, "request_id", "")
        duration = _get_duration(request)
        envelope = build_error(
            code=ErrorCodeEnum.INTERNAL.value,
            message="Internal server error",
            status_code=500,
            request_id=rid,
            duration_ms=duration,
        )
        _log_error(rid, exc)
        return JSONResponse(status_code=500, content=envelope.model_dump(mode="json"))


def _status_to_code(status: int) -> str:
    if status == 404:
        return ErrorCodeEnum.NOT_FOUND.value
    if status == 403:
        return ErrorCodeEnum.AUTHORIZATION.value
    if status == 401:
        return ErrorCodeEnum.AUTHENTICATION.value
    if status == 409:
        return ErrorCodeEnum.CONFLICT.value
    if status == 429:
        return ErrorCodeEnum.RATE_LIMIT.value
    if status == 422:
        return ErrorCodeEnum.VALIDATION.value
    return ErrorCodeEnum.INTERNAL.value


def _get_duration(request: Request) -> float:
    start = getattr(request.state, "start_time", None)
    if start is not None:
        return (__import__("time").time() - start) * 1000
    return 0.0


def _log_error(rid: str, exc: Exception) -> None:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.error(json.dumps({"request_id": rid, "error": str(exc), "traceback": tb}))
