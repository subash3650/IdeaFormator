from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorCode(str, Enum):
    NOT_FOUND = "NOT_FOUND"
    VALIDATION = "VALIDATION"
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    RATE_LIMIT = "RATE_LIMIT"
    CONFLICT = "CONFLICT"
    INTERNAL = "INTERNAL"
    BAD_REQUEST = "BAD_REQUEST"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    UNSUPPORTED_MEDIA = "UNSUPPORTED_MEDIA"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"


class ErrorDetail(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: ErrorCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    trace_id: str = ""
    timestamp: str = ""


class ErrorResponse(BaseModel):
    model_config = ConfigDict(frozen=False, extra="forbid")

    success: bool = False
    error: ErrorDetail | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
