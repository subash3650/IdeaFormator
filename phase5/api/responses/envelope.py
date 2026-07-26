from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ResponseMeta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = ""
    timestamp: str = ""
    duration_ms: float = 0.0
    api_version: str = "v1"


class PageMeta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    total: int | None = None
    has_next: bool = False
    next_cursor: str | None = None
    limit: int = 20
    offset: int = 0


class DataEnvelope(BaseModel, Generic[T]):
    model_config = ConfigDict(frozen=False, extra="forbid")

    success: bool = True
    data: T | None = None
    error: dict[str, Any] | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class PaginatedResponse(BaseModel, Generic[T]):
    model_config = ConfigDict(frozen=False, extra="forbid")

    success: bool = True
    data: list[T] = Field(default_factory=list)
    meta: ResponseMeta = Field(default_factory=ResponseMeta)
    page: PageMeta = Field(default_factory=PageMeta)


def build_success(data: Any, request_id: str = "", duration_ms: float = 0.0) -> DataEnvelope:
    return DataEnvelope(
        success=True,
        data=data,
        meta=ResponseMeta(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_ms=duration_ms,
            api_version="v1",
        ),
    )


def build_error(
    code: str,
    message: str,
    status_code: int = 500,
    details: dict[str, Any] | None = None,
    request_id: str = "",
    duration_ms: float = 0.0,
) -> DataEnvelope:
    return DataEnvelope(
        success=False,
        error={"code": code, "message": message, "details": details or {}},
        meta=ResponseMeta(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_ms=duration_ms,
        ),
    )
