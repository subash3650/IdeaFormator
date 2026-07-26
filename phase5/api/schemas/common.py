from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class PaginationParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cursor: str | None = None
    offset: int = 0
    limit: int = Field(default=20, ge=1, le=100)


class SortParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sort_by: str = "created_at"
    sort_order: str = "desc"


class FilterParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    search: str | None = None
    type_filter: str | None = None
    status_filter: str | None = None
    date_from: str | None = None
    date_to: str | None = None


class PageMeta(BaseModel, Generic[T]):
    model_config = ConfigDict(frozen=True, extra="forbid")

    total: int | None = None
    has_next: bool = False
    next_cursor: str | None = None
    limit: int = 20
    offset: int = 0


class ResponseMeta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = ""
    timestamp: str = ""
    duration_ms: float = 0.0
    api_version: str = "v1"


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
