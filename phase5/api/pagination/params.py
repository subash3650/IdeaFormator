from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PaginationParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cursor: str | None = None
    offset: int = 0
    limit: int = Field(default=20, ge=1, le=100)
