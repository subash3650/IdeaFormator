from __future__ import annotations

from fastapi import Query

from phase5.api.pagination.params import PaginationParams


async def get_pagination_params(
    cursor: str | None = Query(default=None, description="Cursor for cursor-based pagination"),
    offset: int = Query(default=0, ge=0, description="Offset for offset-based pagination"),
    limit: int = Query(default=20, ge=1, le=100, description="Items per page"),
) -> PaginationParams:
    return PaginationParams(cursor=cursor, offset=offset, limit=limit)
