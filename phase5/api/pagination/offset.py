from __future__ import annotations

from typing import Any


class OffsetPaginator:
    @staticmethod
    def paginate(items: list[Any], offset: int, limit: int) -> tuple[list[Any], int, bool]:
        total = len(items)
        end = offset + limit
        page = items[offset:end]
        has_next = end < total
        return page, total, has_next

    @staticmethod
    def paginate_with_cursor(items: list[Any], offset: int, limit: int) -> tuple[list[Any], int, bool, str | None]:
        page, total, has_next = OffsetPaginator.paginate(items, offset, limit)
        next_cursor = None
        if has_next and page:
            from phase5.api.pagination.cursor import CursorPaginator
            next_cursor = CursorPaginator.encode(page[-1])
        return page, total, has_next, next_cursor
