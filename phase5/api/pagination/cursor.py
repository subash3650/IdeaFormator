from __future__ import annotations

import base64
import json
from typing import Any


class CursorPaginator:
    @staticmethod
    def encode(item: dict[str, Any], key: str = "id") -> str:
        cursor_data = {key: item.get(key, "")}
        raw = json.dumps(cursor_data, sort_keys=True)
        return base64.urlsafe_b64encode(raw.encode()).decode()

    @staticmethod
    def decode(cursor: str) -> dict[str, Any]:
        try:
            raw = base64.urlsafe_b64decode(cursor.encode()).decode()
            return json.loads(raw)
        except Exception:
            return {}

    @staticmethod
    def has_next(items: list[Any], limit: int) -> bool:
        return len(items) > limit

    @staticmethod
    def trim(items: list[Any], limit: int) -> list[Any]:
        return items[:limit]
