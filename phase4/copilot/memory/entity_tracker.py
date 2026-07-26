from __future__ import annotations

from typing import Any


class EntityTracker:
    def __init__(self, max_entities: int = 100) -> None:
        self._max = max_entities
        self._entities: list[dict[str, Any]] = []

    def track(self, entity_id: str, entity_type: str, label: str, metadata: dict[str, Any] | None = None) -> None:
        existing = [e for e in self._entities if e["id"] == entity_id]
        if existing:
            existing[0]["mention_count"] = existing[0].get("mention_count", 0) + 1
            existing[0]["last_mentioned"] = __import__("time").time()
            return
        self._entities.append({
            "id": entity_id,
            "type": entity_type,
            "label": label,
            "mention_count": 1,
            "first_mentioned": __import__("time").time(),
            "last_mentioned": __import__("time").time(),
            "metadata": metadata or {},
        })
        if len(self._entities) > self._max:
            self._entities = self._entities[-self._max:]

    def get_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        sorted_entities = sorted(
            self._entities,
            key=lambda e: e["last_mentioned"],
            reverse=True,
        )
        return sorted_entities[:limit]

    def get_by_type(self, entity_type: str) -> list[dict[str, Any]]:
        return [e for e in self._entities if e["type"] == entity_type]

    def get_by_id(self, entity_id: str) -> dict[str, Any] | None:
        for e in self._entities:
            if e["id"] == entity_id:
                return e
        return None

    def search(self, query: str) -> list[dict[str, Any]]:
        q = query.lower()
        return [
            e for e in self._entities
            if q in e["label"].lower() or q in e["id"].lower()
        ]

    def is_mentioned(self, entity_id: str) -> bool:
        return any(e["id"] == entity_id for e in self._entities)

    def clear(self) -> None:
        self._entities.clear()

    def count(self) -> int:
        return len(self._entities)

    def all_entities(self) -> list[dict[str, Any]]:
        return list(self._entities)
