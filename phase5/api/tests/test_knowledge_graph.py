from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


class TestKGRoutes:
    def test_kg_stats(self, client: TestClient) -> None:
        resp = client.get("/api/v1/knowledge-graph/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True

    def test_kg_search(self, client: TestClient) -> None:
        resp = client.get("/api/v1/knowledge-graph/nodes", params={"query": "AI", "top_k": 5})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True

    def test_kg_get_node_nonexistent(self, client: TestClient) -> None:
        resp = client.get("/api/v1/knowledge-graph/nodes/nonexistent-id")
        assert resp.status_code in (200, 404)

    def test_kg_search_with_type_filter(self, client: TestClient) -> None:
        resp = client.get("/api/v1/knowledge-graph/nodes", params={"query": "test", "type_filter": "company"})
        assert resp.status_code == 200
