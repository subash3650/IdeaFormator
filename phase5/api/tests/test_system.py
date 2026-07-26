from __future__ import annotations

from fastapi.testclient import TestClient


class TestSystem:
    def test_system_info(self, client: TestClient) -> None:
        resp = client.get("/api/v1/system/info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["version"] == "4.0.0"
        assert "knowledge_graph" in body["data"]["modules"]

    def test_system_version(self, client: TestClient) -> None:
        resp = client.get("/api/v1/system/version")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["version"] == "4.0.0"
        assert body["data"]["schema_version"] == "4.0"

    def test_system_capabilities(self, client: TestClient) -> None:
        resp = client.get("/api/v1/system/capabilities")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["capabilities"]["knowledge_graph"] is True
        assert body["data"]["capabilities"]["streaming"] is True
        assert "json" in body["data"]["exports"]
