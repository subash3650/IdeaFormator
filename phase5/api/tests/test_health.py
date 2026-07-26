from __future__ import annotations

from fastapi.testclient import TestClient


class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["status"] == "ok"

    def test_health_has_version(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        body = resp.json()
        assert body["data"]["version"] == "4.0.0"

    def test_liveness(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health/live")
        assert resp.status_code == 200

    def test_readiness(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["ready"] is True
        assert "knowledge_graph" in body["data"]["modules"]
