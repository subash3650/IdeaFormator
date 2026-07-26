from __future__ import annotations

from fastapi.testclient import TestClient


class TestReasoningRoutes:
    def test_reasoning_stats(self, client: TestClient) -> None:
        resp = client.get("/api/v1/reasoning/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True

    def test_reasoning_inferences(self, client: TestClient) -> None:
        resp = client.get("/api/v1/reasoning/inferences")
        assert resp.status_code == 200

    def test_reasoning_chains(self, client: TestClient) -> None:
        resp = client.get("/api/v1/reasoning/chains")
        assert resp.status_code == 200

    def test_reasoning_root_causes(self, client: TestClient) -> None:
        resp = client.get("/api/v1/reasoning/root-causes")
        assert resp.status_code == 200

    def test_reasoning_evidence(self, client: TestClient) -> None:
        resp = client.get("/api/v1/reasoning/evidence")
        assert resp.status_code == 200
