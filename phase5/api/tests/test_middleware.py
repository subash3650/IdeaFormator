from __future__ import annotations

from fastapi.testclient import TestClient


class TestMiddleware:
    def test_request_id_header(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert "X-Request-ID" in resp.headers
        assert len(resp.headers["X-Request-ID"]) > 0

    def test_response_time_header(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert "X-Response-Time" in resp.headers

    def test_security_headers(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert "Strict-Transport-Security" in resp.headers

    def test_cors_headers(self, client: TestClient) -> None:
        resp = client.options("/api/v1/health", headers={"Origin": "http://localhost"})
        assert "access-control-allow-origin" in resp.headers or resp.status_code in (200, 204)

    def test_404_returns_envelope(self, client: TestClient) -> None:
        resp = client.get("/api/v1/nonexistent")
        assert resp.status_code == 404
        body = resp.json()
        assert body["success"] is False
        assert "error" in body
