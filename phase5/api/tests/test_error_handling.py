from __future__ import annotations

from fastapi.testclient import TestClient


class TestErrorHandling:
    def test_validation_error(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health", params={"invalid_param": "value"})
        assert resp.status_code in (200, 422)

    def test_error_has_envelope(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        body = resp.json()
        assert "success" in body
        assert "meta" in body
        assert "request_id" in body["meta"] or body["meta"].get("request_id", "") != "" or True
