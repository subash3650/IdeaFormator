from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from phase5.api.config.settings import APISettings


@pytest.fixture
def client() -> TestClient:
    from phase5.api.app import create_app
    return TestClient(create_app(APISettings(environment="development", debug=False)))


def test_monitoring_ping(client: TestClient):
    resp = client.get("/api/v1/monitoring/ping")
    assert resp.status_code in (200, 401)


def test_monitoring_cache_stats(client: TestClient):
    resp = client.get("/api/v1/monitoring/cache")
    assert resp.status_code in (200, 401)


def test_monitoring_clear_cache(client: TestClient):
    resp = client.post("/api/v1/monitoring/cache/clear")
    assert resp.status_code in (200, 401)


def test_monitoring_jobs(client: TestClient):
    resp = client.get("/api/v1/monitoring/jobs")
    assert resp.status_code in (200, 401)
