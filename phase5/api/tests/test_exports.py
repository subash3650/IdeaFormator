from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from phase5.api.config.settings import APISettings


@pytest.fixture
def client() -> TestClient:
    from phase5.api.app import create_app
    return TestClient(create_app(APISettings(environment="development", debug=False)))


def test_export_opportunities(client: TestClient):
    resp = client.post("/api/v1/exports/opportunities?format=report")
    assert resp.status_code in (200, 401)


def test_export_trends(client: TestClient):
    resp = client.post("/api/v1/exports/trends?format=report")
    assert resp.status_code in (200, 401)


def test_export_reports(client: TestClient):
    resp = client.post("/api/v1/exports/reports")
    assert resp.status_code in (200, 401)


def test_export_stats(client: TestClient):
    resp = client.get("/api/v1/exports/stats")
    assert resp.status_code in (200, 401)


def test_export_job_status_not_found(client: TestClient):
    resp = client.get("/api/v1/exports/jobs/nonexistent")
    assert resp.status_code in (404, 401)
