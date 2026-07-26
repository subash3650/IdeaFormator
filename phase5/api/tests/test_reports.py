from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from phase5.api.config.settings import APISettings


@pytest.fixture
def client() -> TestClient:
    from phase5.api.app import create_app
    return TestClient(create_app(APISettings(environment="development", debug=False)))


def test_list_reports(client: TestClient):
    resp = client.get("/api/v1/reports")
    assert resp.status_code in (200, 401)


def test_reports_search(client: TestClient):
    resp = client.get("/api/v1/reports/search?query=AI")
    assert resp.status_code in (200, 401)


def test_reports_stats(client: TestClient):
    resp = client.get("/api/v1/reports/stats")
    assert resp.status_code in (200, 401)


def test_report_by_id_not_found(client: TestClient):
    resp = client.get("/api/v1/reports/nonexistent")
    assert resp.status_code in (404, 401)


def test_report_generate(client: TestClient):
    resp = client.post("/api/v1/reports/generate?report_type=executive_summary")
    assert resp.status_code in (200, 401, 422)
