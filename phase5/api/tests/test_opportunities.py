from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from phase5.api.config.settings import APISettings


@pytest.fixture
def client() -> TestClient:
    from phase5.api.app import create_app
    return TestClient(create_app(APISettings(environment="development", debug=False)))


def test_list_opportunities(client: TestClient):
    resp = client.get("/api/v1/opportunities")
    assert resp.status_code in (200, 401)
    if resp.status_code == 200:
        data = resp.json()
        assert "success" in data


def test_opportunities_top(client: TestClient):
    resp = client.get("/api/v1/opportunities/top?top_k=5")
    assert resp.status_code in (200, 401)


def test_opportunities_search(client: TestClient):
    resp = client.get("/api/v1/opportunities/search?query=AI")
    assert resp.status_code in (200, 401)


def test_opportunities_stats(client: TestClient):
    resp = client.get("/api/v1/opportunities/stats")
    assert resp.status_code in (200, 401)


def test_opportunity_by_id_not_found(client: TestClient):
    resp = client.get("/api/v1/opportunities/nonexistent")
    assert resp.status_code in (404, 401)


def test_opportunity_routes_exist(client: TestClient):
    resp = client.get("/api/v1/opportunities/top")
    assert resp.status_code in (200, 401, 404)
