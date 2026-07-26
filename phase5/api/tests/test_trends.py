from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from phase5.api.config.settings import APISettings


@pytest.fixture
def client() -> TestClient:
    from phase5.api.app import create_app
    return TestClient(create_app(APISettings(environment="development", debug=False)))


def test_list_trends(client: TestClient):
    resp = client.get("/api/v1/trends")
    assert resp.status_code in (200, 401)


def test_trends_growing(client: TestClient):
    resp = client.get("/api/v1/trends/growing?top_k=5")
    assert resp.status_code in (200, 401)


def test_trends_emerging(client: TestClient):
    resp = client.get("/api/v1/trends/emerging?top_k=5")
    assert resp.status_code in (200, 401)


def test_trends_search(client: TestClient):
    resp = client.get("/api/v1/trends/search?query=AI")
    assert resp.status_code in (200, 401)


def test_trends_stats(client: TestClient):
    resp = client.get("/api/v1/trends/stats")
    assert resp.status_code in (200, 401)


def test_trend_by_id_not_found(client: TestClient):
    resp = client.get("/api/v1/trends/nonexistent")
    assert resp.status_code in (404, 401)
