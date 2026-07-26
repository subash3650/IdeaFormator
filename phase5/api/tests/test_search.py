from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from phase5.api.config.settings import APISettings


@pytest.fixture
def client() -> TestClient:
    from phase5.api.app import create_app
    return TestClient(create_app(APISettings(environment="development", debug=False)))


def test_search_all(client: TestClient):
    resp = client.post("/api/v1/search?query=AI")
    assert resp.status_code in (200, 401)


def test_search_specific_modules(client: TestClient):
    resp = client.post("/api/v1/search?query=AI&modules=kg,opportunity")
    assert resp.status_code in (200, 401)


def test_search_no_query(client: TestClient):
    resp = client.post("/api/v1/search?query=")
    assert resp.status_code in (200, 401)
