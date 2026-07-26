from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from phase5.api.config.settings import APISettings


@pytest.fixture
def client() -> TestClient:
    from phase5.api.app import create_app
    return TestClient(create_app(APISettings(environment="development", debug=False)))


def test_config_get(client: TestClient):
    resp = client.get("/api/v1/config")
    assert resp.status_code in (200, 401)


def test_config_put(client: TestClient):
    resp = client.put("/api/v1/config", json={"key": "value"})
    assert resp.status_code in (200, 401)


def test_config_put_empty(client: TestClient):
    resp = client.put("/api/v1/config", json={})
    assert resp.status_code in (422, 401)
