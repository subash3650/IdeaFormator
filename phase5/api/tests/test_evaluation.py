from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from phase5.api.config.settings import APISettings


@pytest.fixture
def client() -> TestClient:
    from phase5.api.app import create_app
    return TestClient(create_app(APISettings(environment="development", debug=False)))


def test_evaluation_stats(client: TestClient):
    resp = client.get("/api/v1/evaluation/stats")
    assert resp.status_code in (200, 401)


def test_evaluation_benchmark_no_path(client: TestClient):
    resp = client.post("/api/v1/evaluation/benchmark")
    assert resp.status_code in (422, 401)


def test_evaluation_intents_no_cases(client: TestClient):
    resp = client.post("/api/v1/evaluation/intents", json=[])
    assert resp.status_code in (422, 401)
