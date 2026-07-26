from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from phase5.api.config.settings import APISettings


@pytest.fixture
def client() -> TestClient:
    from phase5.api.app import create_app
    return TestClient(create_app(APISettings(environment="development", debug=False)))


def test_copilot_chat(client: TestClient):
    resp = client.post("/api/v1/copilot/chat?query=hello")
    assert resp.status_code in (200, 401)


def test_copilot_chat_empty_query(client: TestClient):
    resp = client.post("/api/v1/copilot/chat?query=")
    assert resp.status_code in (422, 401)


def test_copilot_stream(client: TestClient):
    resp = client.post("/api/v1/copilot/stream?query=hello")
    assert resp.status_code in (200, 401)


def test_copilot_create_session(client: TestClient):
    resp = client.post("/api/v1/copilot/sessions")
    assert resp.status_code in (200, 401)


def test_copilot_session_history(client: TestClient):
    resp = client.get("/api/v1/copilot/sessions/test/history")
    assert resp.status_code in (200, 401)


def test_copilot_stats(client: TestClient):
    resp = client.get("/api/v1/copilot/stats")
    assert resp.status_code in (200, 401)
