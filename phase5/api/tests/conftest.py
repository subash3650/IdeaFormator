from __future__ import annotations

from typing import AsyncGenerator

import pytest
from fastapi.testclient import TestClient

from phase5.api.app import create_app
from phase5.api.config.settings import APISettings


@pytest.fixture
def test_settings() -> APISettings:
    return APISettings(environment="development", debug=False)


@pytest.fixture
def client(test_settings: APISettings) -> TestClient:
    app = create_app(test_settings)
    return TestClient(app)
