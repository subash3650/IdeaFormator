"""Tests for HTTP clients (base, retry, rate-limited)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from pain_intelligence.ingestion.clients.base import HttpClient, RateLimitedClient, RetryClient


class MockHttpClient(HttpClient):
    """A simple mock HTTP client for testing."""

    def __init__(self):
        self.call_count = 0

    def get(self, url, params=None, headers=None):
        self.call_count += 1
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"ok": True}
        return response

    def post(self, url, json=None, headers=None):
        self.call_count += 1
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"ok": True}
        return response

    def close(self):
        pass


class FailingHttpClient(HttpClient):
    """A mock client that fails on the first call, succeeds on the second."""

    def __init__(self):
        self.call_count = 0

    def get(self, url, params=None, headers=None):
        self.call_count += 1
        response = MagicMock()
        if self.call_count == 1:
            response.status_code = 503
        else:
            response.status_code = 200
            response.json.return_value = {"ok": True}
        return response

    def post(self, url, json=None, headers=None):
        return self.get(url)

    def close(self):
        pass


class TestRetryClient:
    def test_retry_success(self):
        inner = FailingHttpClient()
        client = RetryClient(inner, max_retries=3, base_delay=0.01)

        response = client.get("http://example.com")
        assert response.status_code == 200
        assert inner.call_count == 2

    def test_retry_exhaustion(self):
        inner = MagicMock(spec=HttpClient)
        response = MagicMock()
        response.status_code = 503
        inner.get.return_value = response

        client = RetryClient(inner, max_retries=2, base_delay=0.01)
        result = client.get("http://example.com")
        assert result.status_code == 503

    def test_close_propagates(self):
        inner = MockHttpClient()
        client = RetryClient(inner)
        client.close()
        # No assertion needed — just verify no exception


class TestRateLimitedClient:
    def test_rate_limiting(self):
        inner = MockHttpClient()
        client = RateLimitedClient(inner, rate_limit=100.0)  # Very fast for tests

        response = client.get("http://example.com")
        assert response.status_code == 200
        assert inner.call_count == 1

    def test_close_propagates(self):
        inner = MockHttpClient()
        client = RateLimitedClient(inner)
        client.close()
