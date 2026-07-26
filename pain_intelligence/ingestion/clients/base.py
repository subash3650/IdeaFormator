"""Base class and decorator wrappers for HTTP client abstraction."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any
import httpx
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()


class HttpClient(ABC):
    """Abstract Base Class for HTTP client implementations."""

    @abstractmethod
    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Execute a synchronous GET request."""

    @abstractmethod
    def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Execute a synchronous POST request."""

    @abstractmethod
    def close(self) -> None:
        """Close the underlying client connections."""


class RetryClient(HttpClient):
    """Decorator/wrapper client that adds exponential backoff retry logic."""

    def __init__(
        self,
        client: HttpClient,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        self.client = client
        self.max_retries = max_retries
        self.base_delay = base_delay

    def _retry_request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Execute a request with retry logic."""
        delay = self.base_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.post(url, **kwargs) if method == "post" else self.client.get(url, **kwargs)
                if response.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    logger.warning(
                        "HTTP {} for {} {}. Retrying in {:.1f}s... (Attempt {}/{})",
                        response.status_code,
                        method.upper(),
                        url,
                        delay,
                        attempt,
                        self.max_retries,
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                return response
            except (httpx.RequestError, httpx.TimeoutException) as e:
                if attempt == self.max_retries:
                    logger.error("HTTP {} failed after {} attempts: {}", method.upper(), self.max_retries, e)
                    raise
                logger.warning(
                    "Network error: {}. Retrying in {:.1f}s... (Attempt {}/{})",
                    e,
                    delay,
                    attempt,
                    self.max_retries,
                )
                time.sleep(delay)
                delay *= 2

        # Fallback
        if method == "post":
            return self.client.post(url, **kwargs)
        return self.client.get(url, **kwargs)

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return self._retry_request("get", url, params=params, headers=headers)

    def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return self._retry_request("post", url, json=json, headers=headers)

    def close(self) -> None:
        self.client.close()


class RateLimitedClient(HttpClient):
    """Decorator/wrapper client that adds token-bucket rate limiting."""

    def __init__(self, client: HttpClient, rate_limit: float = 1.0) -> None:
        """Initialize rate limiter.

        rate_limit: Requests per second.
        """
        self.client = client
        self.rate_limit = rate_limit
        self.last_request_time = 0.0

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        self._wait_if_needed()
        response = self.client.get(url, params=params, headers=headers)
        self.last_request_time = time.time()
        return response

    def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        self._wait_if_needed()
        response = self.client.post(url, json=json, headers=headers)
        self.last_request_time = time.time()
        return response

    def _wait_if_needed(self) -> None:
        """Enforce rate limiting before the next request."""
        if self.rate_limit > 0:
            min_interval = 1.0 / self.rate_limit
            elapsed = time.time() - self.last_request_time
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                logger.debug("Rate limiting: sleeping for {:.2f}s", sleep_time)
                time.sleep(sleep_time)

    def close(self) -> None:
        self.client.close()
