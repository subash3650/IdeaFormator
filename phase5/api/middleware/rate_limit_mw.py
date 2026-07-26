from __future__ import annotations

import time
from collections import defaultdict
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from phase5.api.exceptions.base import RateLimitError


class InMemoryRateLimiter:
    def __init__(self, requests_per_minute: int = 100) -> None:
        self._requests_per_minute = requests_per_minute
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str) -> None:
        now = time.time()
        window = 60.0
        self._buckets[key] = [t for t in self._buckets[key] if now - t < window]
        if len(self._buckets[key]) >= self._requests_per_minute:
            raise RateLimitError(details={"retry_after_seconds": 60})
        self._buckets[key].append(now)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Callable, requests_per_minute: int = 100) -> None:
        super().__init__(app)
        self._limiter = InMemoryRateLimiter(requests_per_minute)

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        key = request.client.host if request.client else "unknown"
        self._limiter.check(key)
        return await call_next(request)
