from __future__ import annotations

import time
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


_global_metrics: dict[str, int | float] = {
    "total_requests": 0,
    "total_errors": 0,
    "total_duration_ms": 0.0,
}


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        _global_metrics["total_requests"] += 1
        _global_metrics["total_duration_ms"] += duration_ms
        if response.status_code >= 400:
            _global_metrics["total_errors"] += 1
        return response


def get_metrics() -> dict[str, int | float]:
    reqs = _global_metrics["total_requests"]
    avg = 0.0
    if reqs:
        avg = _global_metrics["total_duration_ms"] / reqs
    return {
        "total_requests": reqs,
        "total_errors": _global_metrics["total_errors"],
        "avg_duration_ms": round(avg, 1),
    }
