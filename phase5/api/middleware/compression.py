from __future__ import annotations

from starlette.middleware.gzip import GZipMiddleware


def make_compression_middleware(minimum_size: int = 1000) -> dict:
    return {
        "middleware_class": GZipMiddleware,
        "minimum_size": minimum_size,
    }
